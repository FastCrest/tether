/**
 * Reflex contribution-worker — Cloudflare Worker for the Curate wedge.
 *
 * Endpoints:
 *   GET  /healthz                                → health probe
 *   POST /admin/init-bucket                      → confirm R2 bucket exists (admin auth)
 *   GET  /admin/contributors                     → list contributors + stats (admin auth)
 *   POST /admin/manual-purge                     → trigger cascade purge for contributor_id (admin auth)
 *   POST /v1/uploads/sign                        → issue signed PUT URL for an upload
 *   POST /v1/uploads/complete                    → record successful upload, update stats
 *   POST /v1/revoke/cascade                      → mark contributor for purge (30-day SLA)
 *   GET  /v1/contributors/:id/stats              → return contribution stats for `reflex contribute --status`
 *
 * Auth (Phase 1):
 *   - Admin endpoints: Authorization: Bearer <ADMIN_TOKEN>
 *   - Customer endpoints: contributor_id + opted_in_at as soft-proof.
 *     Phase 1.5 will add Ed25519 challenge-response signed by the
 *     consent receipt's user-side key.
 *
 * Storage:
 *   - D1 binding `DB`: contributors, uploads, daily_uploads, revoke_requests
 *   - R2 binding `CURATE_BUCKET`: object payloads under `free-contributors/<id>/`
 *     or `pro-contributors/<id>/` paths
 *
 * Rate limiting:
 *   - 10 GB/day per contributor (configurable via DAILY_BYTES_LIMIT env var)
 *   - 1000 uploads/day per contributor (configurable via DAILY_UPLOADS_LIMIT)
 *   - Cloudflare's built-in DDoS protection on the public endpoints
 */

const ADMIN_TOKEN_HEADER = "Authorization";

const DEFAULT_DAILY_BYTES_LIMIT = 10 * 1024 * 1024 * 1024; // 10 GB
const DEFAULT_DAILY_UPLOADS_LIMIT = 1000;
const SIGNED_URL_TTL_SECONDS = 15 * 60;                    // 15 minutes
const REVOKE_SLA_DAYS = 30;

// Cascade stage SLAs (per consent-revoke_research.md open question 1: tighter
// is better for trust signaling; spec's 24h is conservative).
const TOMBSTONE_DELAY_MS = 5 * 60 * 1000;          // 5 min — covers in-flight uploads
const R2_PURGE_DELAY_MS = 10 * 60 * 1000;          // 10 min total — purge after tombstone
const R2_LIST_PAGE_SIZE = 1000;                     // R2 list pagination size

// ---------- request router ----------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const method = request.method;
    const path = url.pathname;

    try {
      if (method === "GET" && path === "/healthz") return healthz();
      if (method === "POST" && path === "/admin/init-bucket")
        return await adminAuth(request, env, () => adminInitBucket(env));
      if (method === "GET" && path === "/admin/contributors")
        return await adminAuth(request, env, () => adminListContributors(env));
      if (method === "POST" && path === "/admin/manual-purge")
        return await adminAuth(request, env, () => adminManualPurge(request, env));
      if (method === "POST" && path === "/v1/uploads/sign")
        return await postUploadsSign(request, env);
      if (method === "PUT" && path.startsWith("/v1/uploads/put/")) {
        const uploadId = path.split("/").pop();
        return await putUploadBytes(uploadId, request, env);
      }
      if (method === "POST" && path === "/v1/uploads/complete")
        return await postUploadsComplete(request, env);
      if (method === "POST" && path === "/v1/revoke/cascade")
        return await postRevokeCascade(request, env);
      if (method === "GET" && path.startsWith("/v1/revoke/cascade-status/")) {
        const requestId = path.split("/").pop();
        return await getRevokeCascadeStatus(requestId, env);
      }
      if (method === "POST" && path.startsWith("/admin/cascade-execute/")) {
        const requestId = path.split("/").pop();
        return await adminAuth(request, env, () => adminExecuteCascade(requestId, env));
      }
      if (method === "GET" && path.startsWith("/v1/contributors/")) {
        const parts = path.split("/").filter(Boolean);
        // /v1/contributors/<id>/stats
        if (parts.length === 4 && parts[3] === "stats") {
          return await getContributorStats(parts[2], env);
        }
      }
      return jsonResponse(404, { error: "not_found", path });
    } catch (e) {
      console.error("Worker error:", e.message, e.stack);
      return jsonResponse(500, { error: "internal_error", message: e.message });
    }
  },
};

// ---------- middleware ----------

async function adminAuth(request, env, handler) {
  const auth = request.headers.get(ADMIN_TOKEN_HEADER) || "";
  const expected = `Bearer ${env.ADMIN_TOKEN}`;
  if (!env.ADMIN_TOKEN || auth !== expected) {
    return jsonResponse(401, { error: "unauthorized" });
  }
  return await handler();
}

// ---------- handlers: health + admin ----------

function healthz() {
  return jsonResponse(200, { status: "ok", service: "reflex-contribution-worker" });
}

async function adminInitBucket(env) {
  if (!env.CURATE_BUCKET) {
    return jsonResponse(500, {
      error: "bucket_not_bound",
      hint: "Run `wrangler r2 bucket create reflex-curate` then redeploy.",
    });
  }
  // Cheap probe: list with limit=1.
  const list = await env.CURATE_BUCKET.list({ limit: 1 });
  return jsonResponse(200, {
    status: "ok",
    bucket_name: "reflex-curate",
    objects_present: list.objects.length > 0,
  });
}

async function adminListContributors(env) {
  const rows = await env.DB.prepare(
    `SELECT contributor_id, tier, first_seen_at, last_active_at,
            total_episodes, total_bytes, total_uploads, revoked_at
       FROM contributors
       ORDER BY total_bytes DESC
       LIMIT 500`
  ).all();
  return jsonResponse(200, { contributors: rows.results || [] });
}

async function adminManualPurge(request, env) {
  const body = await request.json().catch(() => ({}));
  const contributorId = body.contributor_id;
  if (!contributorId) {
    return jsonResponse(400, { error: "missing_contributor_id" });
  }
  return await initiateRevoke(env, contributorId, "all", "admin");
}

// ---------- handlers: customer-facing ----------

/**
 * POST /v1/uploads/sign
 *
 * Body: {
 *   contributor_id: string,
 *   tier: "free" | "pro",
 *   opted_in_at: ISO8601,
 *   file_name: string,         // e.g. "2026-05-05-sess-abcdef.jsonl"
 *   byte_size: number,
 *   episode_count: number,
 *   privacy_mode: "hash_only" | "raw_opt_in",
 * }
 *
 * Returns: {
 *   upload_id: string,
 *   r2_key: string,
 *   put_url: string,           // signed; PUT raw bytes here
 *   expires_at: ISO8601,
 * }
 *
 * Auth (Phase 1): caller asserts contributor_id + opted_in_at; the worker
 * trusts the client. Phase 1.5 adds Ed25519 challenge-response.
 */
async function postUploadsSign(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });

  const contributorId = body.contributor_id;
  const tier = body.tier;
  const optedInAt = body.opted_in_at;
  const fileName = body.file_name;
  const byteSize = Number(body.byte_size);

  if (!contributorId || !tier || !optedInAt || !fileName || !Number.isFinite(byteSize)) {
    return jsonResponse(400, {
      error: "missing_fields",
      required: ["contributor_id", "tier", "opted_in_at", "file_name", "byte_size"],
    });
  }
  if (!["free", "pro", "enterprise"].includes(tier)) {
    return jsonResponse(400, { error: "invalid_tier", got: tier });
  }
  if (!isSafeFileName(fileName)) {
    return jsonResponse(400, { error: "invalid_file_name", message: "no slashes / nulls / leading dots" });
  }
  if (byteSize <= 0 || byteSize > 1 * 1024 * 1024 * 1024) {
    return jsonResponse(400, { error: "byte_size_out_of_range", limit: "0 < size <= 1 GB per upload" });
  }

  // Refuse if contributor is already revoked.
  const existing = await env.DB.prepare(
    `SELECT revoked_at FROM contributors WHERE contributor_id = ?`
  ).bind(contributorId).first();
  if (existing && existing.revoked_at) {
    return jsonResponse(403, {
      error: "contributor_revoked",
      revoked_at: existing.revoked_at,
      message: "Contributor previously revoked. New uploads will not be accepted.",
    });
  }

  // Rate limit: daily bytes + uploads.
  const dailyBytesLimit = Number(env.DAILY_BYTES_LIMIT) || DEFAULT_DAILY_BYTES_LIMIT;
  const dailyUploadsLimit = Number(env.DAILY_UPLOADS_LIMIT) || DEFAULT_DAILY_UPLOADS_LIMIT;
  const utcDate = new Date().toISOString().slice(0, 10);

  const daily = await env.DB.prepare(
    `SELECT bytes_uploaded, uploads_count FROM daily_uploads
       WHERE contributor_id = ? AND utc_date = ?`
  ).bind(contributorId, utcDate).first();
  const usedBytes = daily?.bytes_uploaded || 0;
  const usedUploads = daily?.uploads_count || 0;
  if (usedBytes + byteSize > dailyBytesLimit) {
    return jsonResponse(429, {
      error: "daily_byte_limit_exceeded",
      used_today: usedBytes,
      limit: dailyBytesLimit,
      retry_after_utc: `${utcDate}T23:59:59Z`,
    });
  }
  if (usedUploads >= dailyUploadsLimit) {
    return jsonResponse(429, {
      error: "daily_upload_count_exceeded",
      used_today: usedUploads,
      limit: dailyUploadsLimit,
    });
  }

  // Ensure contributor row exists.
  const nowIso = new Date().toISOString();
  await env.DB.prepare(
    `INSERT INTO contributors (contributor_id, tier, first_seen_at, last_active_at)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(contributor_id) DO UPDATE SET last_active_at = excluded.last_active_at`
  ).bind(contributorId, tier, nowIso, nowIso).run();

  // Reservation-based rate limiting (Phase 1.5; closes the bypass surfaced in
  // reflex_context/03_experiments/2026-05-06-contribution-worker-stress-test.md).
  // Reserve byte_size + 1 upload count NOW, before returning the signed URL.
  // /complete no longer touches daily_uploads — sign is the consumption event.
  // Effect: a client that signs without completing still consumes its quota,
  // closing the soft-DoS / quota-bypass vector. Daily counter resets at UTC
  // midnight on its own (utc_date PK); no GC required.
  await env.DB.prepare(
    `INSERT INTO daily_uploads (contributor_id, utc_date, bytes_uploaded, uploads_count)
       VALUES (?, ?, ?, 1)
       ON CONFLICT(contributor_id, utc_date) DO UPDATE SET
         bytes_uploaded = bytes_uploaded + excluded.bytes_uploaded,
         uploads_count = uploads_count + 1`
  ).bind(contributorId, utcDate, byteSize).run();

  // Build R2 key + upload_id.
  const subdir = tier === "free" ? "free-contributors" : `${tier}-contributors`;
  const r2Key = `${subdir}/${contributorId}/${utcDate}/${fileName}`;
  const uploadId = `upl_${randomHex(16)}`;
  const expiresAt = new Date(Date.now() + SIGNED_URL_TTL_SECONDS * 1000).toISOString();

  await env.DB.prepare(
    `INSERT INTO uploads (upload_id, contributor_id, r2_key, byte_size, status, signed_at, user_agent, source_ip)
       VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)`
  ).bind(
    uploadId, contributorId, r2Key, byteSize, nowIso,
    request.headers.get("User-Agent") || "",
    request.headers.get("CF-Connecting-IP") || "",
  ).run();

  // Phase 1: Cloudflare R2's S3-compat signed-URL story for Workers is to
  // either (a) use the worker as the upload proxy (receive bytes, write to
  // R2 binding directly), or (b) use AWS SigV4 signed PUTs. We do (a) here
  // — simpler, single auth path. Returns a put_url that points to THIS
  // worker's /v1/uploads/put/<upload_id> endpoint.
  const putUrl = `${new URL(request.url).origin}/v1/uploads/put/${uploadId}`;

  return jsonResponse(200, {
    upload_id: uploadId,
    r2_key: r2Key,
    put_url: putUrl,
    expires_at: expiresAt,
    note: "PUT raw bytes to put_url. Then POST /v1/uploads/complete.",
  });
}

/**
 * PUT /v1/uploads/put/:upload_id
 *
 * Receives the raw upload bytes and writes them to R2 at the r2_key reserved
 * by /v1/uploads/sign. The worker is acting as the upload proxy here (vs
 * issuing AWS SigV4 signed URLs that would let the client PUT to R2 directly).
 * This is the "Phase 1 simple path" called out in /sign's note.
 *
 * Refuses if:
 *   - upload_id not found
 *   - upload status != "pending"
 *   - signed_at older than SIGNED_URL_TTL_SECONDS
 *   - byte size doesn't match what was signed (with 1% tolerance)
 *   - contributor was revoked between sign + put
 */
async function putUploadBytes(uploadId, request, env) {
  const upload = await env.DB.prepare(
    `SELECT u.upload_id, u.contributor_id, u.r2_key, u.byte_size, u.status,
            u.signed_at, c.revoked_at
       FROM uploads u
       LEFT JOIN contributors c ON c.contributor_id = u.contributor_id
       WHERE u.upload_id = ?`
  ).bind(uploadId).first();
  if (!upload) return jsonResponse(404, { error: "upload_not_found" });
  if (upload.status !== "pending") {
    return jsonResponse(409, { error: "upload_not_pending", status: upload.status });
  }
  if (upload.revoked_at) {
    return jsonResponse(403, { error: "contributor_revoked_between_sign_and_put" });
  }
  const signedAtMs = Date.parse(upload.signed_at);
  if (Date.now() - signedAtMs > SIGNED_URL_TTL_SECONDS * 1000) {
    return jsonResponse(410, { error: "upload_url_expired", signed_at: upload.signed_at });
  }

  const body = await request.arrayBuffer();
  const actualSize = body.byteLength;
  // 1% tolerance for header / framing variance.
  const diff = Math.abs(actualSize - upload.byte_size);
  if (diff > Math.max(1024, upload.byte_size * 0.01)) {
    return jsonResponse(400, {
      error: "byte_size_mismatch",
      signed_byte_size: upload.byte_size,
      actual_byte_size: actualSize,
    });
  }

  await env.CURATE_BUCKET.put(upload.r2_key, body, {
    httpMetadata: { contentType: "application/x-jsonlines" },
    customMetadata: {
      contributor_id: upload.contributor_id,
      upload_id: uploadId,
    },
  });

  return jsonResponse(200, {
    upload_id: uploadId,
    r2_key: upload.r2_key,
    bytes_received: actualSize,
    note: "POST /v1/uploads/complete to record the success.",
  });
}


/**
 * POST /v1/uploads/complete
 *
 * Body: { upload_id: string, episode_count: number }
 *
 * Records the upload as completed, increments stats + daily counters.
 * Idempotent — calling twice on the same upload_id is a no-op.
 *
 * Verifies the bytes actually landed in R2 (HEAD on the r2_key) — refuses
 * to mark "completed" if the object doesn't exist.
 */
async function postUploadsComplete(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });

  const uploadId = body.upload_id;
  const episodeCount = Number(body.episode_count) || 0;
  if (!uploadId) return jsonResponse(400, { error: "missing_upload_id" });

  const upload = await env.DB.prepare(
    `SELECT upload_id, contributor_id, r2_key, byte_size, status, signed_at FROM uploads WHERE upload_id = ?`
  ).bind(uploadId).first();
  if (!upload) return jsonResponse(404, { error: "upload_not_found" });
  if (upload.status !== "pending") {
    return jsonResponse(200, { status: upload.status, idempotent: true });
  }

  // Verify the bytes actually landed in R2. Refuse to mark "completed" if
  // the client called /complete without first PUTing the bytes.
  const r2Object = await env.CURATE_BUCKET.head(upload.r2_key);
  if (!r2Object) {
    return jsonResponse(412, {
      error: "r2_object_not_found",
      r2_key: upload.r2_key,
      hint: "PUT bytes to /v1/uploads/put/<upload_id> first",
    });
  }

  const nowIso = new Date().toISOString();
  const utcDate = nowIso.slice(0, 10);

  // Mark the upload completed + update aggregate counters atomically.
  // daily_uploads is NO LONGER updated here — /v1/uploads/sign reserves
  // the quota up front (Phase 1.5 fix; see reservation-based rate
  // limiting comment in postUploadsSign).
  await env.DB.batch([
    env.DB.prepare(
      `UPDATE uploads SET status = 'completed', completed_at = ? WHERE upload_id = ?`
    ).bind(nowIso, uploadId),
    env.DB.prepare(
      `UPDATE contributors
         SET total_episodes = total_episodes + ?,
             total_bytes = total_bytes + ?,
             total_uploads = total_uploads + 1,
             last_active_at = ?
         WHERE contributor_id = ?`
    ).bind(episodeCount, upload.byte_size, nowIso, upload.contributor_id),
  ]);

  return jsonResponse(200, { status: "completed", upload_id: uploadId });
}

/**
 * POST /v1/revoke/cascade
 *
 * Body: { contributor_id: string, scope: "all" | "future_only" }
 *
 * Marks the contributor for revoke. The actual purge cascade (delete R2
 * objects + rebuild derived datasets + email buyers) runs as a separate
 * background job; this endpoint only enqueues the request and updates
 * the contributor's revoked_at marker so future signs are refused.
 */
async function postRevokeCascade(request, env) {
  const body = await request.json().catch(() => null);
  if (!body) return jsonResponse(400, { error: "invalid_json" });
  const contributorId = body.contributor_id;
  const scope = body.scope || "all";
  if (!contributorId) return jsonResponse(400, { error: "missing_contributor_id" });
  if (!["all", "future_only"].includes(scope)) {
    return jsonResponse(400, { error: "invalid_scope", got: scope });
  }
  return await initiateRevoke(env, contributorId, scope, "customer");
}

/**
 * GET /v1/contributors/:id/stats
 *
 * Returns the contributor's running totals. Used by `reflex contribute --status`.
 */
async function getContributorStats(contributorId, env) {
  if (!contributorId) return jsonResponse(400, { error: "missing_contributor_id" });
  const row = await env.DB.prepare(
    `SELECT contributor_id, tier, first_seen_at, last_active_at,
            total_episodes, total_bytes, total_uploads, revoked_at
       FROM contributors WHERE contributor_id = ?`
  ).bind(contributorId).first();
  if (!row) return jsonResponse(404, { error: "not_found" });
  return jsonResponse(200, row);
}

// ---------- helpers ----------

async function initiateRevoke(env, contributorId, scope, source) {
  const requestId = `rev_${randomHex(16)}`;
  const nowIso = new Date().toISOString();

  // Mark contributor revoked. Idempotent; first call wins.
  await env.DB.prepare(
    `INSERT INTO contributors (contributor_id, tier, first_seen_at, last_active_at, revoked_at)
       VALUES (?, 'unknown', ?, ?, ?)
       ON CONFLICT(contributor_id) DO UPDATE SET
         revoked_at = COALESCE(revoked_at, excluded.revoked_at),
         last_active_at = excluded.last_active_at`
  ).bind(contributorId, nowIso, nowIso, nowIso).run();

  // Phase 1 simplification (per consent-revoke_research.md): no derived
  // datasets / buyers exist yet, so Stages 4 + 5 auto-complete at init.
  await env.DB.prepare(
    `INSERT INTO revoke_requests
       (request_id, contributor_id, requested_at, scope, status,
        derived_rebuild_completed_at, buyer_notification_completed_at, notes)
       VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)`
  ).bind(
    requestId, contributorId, nowIso, scope,
    nowIso, nowIso,  // derived_rebuild + buyer_notification auto-complete
    `source=${source}; phase1_no_derived_datasets_no_buyers`,
  ).run();

  // Optional: alert via Slack so an operator can monitor.
  if (env.SLACK_WEBHOOK_URL) {
    await postSlack(env.SLACK_WEBHOOK_URL, {
      text: `Curate revoke requested — contributor_id=${contributorId} scope=${scope} source=${source}. Cascade SLA: ${REVOKE_SLA_DAYS} days.`,
    }).catch((e) => console.error("Slack post failed:", e.message));
  }

  return jsonResponse(200, {
    request_id: requestId,
    contributor_id: contributorId,
    sla_days: REVOKE_SLA_DAYS,
    note: "Cascade auto-progresses via /v1/revoke/cascade-status/<request_id>.",
  });
}


/**
 * GET /v1/revoke/cascade-status/<request_id>
 *
 * Returns the current cascade state for a request. Lazily progresses the
 * cascade through any stages whose SLA has elapsed (no separate cron needed).
 */
async function getRevokeCascadeStatus(requestId, env) {
  if (!requestId) return jsonResponse(400, { error: "missing_request_id" });
  const fresh = await loadAndProgressCascade(requestId, env, { force: false });
  if (!fresh) return jsonResponse(404, { error: "request_not_found" });
  return jsonResponse(200, formatCascadeStatus(fresh));
}


async function adminExecuteCascade(requestId, env) {
  if (!requestId) return jsonResponse(400, { error: "missing_request_id" });
  const fresh = await loadAndProgressCascade(requestId, env, { force: true });
  if (!fresh) return jsonResponse(404, { error: "request_not_found" });
  return jsonResponse(200, formatCascadeStatus(fresh));
}


function formatCascadeStatus(req) {
  const stages = [
    { name: "revoke", at: req.requested_at, status: "completed" },
    { name: "tombstone", at: req.tombstone_at, status: req.tombstone_at ? "completed" : "pending" },
    { name: "r2_purge",
      at: req.r2_purge_completed_at,
      status: req.r2_purge_completed_at ? "completed" :
              req.r2_purge_started_at ? "in_progress" : "pending",
      objects_purged: req.r2_objects_purged || 0 },
    { name: "derived_rebuild",
      at: req.derived_rebuild_completed_at,
      status: req.derived_rebuild_completed_at ? "completed" : "pending",
      datasets_rebuilt: req.derived_datasets_rebuilt || 0 },
    { name: "buyer_notification",
      at: req.buyer_notification_completed_at,
      status: req.buyer_notification_completed_at ? "completed" : "pending",
      notifications_sent: req.buyer_notifications_sent || 0 },
  ];
  const allDone = stages.every((s) => s.status === "completed");
  return {
    request_id: req.request_id,
    contributor_id: req.contributor_id,
    requested_at: req.requested_at,
    overall_status: allDone ? "completed" : "in_progress",
    stages,
    sla_days: REVOKE_SLA_DAYS,
    completed_at: req.completed_at,
  };
}


/**
 * Load the request, progress any stages whose SLA has elapsed, return fresh row.
 * Idempotent — each stage checks its completion timestamp before running.
 *
 * Args:
 *   force: bypass SLA waits (admin path)
 */
async function loadAndProgressCascade(requestId, env, { force = false } = {}) {
  let req = await env.DB.prepare(
    `SELECT * FROM revoke_requests WHERE request_id = ?`
  ).bind(requestId).first();
  if (!req) return null;
  if (req.status === "completed") return req;

  const now = Date.now();
  const requestedAtMs = Date.parse(req.requested_at);
  const nowIso = new Date(now).toISOString();

  // Stage 2 — tombstone (5 min after revoke; immediate on force).
  if (!req.tombstone_at && (force || now - requestedAtMs >= TOMBSTONE_DELAY_MS)) {
    await env.DB.prepare(
      `UPDATE revoke_requests SET tombstone_at = ? WHERE request_id = ?`
    ).bind(nowIso, requestId).run();
    req.tombstone_at = nowIso;
  }

  // Stage 3 — R2 purge (10 min after revoke; immediate on force).
  if (!req.r2_purge_completed_at && (force || now - requestedAtMs >= R2_PURGE_DELAY_MS)) {
    await executeR2Purge(env, req);
    req = await env.DB.prepare(
      `SELECT * FROM revoke_requests WHERE request_id = ?`
    ).bind(requestId).first();
  }

  // Stage 4 + 5 already auto-completed at init for Phase 1 (no derived
  // datasets / buyers exist).

  // Top-level completion check.
  if (
    req.tombstone_at &&
    req.r2_purge_completed_at &&
    req.derived_rebuild_completed_at &&
    req.buyer_notification_completed_at &&
    req.status !== "completed"
  ) {
    await env.DB.prepare(
      `UPDATE revoke_requests SET status = 'completed', completed_at = ? WHERE request_id = ?`
    ).bind(nowIso, requestId).run();
    req.status = "completed";
    req.completed_at = nowIso;
  }

  return req;
}


async function executeR2Purge(env, req) {
  const startedAtIso = new Date().toISOString();
  await env.DB.prepare(
    `UPDATE revoke_requests SET r2_purge_started_at = COALESCE(r2_purge_started_at, ?)
       WHERE request_id = ?`
  ).bind(startedAtIso, req.request_id).run();

  // Determine tier prefix. We don't know tier from the request row alone,
  // so try all 3 tiers ("free-contributors", "pro-contributors",
  // "enterprise-contributors"). At Phase 1 only one will have data.
  const prefixes = [
    `free-contributors/${req.contributor_id}/`,
    `pro-contributors/${req.contributor_id}/`,
    `enterprise-contributors/${req.contributor_id}/`,
  ];

  let totalPurged = 0;
  for (const prefix of prefixes) {
    let cursor = undefined;
    while (true) {
      const list = await env.CURATE_BUCKET.list({
        prefix,
        limit: R2_LIST_PAGE_SIZE,
        cursor,
      });
      if (!list.objects || list.objects.length === 0) break;
      // Delete each object. R2's delete() takes single key per call;
      // delete-many (batch) isn't supported in the worker SDK.
      for (const obj of list.objects) {
        await env.CURATE_BUCKET.delete(obj.key);
        totalPurged += 1;
      }
      if (!list.truncated) break;
      cursor = list.cursor;
    }
  }

  // Also mark all uploads for this contributor as purged in D1.
  const completedAtIso = new Date().toISOString();
  await env.DB.prepare(
    `UPDATE uploads SET status = 'purged' WHERE contributor_id = ?`
  ).bind(req.contributor_id).run();
  await env.DB.prepare(
    `UPDATE revoke_requests
       SET r2_purge_completed_at = ?, r2_objects_purged = ?
       WHERE request_id = ?`
  ).bind(completedAtIso, totalPurged, req.request_id).run();
}

function isSafeFileName(name) {
  if (typeof name !== "string") return false;
  if (name.length === 0 || name.length > 255) return false;
  if (name.includes("/") || name.includes("\\") || name.includes("\x00")) return false;
  if (name.startsWith(".")) return false;
  return /^[A-Za-z0-9._-]+$/.test(name);
}

function randomHex(byteLen) {
  const bytes = new Uint8Array(byteLen);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function jsonResponse(status, obj) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function postSlack(url, payload) {
  await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}
