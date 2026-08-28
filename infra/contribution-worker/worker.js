import {
  authenticateContributorRequest,
  bytesToBase64url,
  canonicalizeJson,
  parseCanonicalJsonBody,
  registerContributor,
  rotateContributorKey,
} from "./contributor-auth.js";
export { ContributorStorageFence } from "./contributor-storage-fence.js";

/**
 * Reflex contribution-worker — Cloudflare Worker for the Curate wedge.
 *
 * Endpoints:
 *   GET  /healthz                                → health probe
 *   POST /admin/init-bucket                      → confirm R2 bucket exists (admin auth)
 *   GET  /admin/contributors                     → list contributors + stats (admin auth)
 *   POST /admin/manual-purge                     → trigger cascade purge for contributor_id (admin auth)
 *   POST /admin/cascade-execute/:request_id       → force cascade progression (admin auth)
 *   POST /v1/contributors/register               → enroll an Ed25519 contributor key
 *   POST /v1/contributors/rotate                 → rotate the active contributor key
 *   POST /v1/uploads/sign                        → issue upload reservation (contributor auth)
 *   POST /v1/uploads/complete                    → record successful upload, update stats
 *   POST /v1/revoke/cascade                      → mark contributor for purge (admin auth)
 *   GET  /v1/revoke/cascade-status/:request_id   → read cascade status (never mutates)
 *   GET  /v1/contributors/:id/stats              → return contribution stats for `reflex contribute --status`
 *
 * Auth:
 *   - Admin and revoke-mutation endpoints: Authorization: Bearer <ADMIN_TOKEN>
 *   - Customer endpoints: Contributor Authentication v1 Ed25519 signatures.
 *
 * Storage:
 *   - D1 binding `DB`: contributors, contributor_keys, contributor_nonces,
 *     uploads, daily_uploads, revoke_requests
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
const SIGNED_URL_TTL_SECONDS = 10 * 60;
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const MAX_SIGN_REQUESTS_PER_HOUR = 60;
const UPLOAD_ATTEMPT_STALE_SECONDS = 5 * 60;
const REVOKE_SLA_DAYS = 30;

// Cascade stage SLAs (per consent-revoke_research.md open question 1: tighter
// is better for trust signaling; spec's 24h is conservative).
const TOMBSTONE_DELAY_MS = 5 * 60 * 1000;          // 5 min — covers in-flight uploads
const R2_PURGE_DELAY_MS = 10 * 60 * 1000;          // 10 min total — purge after tombstone

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
      if (method === "POST" && path === "/v1/contributors/register") {
        const result = await registerContributor(request, env);
        if (!result.ok) return result.response;
        return jsonResponse(result.status, {
          contributor_id: result.contributorId,
          key_id: result.keyId,
          tier: result.tier,
        });
      }
      if (method === "POST" && path === "/v1/contributors/rotate") {
        const result = await rotateContributorKey(request, env);
        if (!result.ok) return result.response;
        return jsonResponse(200, {
          contributor_id: result.contributorId,
          old_key_id: result.oldKeyId,
          key_id: result.keyId,
        });
      }
      if (method === "POST" && path === "/v1/uploads/sign") {
        const auth = await authenticateContributorRequest(request, env);
        if (!auth.ok) return auth.response;
        return await postUploadsSign(request, env, auth);
      }
      if (method === "PUT" && path.startsWith("/v1/uploads/put/")) {
        const uploadId = path.split("/").pop();
        return await putUploadBytes(uploadId, request, env);
      }
      if (method === "POST" && path === "/v1/uploads/complete") {
        const auth = await authenticateContributorRequest(request, env);
        if (!auth.ok) return auth.response;
        return await postUploadsComplete(env, auth);
      }
      if (method === "POST" && path === "/v1/revoke/cascade")
        return await adminAuth(request, env, () => postRevokeCascade(request, env));
      if (method === "GET" && path.startsWith("/v1/revoke/cascade-status/")) {
        const requestId = path.split("/").pop();
        if (await hasValidAdminBearer(request, env)) {
          return await getRevokeCascadeStatus(requestId, env, null);
        }
        const auth = await authenticateContributorRequest(request, env);
        if (!auth.ok) return auth.response;
        return await getRevokeCascadeStatus(requestId, env, auth.principal);
      }
      if (method === "POST" && path.startsWith("/admin/cascade-execute/")) {
        const requestId = path.split("/").pop();
        return await adminAuth(request, env, () => adminExecuteCascade(requestId, env));
      }
      if (method === "GET" && path.startsWith("/v1/contributors/")) {
        const parts = path.split("/").filter(Boolean);
        // /v1/contributors/<id>/stats
        if (parts.length === 4 && parts[3] === "stats") {
          if (await hasValidAdminBearer(request, env)) {
            return await getContributorStats(parts[2], env, null);
          }
          const auth = await authenticateContributorRequest(request, env);
          if (!auth.ok) return auth.response;
          return await getContributorStats(parts[2], env, auth.principal);
        }
      }
      return jsonResponse(404, { error: "not_found", path });
    } catch (e) {
      console.error(JSON.stringify({
        message: "contribution_worker_request_failed",
        method,
        path,
        error: e instanceof Error ? e.message : String(e),
      }));
      return jsonResponse(500, { error: "internal_error" });
    }
  },

  scheduled(_controller, env, ctx) {
    // Timed cascade progression is deliberately unavailable over public HTTP.
    // Cloudflare invokes this hook from the configured cron trigger.
    ctx.waitUntil(Promise.all([
      progressPendingCascades(env),
      recoverStaleUploadAttempts(env),
      pruneExpiredContributorNonces(env),
    ]));
  },
};

// ---------- middleware ----------

async function adminAuth(request, env, handler) {
  if (!(await hasValidAdminBearer(request, env))) {
    return jsonResponse(401, { error: "unauthorized" });
  }
  return await handler();
}

/**
 * Verify the admin bearer without an early-exit string comparison.
 *
 * Both values are reduced to fixed-length SHA-256 digests before the XOR
 * comparison, so differing token lengths and the first mismatching byte do
 * not change the comparison loop. An absent server-side secret always fails.
 */
async function hasValidAdminBearer(request, env) {
  const authorization = request.headers.get(ADMIN_TOKEN_HEADER) || "";
  const supplied = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length)
    : "";
  const expected = typeof env.ADMIN_TOKEN === "string" ? env.ADMIN_TOKEN : "";
  const encoder = new TextEncoder();
  const [suppliedDigest, expectedDigest] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(supplied)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);

  const suppliedBytes = new Uint8Array(suppliedDigest);
  const expectedBytes = new Uint8Array(expectedDigest);
  let mismatch = expected.length === 0 ? 1 : 0;
  for (let i = 0; i < expectedBytes.length; i += 1) {
    mismatch |= suppliedBytes[i] ^ expectedBytes[i];
  }
  return mismatch === 0;
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
 *   file_name: string,         // e.g. "2026-05-05-sess-abcdef.jsonl"
 *   byte_size: number,
 *   manifest: { ...tether.anonymization.manifest... },
 * }
 *
 * Returns: {
 *   upload_id: string,
 *   r2_key: string,
 *   put_url: string,           // signed; PUT raw bytes here
 *   expires_at: ISO8601,
 * }
 *
 * Contributor identity and tier come only from the verified principal.
 */
async function postUploadsSign(request, env, auth) {
  const parsedBody = parseCanonicalJsonBody(auth.bodyBytes);
  if (parsedBody.response) return parsedBody.response;
  const body = parsedBody.value;

  const contributorId = auth.principal.contributorId;
  const tier = auth.principal.tier;
  const fileName = body.file_name;
  const byteSize = Number(body.byte_size);
  const reservationKey = body.reservation_key;
  if (/^[A-Za-z0-9_-]{43}$/.test(reservationKey || "")) {
    const prior = await replayReservationResponse(
      request, env, contributorId, reservationKey, auth.headers.bodySha256,
    );
    if (prior) return prior;
  }
  const manifest = validateAnonymizationManifest(body.manifest, auth.nowSeconds);

  if (!fileName || !Number.isInteger(byteSize) || manifest.response
      || !/^[A-Za-z0-9_-]{43}$/.test(reservationKey || "")) {
    if (manifest.response) return manifest.response;
    return jsonResponse(400, {
      error: "missing_fields",
      required: ["file_name", "byte_size", "manifest", "reservation_key"],
    });
  }
  if (!["free", "pro", "enterprise"].includes(tier)) {
    return jsonResponse(500, { error: "invalid_server_tier" });
  }
  if (!isSafeFileName(fileName)) {
    return jsonResponse(400, { error: "invalid_file_name", message: "no slashes / nulls / leading dots" });
  }
  if (byteSize <= 0 || byteSize > MAX_UPLOAD_BYTES) {
    return jsonResponse(byteSize > MAX_UPLOAD_BYTES ? 413 : 400, {
      error: "byte_size_out_of_range",
      limit: MAX_UPLOAD_BYTES,
    });
  }

  // Rate limit: daily bytes + uploads.
  const dailyBytesLimit = Number(env.DAILY_BYTES_LIMIT) || DEFAULT_DAILY_BYTES_LIMIT;
  const dailyUploadsLimit = Number(env.DAILY_UPLOADS_LIMIT) || DEFAULT_DAILY_UPLOADS_LIMIT;
  const utcDate = new Date(auth.nowSeconds * 1000).toISOString().slice(0, 10);

  const dayStartSeconds = Math.floor(
    Date.parse(`${utcDate}T00:00:00.000Z`) / 1000,
  );
  // Admission is a single conditional INSERT below, followed by an
  // idempotency reread before any quota response.

  const nowSeconds = auth.nowSeconds;
  const manifestSha256 = await sha256Hex(
    new TextEncoder().encode(canonicalizeJson(manifest.value)),
  );
  const capabilityBytes = await reservationCapability(
    env, contributorId, reservationKey,
  );
  if (!capabilityBytes) {
    return jsonResponse(500, { error: "upload_capability_secret_missing" });
  }
  const uploadCapability = bytesToBase64url(capabilityBytes);
  const capabilitySha256 = await sha256Hex(capabilityBytes);

  const replay = await env.DB.prepare(
    `SELECT * FROM uploads WHERE contributor_id = ? AND reservation_key = ?`
  ).bind(contributorId, reservationKey).first();
  if (replay) {
    if (!constantTimeHexEqual(replay.request_sha256, auth.headers.bodySha256)) {
      return jsonResponse(409, { error: "reservation_key_payload_mismatch" });
    }
    return reservationResponse(request, replay, uploadCapability, true);
  }

  const nowIso = new Date(auth.nowSeconds * 1000).toISOString();
  await env.DB.prepare(
    `UPDATE contributors SET last_active_at = ? WHERE contributor_id = ?`
  ).bind(nowIso, contributorId).run();

  // Build R2 key + upload_id.
  const subdir = tier === "free" ? "free-contributors" : `${tier}-contributors`;
  const r2Key = `${subdir}/${contributorId}/${utcDate}/${fileName}`;
  const uploadId = `upl_${randomHex(16)}`;
  const expiresAtSeconds = nowSeconds + SIGNED_URL_TTL_SECONDS;
  const expiresAt = new Date(expiresAtSeconds * 1000).toISOString();
  const inserted = await env.DB.prepare(
    `INSERT OR IGNORE INTO uploads
       (upload_id, contributor_id, r2_key, byte_size, media_type, content_sha256,
        manifest_sha256, capability_sha256, expires_at, status, signed_at,
        signed_at_epoch, user_agent, source_ip, reservation_key, request_sha256)
     SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?
      WHERE (SELECT COUNT(*) FROM uploads
              WHERE contributor_id = ? AND signed_at_epoch >= ?) < ?
        AND COALESCE((SELECT SUM(byte_size) FROM uploads
                       WHERE contributor_id = ? AND signed_at_epoch >= ?), 0) + ? <= ?
        AND (SELECT COUNT(*) FROM uploads
              WHERE contributor_id = ? AND signed_at_epoch >= ?) < ?`
  ).bind(
    uploadId, contributorId, r2Key, byteSize, manifest.value.media_type,
    manifest.value.content_sha256,
    manifestSha256, capabilitySha256, expiresAtSeconds, nowIso, nowSeconds,
    request.headers.get("User-Agent") || "",
    request.headers.get("CF-Connecting-IP") || "", reservationKey, auth.headers.bodySha256,
    contributorId, nowSeconds - 60 * 60, MAX_SIGN_REQUESTS_PER_HOUR,
    contributorId, dayStartSeconds, byteSize, dailyBytesLimit,
    contributorId, dayStartSeconds, dailyUploadsLimit,
  ).run();
  if (Number(inserted?.meta?.changes ?? inserted?.changes ?? 0) !== 1) {
    const won = await env.DB.prepare(
      `SELECT * FROM uploads WHERE contributor_id = ? AND reservation_key = ?`
    ).bind(contributorId, reservationKey).first();
    if (won) {
      if (!constantTimeHexEqual(won.request_sha256, auth.headers.bodySha256)) {
        return jsonResponse(409, { error: "reservation_key_payload_mismatch" });
      }
      return reservationResponse(request, won, uploadCapability, true);
    }
    const finalDaily = await env.DB.prepare(
      `SELECT COALESCE(SUM(byte_size), 0) AS bytes_uploaded,
              COUNT(*) AS uploads_count
         FROM uploads WHERE contributor_id = ? AND signed_at_epoch >= ?`
    ).bind(contributorId, dayStartSeconds).first();
    if (Number(finalDaily?.bytes_uploaded || 0) + byteSize > dailyBytesLimit) {
      return jsonResponse(429, {
        error: "daily_byte_limit_exceeded",
        used_today: Number(finalDaily?.bytes_uploaded || 0),
        limit: dailyBytesLimit,
        retry_after_utc: `${utcDate}T23:59:59Z`,
      });
    }
    if (Number(finalDaily?.uploads_count || 0) >= dailyUploadsLimit) {
      return jsonResponse(429, {
        error: "daily_upload_count_exceeded",
        used_today: Number(finalDaily?.uploads_count || 0),
        limit: dailyUploadsLimit,
      });
    }
    const finalRecent = await env.DB.prepare(
      `SELECT COUNT(*) AS count FROM uploads
        WHERE contributor_id = ? AND signed_at_epoch >= ?`
    ).bind(contributorId, nowSeconds - 60 * 60).first();
    if (Number(finalRecent?.count || 0) >= MAX_SIGN_REQUESTS_PER_HOUR) {
      return jsonResponse(429, {
        error: "sign_rate_limit_exceeded",
        limit: MAX_SIGN_REQUESTS_PER_HOUR,
        window_seconds: 3600,
      });
    }
    return jsonResponse(429, { error: "reservation_quota_exceeded" });
  }

  // Phase 1: Cloudflare R2's S3-compat signed-URL story for Workers is to
  // either (a) use the worker as the upload proxy (receive bytes, write to
  // R2 binding directly), or (b) use AWS SigV4 signed PUTs. We do (a) here
  // — simpler, single auth path. Returns a put_url that points to THIS
  // worker's /v1/uploads/put/<upload_id> endpoint.
  return reservationResponse(request, {
    upload_id: uploadId, r2_key: r2Key, expires_at: expiresAtSeconds,
  }, uploadCapability, false);
}

async function replayReservationResponse(request, env, contributorId, reservationKey, bodySha256) {
  const row = await env.DB.prepare(
    `SELECT * FROM uploads WHERE contributor_id = ? AND reservation_key = ?`
  ).bind(contributorId, reservationKey).first();
  if (!row) return null;
  if (!constantTimeHexEqual(row.request_sha256, bodySha256)) {
    return jsonResponse(409, { error: "reservation_key_payload_mismatch" });
  }
  const capability = await reservationCapability(env, contributorId, reservationKey);
  if (!capability) return jsonResponse(500, { error: "upload_capability_secret_missing" });
  return reservationResponse(request, row, bytesToBase64url(capability), true);
}

function constantTimeHexEqual(left, right) {
  const a = typeof left === "string" ? left : "";
  const b = typeof right === "string" ? right : "";
  let mismatch = a.length ^ b.length;
  for (let index = 0; index < 64; index += 1) {
    mismatch |= (a.charCodeAt(index) || 0) ^ (b.charCodeAt(index) || 0);
  }
  return mismatch === 0;
}

function reservationResponse(request, row, uploadCapability, idempotent) {
  return jsonResponse(200, {
    upload_id: row.upload_id,
    r2_key: row.r2_key,
    put_url: `${new URL(request.url).origin}/v1/uploads/put/${row.upload_id}`,
    upload_capability: uploadCapability,
    expires_at: new Date(Number(row.expires_at) * 1000).toISOString(),
    idempotent,
    note: "PUT raw bytes with Authorization: Upload <upload_capability>. Then sign POST /v1/uploads/complete.",
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
 * The one-time upload capability claims one attempt before request bytes are
 * read. Every attempt writes a unique R2 key so a stale/losing request cannot
 * delete an object selected by another attempt.
 */
async function putUploadBytes(uploadId, request, env) {
  const upload = await env.DB.prepare(
    `SELECT u.upload_id, u.contributor_id, u.r2_key, u.byte_size, u.media_type, u.status,
            u.content_sha256, u.manifest_sha256, u.capability_sha256,
            u.expires_at, c.revoked_at
       FROM uploads u
       LEFT JOIN contributors c ON c.contributor_id = u.contributor_id
       WHERE u.upload_id = ?`
  ).bind(uploadId).first();
  if (!upload) return jsonResponse(404, { error: "upload_not_found" });
  const authorization = request.headers.get("Authorization") || "";
  const capability = authorization.startsWith("Upload ")
    ? authorization.slice("Upload ".length)
    : "";
  const capabilityBytes = base64urlToBytesExact(capability, 32);
  const suppliedHash = capabilityBytes ? await sha256Hex(capabilityBytes) : "0".repeat(64);
  if (!constantTimeTextEqual(suppliedHash, upload.capability_sha256 || "")) {
    return jsonResponse(401, { error: "invalid_upload_capability" });
  }
  const nowSeconds = Math.floor(Date.now() / 1000);
  if (upload.status === "expired" || (
    upload.status === "pending" && nowSeconds > Number(upload.expires_at)
  )) {
    await env.DB.prepare(
      `UPDATE uploads SET status = 'expired'
        WHERE upload_id = ? AND status = 'pending'`
    ).bind(uploadId).run();
    return jsonResponse(410, { error: "upload_reservation_expired" });
  }
  if (upload.status !== "pending") {
    return jsonResponse(409, { error: "upload_not_pending", status: upload.status });
  }
  if (upload.revoked_at) {
    return jsonResponse(403, { error: "contributor_revoked_between_sign_and_put" });
  }
  const contentLength = request.headers.get("Content-Length");
  if (contentLength === null) {
    return jsonResponse(411, { error: "content_length_required" });
  }
  if (!/^(0|[1-9][0-9]*)$/.test(contentLength)) {
    return jsonResponse(400, { error: "invalid_content_length" });
  }
  const declaredSize = Number(contentLength);
  if (declaredSize > MAX_UPLOAD_BYTES) {
    return jsonResponse(413, { error: "upload_too_large", limit: MAX_UPLOAD_BYTES });
  }
  if (declaredSize !== Number(upload.byte_size)) {
    return jsonResponse(400, {
      error: "content_length_mismatch",
      reserved_byte_size: upload.byte_size,
      content_length: declaredSize,
    });
  }

  const attemptId = `att_${randomHex(16)}`;
  const claimed = await env.DB.prepare(
    `UPDATE uploads
        SET status = 'uploading', attempt_id = ?, attempt_started_at = ?
      WHERE upload_id = ? AND status = 'pending' AND expires_at >= ?
        AND NOT EXISTS (
          SELECT 1 FROM contributors
           WHERE contributor_id = uploads.contributor_id
             AND revoked_at IS NOT NULL
        )`
  ).bind(attemptId, nowSeconds, uploadId, nowSeconds).run();
  if (Number(claimed?.meta?.changes ?? claimed?.changes ?? 0) !== 1) {
    return jsonResponse(409, { error: "upload_claim_conflict" });
  }

  const attemptKey = `${upload.r2_key}/attempts/${attemptId}`;
  try {
    const bodyResult = await readExactUploadBody(request, declaredSize);
    if (!bodyResult.ok) {
      await rejectUploadAttempt(env, uploadId, attemptId, attemptKey);
      return jsonResponse(400, {
        error: "body_size_mismatch",
        reserved_byte_size: upload.byte_size,
        actual_byte_size: bodyResult.actualSize,
      });
    }
    const body = bodyResult.bytes;
    const actualDigest = await sha256Hex(body);
    if (!constantTimeTextEqual(actualDigest, upload.content_sha256)) {
      await rejectUploadAttempt(env, uploadId, attemptId, attemptKey);
      return jsonResponse(400, { error: "content_digest_mismatch" });
    }

    const storage = env.CONTRIBUTOR_STORAGE.getByName(upload.contributor_id);
    const stored = await storage.fetch(new Request("https://storage.internal/payload", {
      method: "PUT",
      headers: {
        "Content-Type": upload.media_type,
        "X-Tether-Contributor-Id": upload.contributor_id,
        "X-Tether-R2-Key": attemptKey,
        "X-Tether-R2-Metadata": JSON.stringify({
          attempt_id: attemptId,
          byte_size: String(declaredSize),
          content_sha256: upload.content_sha256,
          contributor_id: upload.contributor_id,
          manifest_sha256: upload.manifest_sha256,
          upload_id: uploadId,
        }),
      },
      body,
    }));
    if (stored.status === 409) {
      await rejectUploadAttempt(env, uploadId, attemptId, attemptKey);
      return jsonResponse(409, { error: "contributor_purged_during_upload" });
    }
    if (!stored.ok) {
      throw new Error(`storage_fence_write_failed status=${stored.status}`);
    }

    const finalized = await env.DB.prepare(
      `UPDATE uploads
          SET status = 'uploaded', r2_key = ?, uploaded_at = ?
        WHERE upload_id = ? AND status = 'uploading' AND attempt_id = ?`
    ).bind(attemptKey, nowSeconds, uploadId, attemptId).run();
    if (Number(finalized?.meta?.changes ?? finalized?.changes ?? 0) !== 1) {
      await rejectUploadAttempt(env, uploadId, attemptId, attemptKey);
      return jsonResponse(409, { error: "upload_finalize_conflict" });
    }

    return jsonResponse(200, {
      upload_id: uploadId,
      r2_key: attemptKey,
      bytes_received: declaredSize,
      status: "uploaded",
    });
  } catch (error) {
    await rejectUploadAttempt(env, uploadId, attemptId, attemptKey);
    throw error;
  }
}

async function rejectUploadAttempt(env, uploadId, attemptId, attemptKey) {
  const rejected = await env.DB.prepare(
    `UPDATE uploads SET status = 'rejected'
      WHERE upload_id = ? AND status = 'uploading' AND attempt_id = ?`
  ).bind(uploadId, attemptId).run();
  if (Number(rejected?.meta?.changes ?? rejected?.changes ?? 0) === 1) {
    await env.CURATE_BUCKET.delete(attemptKey);
    return;
  }

  // Recovery may have finalized this exact attempt between the R2 write and
  // the original request's final CAS, and completion may already have advanced
  // it again. Never delete any attempt key D1 selected as authoritative,
  // regardless of its later lifecycle state.
  const current = await env.DB.prepare(
    `SELECT status, r2_key, attempt_id FROM uploads WHERE upload_id = ?`
  ).bind(uploadId).first();
  if (current?.status !== "purged" && current?.r2_key === attemptKey) return;
  await env.CURATE_BUCKET.delete(attemptKey);
  // A purge fence retains the attempt marker until its claimed writer has
  // acknowledged cleanup. Clearing it after deletion lets the retrying purge
  // prove that no post-enumeration write remains in flight.
  await env.DB.prepare(
    `UPDATE uploads
        SET attempt_id = NULL, attempt_started_at = NULL
      WHERE upload_id = ? AND status = 'purged' AND attempt_id = ?`
  ).bind(uploadId, attemptId).run();
}

async function readExactUploadBody(request, expectedSize) {
  if (request.body === null) {
    return { ok: expectedSize === 0, actualSize: 0, bytes: new Uint8Array() };
  }
  const bytes = new Uint8Array(expectedSize);
  const reader = request.body.getReader();
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (total + value.byteLength > expectedSize) {
      total += value.byteLength;
      await reader.cancel("body exceeds reserved size");
      return { ok: false, actualSize: total };
    }
    bytes.set(value, total);
    total += value.byteLength;
  }
  return total === expectedSize
    ? { ok: true, actualSize: total, bytes }
    : { ok: false, actualSize: total };
}


/**
 * POST /v1/uploads/complete
 *
 * Body: { upload_id: string }
 *
 * Records the upload as completed, increments stats + daily counters.
 * Idempotent — calling twice on the same upload_id is a no-op.
 *
 * Verifies the bytes actually landed in R2 (HEAD on the r2_key) — refuses
 * to mark "completed" if the object doesn't exist.
 */
async function postUploadsComplete(env, auth) {
  const parsedBody = parseJsonBytes(auth.bodyBytes);
  if (parsedBody.response) return parsedBody.response;
  const body = parsedBody.value;

  const uploadId = body.upload_id;
  if (!uploadId) return jsonResponse(400, { error: "missing_upload_id" });

  const upload = await env.DB.prepare(
    `SELECT upload_id, contributor_id, r2_key, byte_size, status, signed_at,
            content_sha256, manifest_sha256, attempt_id, expires_at
       FROM uploads WHERE upload_id = ?`
  ).bind(uploadId).first();
  if (!upload) return jsonResponse(404, { error: "upload_not_found" });
  if (upload.contributor_id !== auth.principal.contributorId) {
    return jsonResponse(403, { error: "cross_principal_upload" });
  }
  if (upload.status === "completed") {
    return jsonResponse(200, { status: "completed", upload_id: uploadId, idempotent: true });
  }
  if (upload.status === "expired") {
    return jsonResponse(410, { error: "upload_reservation_expired" });
  }
  const nowSeconds = Math.floor(Date.now() / 1000);
  if (upload.status === "pending" && nowSeconds > Number(upload.expires_at)) {
    await env.DB.prepare(
      `UPDATE uploads SET status = 'expired'
        WHERE upload_id = ? AND contributor_id = ? AND status = 'pending'`
    ).bind(uploadId, auth.principal.contributorId).run();
    return jsonResponse(410, { error: "upload_reservation_expired" });
  }
  if (upload.status !== "uploaded") {
    return jsonResponse(409, { error: "upload_not_ready", status: upload.status });
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
  if (Number(r2Object.size) !== Number(upload.byte_size)) {
    return jsonResponse(412, {
      error: "r2_object_size_mismatch",
      reserved_byte_size: upload.byte_size,
      actual_byte_size: r2Object.size,
    });
  }
  const metadata = r2Object.customMetadata || {};
  if (
    metadata.attempt_id !== upload.attempt_id
    || metadata.content_sha256 !== upload.content_sha256
    || metadata.manifest_sha256 !== upload.manifest_sha256
    || metadata.upload_id !== upload.upload_id
  ) {
    return jsonResponse(412, { error: "r2_object_metadata_mismatch" });
  }

  const nowIso = new Date().toISOString();
  const completionId = `cmp_${randomHex(16)}`;
  const completion = await env.DB.batch([
    env.DB.prepare(
      `UPDATE uploads
          SET status = 'completed', completed_at = ?, completion_id = ?
        WHERE upload_id = ? AND contributor_id = ? AND status = 'uploaded'`
    ).bind(nowIso, completionId, uploadId, auth.principal.contributorId),
    env.DB.prepare(
      `UPDATE contributors
         SET total_bytes = total_bytes + ?,
             total_uploads = total_uploads + 1,
             last_active_at = ?
       WHERE contributor_id = ?
         AND EXISTS (
           SELECT 1 FROM uploads
            WHERE upload_id = ? AND contributor_id = ?
              AND status = 'completed' AND completion_id = ?
         )`
    ).bind(
      upload.byte_size,
      nowIso,
      upload.contributor_id,
      uploadId,
      auth.principal.contributorId,
      completionId,
    ),
  ]);
  if (Number(completion[0]?.meta?.changes ?? completion[0]?.changes ?? 0) !== 1) {
    const current = await env.DB.prepare(
      `SELECT status FROM uploads WHERE upload_id = ? AND contributor_id = ?`
    ).bind(uploadId, auth.principal.contributorId).first();
    if (current?.status === "completed") {
      return jsonResponse(200, {
        status: "completed", upload_id: uploadId, idempotent: true,
      });
    }
    if (current?.status === "purged") {
      return jsonResponse(410, { error: "upload_purged", upload_id: uploadId });
    }
    return jsonResponse(409, {
      error: "upload_completion_conflict",
      upload_id: uploadId,
      status: current?.status || "missing",
    });
  }

  return jsonResponse(200, { status: "completed", upload_id: uploadId });
}

/**
 * POST /v1/revoke/cascade
 *
 * Body: { contributor_id: string, scope: "all" | "future_only" }
 *
 * Emergency containment: this endpoint requires the same admin bearer as
 * /admin routes until contributor proof-of-possession authentication ships.
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
  return await initiateRevoke(env, contributorId, scope, "admin_api");
}

/**
 * GET /v1/contributors/:id/stats
 *
 * Returns the contributor's running totals. Used by `reflex contribute --status`.
 */
async function getContributorStats(contributorId, env, principal) {
  if (!contributorId) return jsonResponse(400, { error: "missing_contributor_id" });
  if (principal && contributorId !== principal.contributorId) {
    return jsonResponse(403, { error: "cross_principal_stats" });
  }
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
    note: "Cascade advances through the private scheduled handler; status GET is read-only.",
  });
}


/**
 * GET /v1/revoke/cascade-status/<request_id>
 *
 * Returns the current cascade state for a request. This handler is strictly
 * read-only; timed progression runs only from the Worker's scheduled hook.
 */
async function getRevokeCascadeStatus(requestId, env, principal) {
  if (!requestId) return jsonResponse(400, { error: "missing_request_id" });
  const fresh = await env.DB.prepare(
    `SELECT * FROM revoke_requests WHERE request_id = ?`
  ).bind(requestId).first();
  if (!fresh) return jsonResponse(404, { error: "request_not_found" });
  if (principal && fresh.contributor_id !== principal.contributorId) {
    return jsonResponse(403, { error: "cross_principal_revoke_status" });
  }
  return jsonResponse(200, formatCascadeStatus(fresh));
}


async function adminExecuteCascade(requestId, env) {
  if (!requestId) return jsonResponse(400, { error: "missing_request_id" });
  const fresh = await loadAndProgressCascade(requestId, env, { force: true });
  if (!fresh) return jsonResponse(404, { error: "request_not_found" });
  return jsonResponse(200, formatCascadeStatus(fresh));
}


/**
 * Progress pending cascades from Cloudflare's non-HTTP scheduled event.
 * Each stage remains idempotent in loadAndProgressCascade, so retrying a
 * scheduled event is safe.
 */
async function progressPendingCascades(env) {
  const pending = await env.DB.prepare(
    `SELECT request_id FROM revoke_requests WHERE status != 'completed' LIMIT 100`
  ).all();
  for (const row of pending.results || []) {
    try {
      await loadAndProgressCascade(row.request_id, env, { force: false });
    } catch (error) {
      // One malformed historical row or transient storage failure must not
      // starve the other revocations selected by this scheduled invocation.
      // The failed cascade remains non-completed and is retried next time.
      console.error(JSON.stringify({
        message: "revoke_cascade_progress_failed",
        request_id: row.request_id,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }
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

  // Fence every upload before observing R2. A pending PUT can no longer claim,
  // and every finalizer/recovery CAS requires status='uploading'. Storage-level
  // serialization below handles already-claimed requests without a time lease.
  await env.DB.prepare(
    `UPDATE uploads
        SET status = 'purged',
            attempt_id = NULL,
            attempt_started_at = NULL
      WHERE contributor_id = ? AND status != 'purged'`
  ).bind(req.contributor_id).run();

  // The per-contributor Durable Object orders this purge against every R2.put.
  // If a claimed PUT arrived first, purge waits and deletes it. If purge arrived
  // first, the persistent terminal fence rejects that PUT before storage.
  const storage = env.CONTRIBUTOR_STORAGE.getByName(req.contributor_id);
  const purged = await storage.fetch(new Request("https://storage.internal/purge", {
    method: "POST",
    headers: { "X-Tether-Contributor-Id": req.contributor_id },
  }));
  if (!purged.ok) {
    throw new Error(`storage_fence_purge_failed status=${purged.status}`);
  }
  const purgeResult = await purged.json();
  const totalPurged = Number(purgeResult.objects_purged || 0);

  const completedAtIso = new Date().toISOString();
  await env.DB.prepare(
    `UPDATE revoke_requests
       SET r2_purge_completed_at = ?,
           r2_objects_purged = ?
       WHERE request_id = ? AND r2_purge_completed_at IS NULL`
  ).bind(completedAtIso, totalPurged, req.request_id).run();
}

async function recoverStaleUploadAttempts(env) {
  const nowSeconds = Math.floor(Date.now() / 1000);
  const rows = await env.DB.prepare(
    `SELECT upload_id, r2_key, byte_size, content_sha256, manifest_sha256,
            attempt_id, attempt_started_at, expires_at
       FROM uploads
      WHERE status = 'uploading' AND attempt_started_at <= ?
      LIMIT 100`
  ).bind(nowSeconds - UPLOAD_ATTEMPT_STALE_SECONDS).all();
  for (const row of rows.results || []) {
    const attemptKey = `${row.r2_key}/attempts/${row.attempt_id}`;
    const object = await env.CURATE_BUCKET.head(attemptKey);
    const metadata = object?.customMetadata || {};
    const valid = Boolean(
      object
      && Number(object.size) === Number(row.byte_size)
      && metadata.attempt_id === row.attempt_id
      && metadata.content_sha256 === row.content_sha256
      && metadata.manifest_sha256 === row.manifest_sha256
      && metadata.upload_id === row.upload_id,
    );
    if (valid) {
      await env.DB.prepare(
        `UPDATE uploads
            SET status = 'uploaded', r2_key = ?, uploaded_at = ?
          WHERE upload_id = ? AND status = 'uploading' AND attempt_id = ?`
      ).bind(attemptKey, nowSeconds, row.upload_id, row.attempt_id).run();
      continue;
    }
    // Claim the exact invalid attempt before deletion. If the owner wrote and
    // finalized after our inspection, this CAS loses and its authoritative
    // object must remain. If we win, the original PUT is fenced and can only
    // lose its final CAS, so deleting this abandoned attempt is safe.
    const reset = await env.DB.prepare(
      `UPDATE uploads
          SET status = CASE WHEN expires_at >= ? THEN 'pending' ELSE 'expired' END,
              attempt_id = NULL,
              attempt_started_at = NULL
        WHERE upload_id = ? AND status = 'uploading' AND attempt_id = ?`
    ).bind(nowSeconds, row.upload_id, row.attempt_id).run();
    if (Number(reset?.meta?.changes ?? reset?.changes ?? 0) === 1 && object) {
      await env.CURATE_BUCKET.delete(attemptKey);
    }
  }
}

async function pruneExpiredContributorNonces(env) {
  await env.DB.prepare(
    `DELETE FROM contributor_nonces WHERE expires_at <= ?`
  ).bind(Math.floor(Date.now() / 1000)).run();
}

function validateAnonymizationManifest(value, nowSeconds) {
  const expectedKeys = [
    "anonymizer_version",
    "content_sha256",
    "domain",
    "media_type",
    "removed_fields",
    "scan_timestamp",
    "scanner_version",
    "schema_version",
  ];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { response: jsonResponse(400, { error: "missing_anonymization_manifest" }) };
  }
  if (JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(expectedKeys)) {
    return { response: jsonResponse(400, { error: "invalid_manifest_fields" }) };
  }
  if (value.domain !== "tether.anonymization.manifest" || value.schema_version !== 1) {
    return { response: jsonResponse(400, { error: "invalid_manifest_version" }) };
  }
  if (!["application/jsonl", "application/x-parquet"].includes(value.media_type)) {
    return { response: jsonResponse(400, { error: "unsupported_manifest_media_type" }) };
  }
  if (!/^[0-9a-f]{64}$/.test(value.content_sha256 || "")) {
    return { response: jsonResponse(400, { error: "invalid_manifest_content_digest" }) };
  }
  for (const field of ["anonymizer_version", "scanner_version"]) {
    if (typeof value[field] !== "string" || value[field].length === 0 || value[field].length > 128) {
      return { response: jsonResponse(400, { error: `invalid_manifest_${field}` }) };
    }
  }
  if (!Number.isSafeInteger(value.scan_timestamp)
      || Math.abs(nowSeconds - value.scan_timestamp) > 5 * 60) {
    return { response: jsonResponse(400, { error: "stale_manifest" }) };
  }
  const removed = value.removed_fields;
  if (!removed || typeof removed !== "object" || Array.isArray(removed)
      || JSON.stringify(Object.keys(removed).sort()) !== JSON.stringify(["email", "face", "name"])) {
    return { response: jsonResponse(400, { error: "invalid_removed_fields" }) };
  }
  for (const field of ["email", "face", "name"]) {
    if (!Number.isSafeInteger(removed[field]) || removed[field] < 0) {
      return { response: jsonResponse(400, { error: "invalid_removed_field_count" }) };
    }
  }
  return { value };
}

function isSafeFileName(name) {
  if (typeof name !== "string") return false;
  if (name.length === 0 || name.length > 255) return false;
  if (name.includes("/") || name.includes("\\") || name.includes("\x00")) return false;
  if (name.startsWith(".")) return false;
  return /^[A-Za-z0-9._-]+$/.test(name);
}

function parseJsonBytes(bytes) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return { response: jsonResponse(400, { error: "invalid_utf8" }) };
  }
  try {
    const value = JSON.parse(text);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { response: jsonResponse(400, { error: "invalid_json_object" }) };
    }
    return { value };
  } catch {
    return { response: jsonResponse(400, { error: "invalid_json" }) };
  }
}

function randomHex(byteLen) {
  const bytes = randomBytes(byteLen);
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function randomBytes(byteLen) {
  const bytes = new Uint8Array(byteLen);
  crypto.getRandomValues(bytes);
  return bytes;
}

async function reservationCapability(env, contributorId, reservationKey) {
  const secret = env.UPLOAD_CAPABILITY_SECRET;
  if (typeof secret !== "string" || secret.length < 16) return null;
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  return new Uint8Array(await crypto.subtle.sign(
    "HMAC", key,
    new TextEncoder().encode(`tether.upload-capability.v1\n${contributorId}\n${reservationKey}`),
  ));
}

async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function base64urlToBytesExact(value, expectedLength) {
  if (!/^[A-Za-z0-9_-]+$/.test(value) || value.includes("=")) return null;
  try {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
    const binary = atob(base64 + "=".repeat((4 - (base64.length % 4)) % 4));
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    return bytes.byteLength === expectedLength && bytesToBase64url(bytes) === value
      ? bytes
      : null;
  } catch {
    return null;
  }
}

function constantTimeTextEqual(left, right) {
  const a = typeof left === "string" ? left : "";
  const b = typeof right === "string" ? right : "";
  const length = Math.max(a.length, b.length, 64);
  let mismatch = a.length ^ b.length;
  for (let i = 0; i < length; i += 1) {
    mismatch |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return mismatch === 0;
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
