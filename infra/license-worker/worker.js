/**
 * Reflex Pro license-server — Cloudflare Worker.
 *
 * Endpoints:
 *   GET  /healthz                          → health probe
 *   POST /admin/init                       → generate Ed25519 keypair (one-time, admin auth)
 *   POST /admin/issue                      → sign + store new license + activation code (admin auth)
 *   POST /admin/revoke                     → revoke license_id (admin auth)
 *   GET  /admin/list                       → list licenses with status (admin auth)
 *   GET  /v1/pubkey                        → return current Ed25519 public key (PEM)
 *   GET  /v1/activation/:code              → fetch signed license by one-time code
 *   POST /v1/heartbeat                     → record heartbeat + check revocation
 *   GET  /v1/revocation/:license_id        → check if license is revoked
 *
 * Security:
 *   Admin endpoints require Authorization: Bearer <ADMIN_TOKEN>. The token is
 *   stored as a Cloudflare Secret. Public/customer endpoints have no auth but
 *   rate-limit themselves via Cloudflare's built-in DDoS protection.
 *
 *   The Ed25519 private key is stored as a Cloudflare Secret named PRIVATE_KEY
 *   (PKCS8 base64). The /admin/init endpoint generates the keypair and prints
 *   the secret-set commands; you paste them into wrangler. The private key
 *   never touches your laptop or this worker's logs after init.
 *
 * Storage: Cloudflare D1, schema at schema.sql.
 */

const ADMIN_TOKEN_HEADER = "Authorization";

// Activation codes: REFLEX-XXXX-XXXX-XXXX (4-block, 16 hex chars). 24h TTL.
const ACTIVATION_CODE_TTL_MS = 24 * 60 * 60 * 1000;

// Heartbeat: customers ping daily; we accept up to 7-day staleness for the
// grace period (matches src/reflex/pro/license.py HEARTBEAT_FRESHNESS_S).
const HEARTBEAT_GRACE_MS = 7 * 24 * 60 * 60 * 1000;

// Sharing-detection threshold: a license heartbeat'd from more than this many
// distinct hardware_fingerprint values within a 7-day window is flagged.
const SHARING_FINGERPRINT_THRESHOLD = 3;

// ---------- request router ----------

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const method = request.method;
    const path = url.pathname;

    try {
      if (method === "GET" && path === "/healthz") return healthz();
      if (method === "POST" && path === "/admin/init") return await adminAuth(request, env, () => adminInit(request, env));
      if (method === "POST" && path === "/admin/issue") return await adminAuth(request, env, () => adminIssue(request, env));
      if (method === "POST" && path === "/admin/revoke") return await adminAuth(request, env, () => adminRevoke(request, env));
      if (method === "GET" && path === "/admin/list") return await adminAuth(request, env, () => adminList(request, env));
      if (method === "GET" && path === "/v1/pubkey") return await getPubkey(env);
      if (method === "GET" && path.startsWith("/v1/activation/")) return await getActivation(path.split("/").pop(), env);
      if (method === "POST" && path === "/v1/heartbeat") return await postHeartbeat(request, env);
      if (method === "GET" && path.startsWith("/v1/revocation/")) return await getRevocation(path.split("/").pop(), env);
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

// ---------- handlers ----------

function healthz() {
  return jsonResponse(200, { status: "ok", service: "reflex-license-worker" });
}

/**
 * One-time keypair generation.
 *
 * Generates an Ed25519 keypair, stores the public key in D1 (master_keys table),
 * and returns BOTH the public key (to be bundled into the reflex package) AND
 * the wrangler command to set the private key as a Cloudflare Secret. The
 * private key is returned ONCE in the response and never persisted by this
 * worker — the operator must immediately set it as a Secret.
 */
async function adminInit(request, env) {
  // Refuse if a key already exists.
  const existing = await env.DB.prepare(
    "SELECT key_id FROM master_keys WHERE retired_at IS NULL ORDER BY generated_at DESC LIMIT 1"
  ).first();
  if (existing) {
    return jsonResponse(409, {
      error: "key_already_exists",
      message: `An active key (${existing.key_id}) is in use. To rotate, POST /admin/rotate (not yet implemented).`,
    });
  }

  // Generate Ed25519 keypair via Web Crypto.
  const keyPair = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
  const pubRaw = await crypto.subtle.exportKey("raw", keyPair.publicKey);
  const privPkcs8 = await crypto.subtle.exportKey("pkcs8", keyPair.privateKey);
  const pubB64 = arrayBufferToBase64(pubRaw);
  const privB64 = arrayBufferToBase64(privPkcs8);
  const keyId = `key_${Date.now().toString(36)}_${randomHex(8)}`;

  await env.DB.prepare(
    `INSERT INTO master_keys (key_id, public_key_b64, generated_at) VALUES (?, ?, ?)`
  ).bind(keyId, pubB64, new Date().toISOString()).run();

  return jsonResponse(200, {
    status: "keypair_generated",
    key_id: keyId,
    public_key_b64: pubB64,
    private_key_b64: privB64,
    next_steps: [
      `1. Set the private key as a Worker Secret IMMEDIATELY:`,
      `     echo '${privB64}' | wrangler secret put PRIVATE_KEY`,
      `2. Paste the public_key_b64 into src/tether/pro/_public_key.py BUNDLED_PUBLIC_KEY_B64`,
      `3. Commit + push the public key change`,
      `4. The private key from this response is ONE-TIME — discard it after setting the Secret`,
    ],
  });
}

/**
 * Issue a new license: sign payload, store, generate activation code.
 *
 * Body: { customer_id, tier, expires_in_days, max_seats?, notes? }
 * Returns: { license_id, activation_code, license_payload }
 */
async function adminIssue(request, env) {
  const body = await request.json().catch(() => ({}));
  const customerId = String(body.customer_id || "").trim();
  const tier = String(body.tier || "pro").trim();
  const expiresInDays = Math.max(1, parseInt(body.expires_in_days || "30", 10));
  const maxSeats = Math.max(1, parseInt(body.max_seats || "1", 10));
  const notes = String(body.notes || "");

  if (!customerId) return jsonResponse(400, { error: "customer_id_required" });
  if (!["trial", "pro", "team", "enterprise", "educational", "research", "oss"].includes(tier)) {
    return jsonResponse(400, { error: "invalid_tier", tier });
  }

  // Build the canonical payload.
  const now = new Date();
  const expiresAt = new Date(now.getTime() + expiresInDays * 24 * 60 * 60 * 1000);
  const licenseId = `lic_${Date.now().toString(36)}_${randomHex(8)}`;
  const payload = {
    license_version: 2, // v2 = signed Ed25519
    license_id: licenseId,
    customer_id: customerId,
    tier,
    issued_at: now.toISOString(),
    expires_at: expiresAt.toISOString(),
    max_seats: maxSeats,
    hardware_binding: null, // unbound until first activation
  };

  // Sign with Ed25519.
  const privKey = await loadPrivateKey(env);
  const canonical = canonicalJson(payload);
  const sigBuf = await crypto.subtle.sign("Ed25519", privKey, new TextEncoder().encode(canonical));
  const signature = arrayBufferToBase64(sigBuf);

  const license = { ...payload, signature, key_id: await activeKeyId(env) };

  // Persist license + generate activation code.
  const activationCode = generateActivationCode();
  const activationExpires = new Date(now.getTime() + ACTIVATION_CODE_TTL_MS);

  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO licenses
       (license_id, customer_id, tier, issued_at, expires_at, max_seats, signature, key_id, notes, license_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).bind(
      licenseId, customerId, tier, payload.issued_at, payload.expires_at,
      maxSeats, signature, license.key_id, notes, JSON.stringify(license)
    ),
    env.DB.prepare(
      `INSERT INTO activation_codes (code, license_id, expires_at, used) VALUES (?, ?, ?, 0)`
    ).bind(activationCode, licenseId, activationExpires.toISOString()),
  ]);

  // Best-effort Slack notify; never blocks the response.
  if (env.SLACK_WEBHOOK_URL) {
    fetch(env.SLACK_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: `:tada: New ${tier} license issued for ${customerId} (${expiresInDays}d, ${maxSeats} seat${maxSeats > 1 ? "s" : ""})`,
      }),
    }).catch(() => {});
  }

  return jsonResponse(200, {
    license_id: licenseId,
    activation_code: activationCode,
    activation_expires_at: activationExpires.toISOString(),
    license,
  });
}

async function adminRevoke(request, env) {
  const body = await request.json().catch(() => ({}));
  const licenseId = String(body.license_id || "").trim();
  const reason = String(body.reason || "admin_revoke").trim();
  if (!licenseId) return jsonResponse(400, { error: "license_id_required" });

  const existing = await env.DB.prepare(
    "SELECT license_id, customer_id FROM licenses WHERE license_id = ?"
  ).bind(licenseId).first();
  if (!existing) return jsonResponse(404, { error: "license_not_found" });

  const revokedAt = new Date().toISOString();
  await env.DB.batch([
    env.DB.prepare(
      `INSERT OR REPLACE INTO revocation_list (license_id, revoked_at, reason) VALUES (?, ?, ?)`
    ).bind(licenseId, revokedAt, reason),
    env.DB.prepare(`UPDATE licenses SET revoked_at = ? WHERE license_id = ?`)
      .bind(revokedAt, licenseId),
  ]);

  if (env.SLACK_WEBHOOK_URL) {
    fetch(env.SLACK_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: `:warning: License revoked: ${licenseId} (customer ${existing.customer_id}). Reason: ${reason}`,
      }),
    }).catch(() => {});
  }

  return jsonResponse(200, { license_id: licenseId, revoked_at: revokedAt, reason });
}

async function adminList(request, env) {
  const url = new URL(request.url);
  const limit = Math.min(500, Math.max(1, parseInt(url.searchParams.get("limit") || "100", 10)));
  const rows = await env.DB.prepare(
    `SELECT l.license_id, l.customer_id, l.tier, l.issued_at, l.expires_at, l.max_seats,
            l.revoked_at,
            (SELECT COUNT(DISTINCT hardware_fingerprint) FROM heartbeats
             WHERE heartbeats.license_id = l.license_id
             AND server_timestamp > datetime('now', '-7 days')) AS distinct_fingerprints_7d,
            (SELECT MAX(server_timestamp) FROM heartbeats
             WHERE heartbeats.license_id = l.license_id) AS last_heartbeat
     FROM licenses l
     ORDER BY l.issued_at DESC
     LIMIT ?`
  ).bind(limit).all();
  return jsonResponse(200, { licenses: rows.results });
}

async function getPubkey(env) {
  const row = await env.DB.prepare(
    "SELECT key_id, public_key_b64, generated_at FROM master_keys WHERE retired_at IS NULL ORDER BY generated_at DESC LIMIT 1"
  ).first();
  if (!row) return jsonResponse(404, { error: "no_active_key", message: "Run POST /admin/init first." });
  return jsonResponse(200, { key_id: row.key_id, public_key_b64: row.public_key_b64, generated_at: row.generated_at });
}

async function getActivation(code, env) {
  if (!code || !/^REFLEX-[A-Z0-9-]+$/.test(code)) return jsonResponse(400, { error: "invalid_code_format" });

  const row = await env.DB.prepare(
    "SELECT license_id, expires_at, used FROM activation_codes WHERE code = ?"
  ).bind(code).first();
  if (!row) return jsonResponse(404, { error: "code_not_found_or_expired" });
  if (row.used) return jsonResponse(410, { error: "code_already_used" });
  if (new Date(row.expires_at) < new Date()) return jsonResponse(410, { error: "code_expired" });

  const license = await env.DB.prepare("SELECT license_json FROM licenses WHERE license_id = ?")
    .bind(row.license_id).first();
  if (!license) return jsonResponse(500, { error: "license_missing", license_id: row.license_id });

  // Mark code used so it's one-shot.
  await env.DB.prepare("UPDATE activation_codes SET used = 1, used_at = ? WHERE code = ?")
    .bind(new Date().toISOString(), code).run();

  return jsonResponse(200, { license: JSON.parse(license.license_json) });
}

async function postHeartbeat(request, env) {
  const body = await request.json().catch(() => ({}));
  const licenseId = String(body.license_id || "").trim();
  const hardwareFingerprint = String(body.hardware_fingerprint || "").trim();
  // Accept tether_version (current clients) or reflex_version (legacy). Without
  // this the renamed client sends tether_version and this column was recorded
  // as all-"unknown" — silent version-analytics loss. D1 column stays
  // reflex_version for continuity.
  const tetherVersion = String(body.tether_version || body.reflex_version || "unknown").slice(0, 64);
  if (!licenseId || !hardwareFingerprint) {
    return jsonResponse(400, { error: "license_id_and_hardware_fingerprint_required" });
  }

  // Check revocation first — heartbeat from a revoked license is rejected loudly.
  const revoked = await env.DB.prepare(
    "SELECT revoked_at, reason FROM revocation_list WHERE license_id = ?"
  ).bind(licenseId).first();
  if (revoked) {
    return jsonResponse(403, { revoked: true, revoked_at: revoked.revoked_at, reason: revoked.reason });
  }

  // Check expiry from licenses table.
  const license = await env.DB.prepare(
    "SELECT customer_id, expires_at, max_seats FROM licenses WHERE license_id = ?"
  ).bind(licenseId).first();
  if (!license) return jsonResponse(404, { error: "license_not_found" });
  if (new Date(license.expires_at) < new Date()) {
    return jsonResponse(403, { expired: true, expires_at: license.expires_at });
  }

  // Best-effort country geo (Cloudflare Cf-IPCountry header).
  const country = request.headers.get("Cf-IPCountry") || "??";

  // Record heartbeat (we do NOT log Cf-Connecting-IP, only the country).
  await env.DB.prepare(
    `INSERT INTO heartbeats
     (license_id, hardware_fingerprint, ip_country, reflex_version, server_timestamp)
     VALUES (?, ?, ?, ?, ?)`
  ).bind(licenseId, hardwareFingerprint, country, tetherVersion, new Date().toISOString()).run();

  // Sharing-detection check (async; doesn't block response).
  detectSharing(licenseId, env).catch((e) => console.error("sharing-detect failed:", e.message));

  return jsonResponse(200, {
    valid: true,
    license_id: licenseId,
    expires_at: license.expires_at,
    max_seats: license.max_seats,
  });
}

async function getRevocation(licenseId, env) {
  const row = await env.DB.prepare(
    "SELECT revoked_at, reason FROM revocation_list WHERE license_id = ?"
  ).bind(licenseId).first();
  if (!row) return jsonResponse(200, { license_id: licenseId, revoked: false });
  return jsonResponse(200, { license_id: licenseId, revoked: true, revoked_at: row.revoked_at, reason: row.reason });
}

// ---------- abuse detection ----------

async function detectSharing(licenseId, env) {
  // If we already flagged this license for sharing in the last 24h, skip
  // (avoid Slack-spam loops).
  const recent = await env.DB.prepare(
    `SELECT 1 FROM abuse_signals
     WHERE license_id = ? AND signal_type = 'sharing'
     AND detected_at > datetime('now', '-1 day')
     LIMIT 1`
  ).bind(licenseId).first();
  if (recent) return;

  const row = await env.DB.prepare(
    `SELECT COUNT(DISTINCT hardware_fingerprint) AS distinct_fps,
            COUNT(DISTINCT ip_country) AS distinct_countries
     FROM heartbeats
     WHERE license_id = ?
     AND server_timestamp > datetime('now', '-7 days')`
  ).bind(licenseId).first();

  const isSharing = row && row.distinct_fps > SHARING_FINGERPRINT_THRESHOLD;
  if (!isSharing) return;

  // Check override list (false-positive suppression).
  const overridden = await env.DB.prepare(
    "SELECT 1 FROM override_list WHERE license_id = ? LIMIT 1"
  ).bind(licenseId).first();
  if (overridden) return;

  const detail = JSON.stringify({
    distinct_fingerprints_7d: row.distinct_fps,
    distinct_countries_7d: row.distinct_countries,
    threshold: SHARING_FINGERPRINT_THRESHOLD,
  });
  await env.DB.prepare(
    `INSERT INTO abuse_signals (license_id, signal_type, details, detected_at)
     VALUES (?, 'sharing', ?, ?)`
  ).bind(licenseId, detail, new Date().toISOString()).run();

  if (env.SLACK_WEBHOOK_URL) {
    await fetch(env.SLACK_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: `:rotating_light: Sharing detected: license ${licenseId} active on ${row.distinct_fps} distinct hardware fingerprints in last 7d (${row.distinct_countries} countries). Threshold ${SHARING_FINGERPRINT_THRESHOLD}.`,
      }),
    }).catch(() => {});
  }
}

// ---------- helpers ----------

async function loadPrivateKey(env) {
  if (!env.PRIVATE_KEY) {
    throw new Error("PRIVATE_KEY secret not set. Run POST /admin/init then `wrangler secret put PRIVATE_KEY`.");
  }
  const pkcs8 = base64ToArrayBuffer(env.PRIVATE_KEY);
  return await crypto.subtle.importKey("pkcs8", pkcs8, { name: "Ed25519" }, false, ["sign"]);
}

async function activeKeyId(env) {
  const row = await env.DB.prepare(
    "SELECT key_id FROM master_keys WHERE retired_at IS NULL ORDER BY generated_at DESC LIMIT 1"
  ).first();
  return row ? row.key_id : null;
}

function canonicalJson(obj) {
  // Sort keys deterministically — must match the customer-side verifier.
  if (obj === null || typeof obj !== "object") return JSON.stringify(obj);
  if (Array.isArray(obj)) return "[" + obj.map(canonicalJson).join(",") + "]";
  const keys = Object.keys(obj).sort();
  return "{" + keys.map((k) => JSON.stringify(k) + ":" + canonicalJson(obj[k])).join(",") + "}";
}

function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function randomHex(nBytes) {
  const bytes = new Uint8Array(nBytes);
  crypto.getRandomValues(bytes);
  return Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function generateActivationCode() {
  // REFLEX-XXXX-XXXX-XXXX (4 hex blocks of 4 chars each = 16 hex chars total)
  const blocks = [];
  for (let i = 0; i < 3; i++) blocks.push(randomHex(2).toUpperCase());
  return `REFLEX-${blocks.join("-")}`;
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
