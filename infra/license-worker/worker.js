/**
 * Reflex Pro license-server — Cloudflare Worker.
 *
 * Endpoints:
 *   GET  /healthz                          → health probe
 *   POST /admin/init                       → generate Ed25519 keypair (one-time, admin auth)
 *   POST /admin/issue                      → sign + store new license + activation code (admin auth)
 *   POST /admin/revoke                     → revoke license_id (admin auth)
 *   GET  /admin/list                       → list licenses with status (admin auth)
 *   GET  /admin/signer                     → verify configured signer binding (admin auth)
 *   GET  /v1/pubkey                        → return current Ed25519 public key (PEM)
 *   POST /v1/activation/:code              → bind hardware + fetch signed v2 license
 *   POST /v1/heartbeat                     → return signed revocation attestation
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

const HEARTBEAT_VALIDITY_SECONDS = 24 * 60 * 60;
const HEARTBEAT_DOMAIN = "tether.license.heartbeat";

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
      if (method === "GET" && path === "/admin/signer") return await adminAuth(request, env, () => adminSigner(env));
      if (method === "GET" && path === "/v1/pubkey") return await getPubkey(env);
      if (method === "POST" && path.startsWith("/v1/activation/")) return await postActivation(path.split("/").pop(), request, env);
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
      `2. Bind that secret to this exact public-key row:`,
      `     echo '${keyId}' | wrangler secret put SIGNING_KEY_ID`,
      `3. Add public_key_b64 under ${keyId} in TRUSTED_PUBLIC_KEYS_B64`,
      `4. Commit + release the trusted public key before serving signed responses`,
      `5. The private key from this response is ONE-TIME — discard it after setting the Secret`,
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

  // Sign with the key-id/private-key pair verified by loadSigner().
  const signed = await signPayload(payload, env, { base64url: false });
  const signature = signed.signature;
  const license = { ...payload, signature, key_id: signed.keyId };

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

async function adminSigner(env) {
  const { keyId } = await loadSigner(env);
  return jsonResponse(200, { verified: true, key_id: keyId });
}

async function getPubkey(env) {
  if (!env.SIGNING_KEY_ID) {
    return jsonResponse(503, { error: "signing_key_id_not_configured" });
  }
  const row = await env.DB.prepare(
    "SELECT key_id, public_key_b64, generated_at FROM master_keys WHERE key_id = ? AND retired_at IS NULL"
  ).bind(env.SIGNING_KEY_ID).first();
  if (!row) return jsonResponse(404, { error: "configured_signing_key_not_active" });
  return jsonResponse(200, { key_id: row.key_id, public_key_b64: row.public_key_b64, generated_at: row.generated_at });
}

async function postActivation(code, request, env) {
  if (!code || !/^REFLEX-[A-Z0-9-]+$/.test(code)) return jsonResponse(400, { error: "invalid_code_format" });

  const body = await request.json().catch(() => ({}));
  const hardwareBinding = body.hardware_binding;
  if (!isValidHardwareBinding(hardwareBinding)) {
    return jsonResponse(400, { error: "valid_hardware_binding_required" });
  }

  const row = await env.DB.prepare(
    "SELECT license_id, expires_at, used FROM activation_codes WHERE code = ?"
  ).bind(code).first();
  if (!row) return jsonResponse(404, { error: "code_not_found_or_expired" });
  if (row.used) return jsonResponse(410, { error: "code_already_used" });
  if (new Date(row.expires_at) < new Date()) return jsonResponse(410, { error: "code_expired" });

  // Atomically consume the capability before issuing a hardware-bound
  // license. Concurrent redeemers cannot obtain two valid bindings.
  const claimed = await env.DB.prepare(
    "UPDATE activation_codes SET used = 1, used_at = ? WHERE code = ? AND used = 0"
  ).bind(new Date().toISOString(), code).run();
  const claimChanges = claimed?.meta?.changes ?? claimed?.changes ?? 0;
  if (claimChanges !== 1) return jsonResponse(410, { error: "code_already_used" });

  const stored = await env.DB.prepare("SELECT license_json FROM licenses WHERE license_id = ?")
    .bind(row.license_id).first();
  if (!stored) return jsonResponse(500, { error: "license_missing", license_id: row.license_id });

  const issued = JSON.parse(stored.license_json);
  const payload = {
    license_version: 2,
    license_id: issued.license_id,
    customer_id: issued.customer_id,
    tier: issued.tier,
    issued_at: issued.issued_at,
    expires_at: issued.expires_at,
    max_seats: issued.max_seats,
    hardware_binding: hardwareBinding,
  };
  const signed = await signPayload(payload, env, { base64url: false });
  const license = { ...payload, signature: signed.signature, key_id: signed.keyId };

  // Persist exactly the hardware-bound object returned to the customer.
  await env.DB.prepare(
    "UPDATE licenses SET signature = ?, key_id = ?, license_json = ? WHERE license_id = ?"
  ).bind(signed.signature, signed.keyId, JSON.stringify(license), row.license_id).run();

  return jsonResponse(200, { license });
}

async function postHeartbeat(request, env) {
  const body = await request.json().catch(() => ({}));
  const licenseId = String(body.license_id || "").trim();
  const hardwareFingerprint = String(body.hardware_fingerprint || "").trim();
  const tetherVersion = String(body.tether_version || "unknown").slice(0, 64);
  const requestNonce = String(body.request_nonce || "");
  if (!licenseId || !hardwareFingerprint || !isNonce(requestNonce)) {
    return jsonResponse(400, { error: "license_id_hardware_fingerprint_and_nonce_required" });
  }

  const license = await env.DB.prepare(
    "SELECT customer_id, expires_at, max_seats FROM licenses WHERE license_id = ?"
  ).bind(licenseId).first();
  if (!license) return jsonResponse(404, { error: "license_not_found" });

  const revoked = await env.DB.prepare(
    "SELECT revoked_at, reason FROM revocation_list WHERE license_id = ?"
  ).bind(licenseId).first();

  const issuedAt = Math.floor(Date.now() / 1000);
  const expiresAt = Math.floor(new Date(license.expires_at).getTime() / 1000);
  const status = revoked ? "revoked" : expiresAt <= issuedAt ? "expired" : "active";
  const validUntil = Math.min(issuedAt + HEARTBEAT_VALIDITY_SECONDS, expiresAt);
  const attestation = {
    domain: HEARTBEAT_DOMAIN,
    issued_at: issuedAt,
    key_id: String(env.SIGNING_KEY_ID || ""),
    license_id: licenseId,
    request_nonce: requestNonce,
    status,
    v: 1,
    valid_until: validUntil,
  };
  const signed = await signPayload(attestation, env, { base64url: true });
  const signedAttestation = { ...attestation, signature: signed.signature };

  if (status !== "active") return jsonResponse(200, signedAttestation);

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

  return jsonResponse(200, signedAttestation);
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

async function signPayload(payload, env, { base64url = false } = {}) {
  const { keyId, privateKey } = await loadSigner(env);
  const signatureBuffer = await crypto.subtle.sign(
    "Ed25519", privateKey, new TextEncoder().encode(canonicalJson(payload)),
  );
  return {
    keyId,
    signature: base64url
      ? arrayBufferToBase64Url(signatureBuffer)
      : arrayBufferToBase64(signatureBuffer),
  };
}

async function loadSigner(env) {
  const keyId = String(env.SIGNING_KEY_ID || "");
  if (!keyId) throw new Error("SIGNING_KEY_ID secret not set");
  const row = await env.DB.prepare(
    "SELECT public_key_b64 FROM master_keys WHERE key_id = ? AND retired_at IS NULL"
  ).bind(keyId).first();
  if (!row) throw new Error(`SIGNING_KEY_ID ${keyId} is missing or retired`);

  const privateKey = await loadPrivateKey(env);
  const publicKey = await crypto.subtle.importKey(
    "raw", base64ToArrayBuffer(row.public_key_b64), { name: "Ed25519" }, false, ["verify"],
  );
  const challenge = new TextEncoder().encode("tether.signer.key-binding.v1");
  const proof = await crypto.subtle.sign("Ed25519", privateKey, challenge);
  if (!(await crypto.subtle.verify("Ed25519", publicKey, proof, challenge))) {
    throw new Error(`SIGNING_KEY_ID ${keyId} does not match PRIVATE_KEY`);
  }
  return { keyId, privateKey };
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

function arrayBufferToBase64Url(buf) {
  return arrayBufferToBase64(buf)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function isNonce(value) {
  if (!/^[A-Za-z0-9_-]{22}$/.test(value)) return false;
  try {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/") + "==";
    return base64ToArrayBuffer(base64).byteLength === 16;
  } catch (_error) {
    return false;
  }
}

function isValidHardwareBinding(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  if (Object.keys(value).sort().join(",") !== "cpu_count,gpu_name,gpu_uuid") return false;
  return (
    typeof value.gpu_uuid === "string" && value.gpu_uuid.length > 0 && value.gpu_uuid.length <= 256 &&
    typeof value.gpu_name === "string" && value.gpu_name.length > 0 && value.gpu_name.length <= 256 &&
    Number.isInteger(value.cpu_count) && value.cpu_count > 0 && value.cpu_count <= 1048576
  );
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
