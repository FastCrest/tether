const AUTH_VERSION = "1";
const AUTH_DOMAIN = "tether.contrib.request";
const MAX_CLOCK_SKEW_SECONDS = 5 * 60;
const NONCE_TTL_SECONDS = 10 * 60;
const MAX_AUTH_BODY_BYTES = 1024 * 1024;
const EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

const AUTH_HEADERS = Object.freeze({
  version: "X-Tether-Auth-Version",
  contributorId: "X-Tether-Contributor-Id",
  keyId: "X-Tether-Key-Id",
  timestamp: "X-Tether-Timestamp",
  nonce: "X-Tether-Nonce",
  bodySha256: "X-Tether-Content-SHA256",
  signature: "X-Tether-Signature",
});

const CONTRIBUTOR_ID_RE = /^ctr_[0-9a-f]{32}$/;
const KEY_ID_RE = /^key_[0-9a-f]{32}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const BASE64URL_RE = /^[A-Za-z0-9_-]+$/;

function errorResponse(status, error, details = {}) {
  return new Response(JSON.stringify({ error, ...details }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function invalidUnicode(value) {
  for (let i = 0; i < value.length; i += 1) {
    const unit = value.charCodeAt(i);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      if (i + 1 >= value.length) return true;
      const next = value.charCodeAt(i + 1);
      if (next < 0xdc00 || next > 0xdfff) return true;
      i += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      return true;
    }
  }
  return false;
}

/** RFC 8785 / JCS serialization for JSON-compatible values. */
export function canonicalizeJson(value) {
  if (value === null || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") {
    if (invalidUnicode(value)) throw new TypeError("invalid Unicode scalar value");
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("non-finite JSON number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalizeJson(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const keys = Object.keys(value).sort();
    for (const key of keys) {
      if (invalidUnicode(key)) throw new TypeError("invalid Unicode object key");
      if (value[key] === undefined) throw new TypeError("undefined is not JSON");
    }
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalizeJson(value[key])}`).join(",")}}`;
  }
  throw new TypeError("value is not JSON-compatible");
}

function compareCodePoints(left, right) {
  const a = Array.from(left, (char) => char.codePointAt(0));
  const b = Array.from(right, (char) => char.codePointAt(0));
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

export function canonicalQuery(url) {
  return Array.from(url.searchParams.entries()).sort((left, right) => {
    const keyOrder = compareCodePoints(left[0], right[0]);
    return keyOrder === 0 ? compareCodePoints(left[1], right[1]) : keyOrder;
  });
}

export function buildRequestEnvelope({
  request,
  contributorId,
  keyId,
  timestamp,
  nonce,
  bodySha256,
}) {
  const url = new URL(request.url);
  validateUrlEncoding(url);
  return {
    body_sha256: bodySha256,
    contributor_id: contributorId,
    domain: AUTH_DOMAIN,
    key_id: keyId,
    method: request.method.toUpperCase(),
    nonce,
    path: url.pathname,
    query: canonicalQuery(url),
    timestamp,
    v: 1,
  };
}

function validateUrlEncoding(url) {
  // URLSearchParams replaces malformed UTF-8 with U+FFFD. Validate the raw
  // percent-encoded components first so malformed encodings cannot acquire a
  // different canonical meaning before signing.
  decodeURIComponent(url.pathname);
  for (const component of url.search.slice(1).split(/[&=]/u)) {
    decodeURIComponent(component.replace(/\+/g, " "));
  }
}

function bytesToHex(bytes) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function base64urlToBytes(value, expectedLength) {
  if (!BASE64URL_RE.test(value) || value.includes("=")) return null;
  try {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    if (bytes.byteLength !== expectedLength) return null;
    if (bytesToBase64url(bytes) !== value) return null;
    return bytes;
  } catch {
    return null;
  }
}

export function bytesToBase64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/u, "");
}

async function sha256(bytes) {
  return new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
}

export async function deriveContributorIdentifiers(publicKeyBase64url) {
  const publicKey = base64urlToBytes(publicKeyBase64url, 32);
  if (!publicKey) throw new TypeError("public key must be 32-byte unpadded base64url");
  const digest = bytesToHex(await sha256(publicKey));
  return {
    contributorId: `ctr_${digest.slice(0, 32)}`,
    keyId: `key_${digest.slice(0, 32)}`,
    publicKey,
  };
}

function singleHeader(headers, name) {
  const value = headers.get(name);
  // Fetch coalesces duplicate request headers with a comma. None of the auth
  // encodings permits commas, so this rejects both duplicates and malformed
  // single values in runtimes where raw header multiplicity is unavailable.
  if (value === null || value.includes(",")) return null;
  return value;
}

async function readBoundedBody(request) {
  const declared = request.headers.get("Content-Length");
  if (declared !== null) {
    if (!/^(0|[1-9][0-9]*)$/.test(declared)) {
      return { response: errorResponse(400, "invalid_content_length") };
    }
    if (Number(declared) > MAX_AUTH_BODY_BYTES) {
      return { response: errorResponse(413, "auth_body_too_large") };
    }
  }
  if (request.body === null) return { bytes: new Uint8Array() };

  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > MAX_AUTH_BODY_BYTES) {
      await reader.cancel("auth body too large");
      return { response: errorResponse(413, "auth_body_too_large") };
    }
    chunks.push(value);
  }
  if (declared !== null && Number(declared) !== total) {
    return { response: errorResponse(400, "content_length_mismatch") };
  }
  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { bytes: body };
}

function parseHeaders(request) {
  const values = {};
  for (const [field, name] of Object.entries(AUTH_HEADERS)) {
    values[field] = singleHeader(request.headers, name);
    if (values[field] === null) {
      return { response: errorResponse(401, "missing_or_duplicate_auth_header", { header: name }) };
    }
  }
  if (values.version !== AUTH_VERSION) {
    return { response: errorResponse(401, "unsupported_auth_version") };
  }
  if (!CONTRIBUTOR_ID_RE.test(values.contributorId) || !KEY_ID_RE.test(values.keyId)) {
    return { response: errorResponse(400, "malformed_auth_id") };
  }
  if (!SHA256_RE.test(values.bodySha256)) {
    return { response: errorResponse(400, "malformed_body_digest") };
  }
  if (!/^(0|[1-9][0-9]*)$/.test(values.timestamp)) {
    return { response: errorResponse(400, "malformed_timestamp") };
  }
  const timestamp = Number(values.timestamp);
  if (!Number.isSafeInteger(timestamp)) {
    return { response: errorResponse(400, "malformed_timestamp") };
  }
  if (!base64urlToBytes(values.nonce, 16)) {
    return { response: errorResponse(400, "malformed_nonce") };
  }
  const signature = base64urlToBytes(values.signature, 64);
  if (!signature) {
    return { response: errorResponse(400, "malformed_signature") };
  }
  return { values, timestamp, signature };
}

async function verifyEnvelopeSignature(envelope, signature, publicKey) {
  try {
    const key = await crypto.subtle.importKey(
      "raw",
      publicKey,
      { name: "Ed25519" },
      false,
      ["verify"],
    );
    return await crypto.subtle.verify(
      "Ed25519",
      key,
      signature,
      new TextEncoder().encode(canonicalizeJson(envelope)),
    );
  } catch {
    return false;
  }
}

async function consumeNonce(env, contributorId, nonce, nowSeconds) {
  const result = await env.DB.prepare(
    `INSERT INTO contributor_nonces (contributor_id, nonce, expires_at)
       VALUES (?, ?, ?)
       ON CONFLICT(contributor_id, nonce) DO UPDATE SET
         expires_at = excluded.expires_at
       WHERE contributor_nonces.expires_at <= ?`,
  ).bind(
    contributorId,
    nonce,
    nowSeconds + NONCE_TTL_SECONDS,
    nowSeconds,
  ).run();
  return Number(result?.meta?.changes ?? result?.changes ?? 0) === 1;
}

async function verifyRequestWithKey(request, env, publicKeyBase64url, options = {}) {
  const parsed = parseHeaders(request);
  if (parsed.response) return { ok: false, response: parsed.response };
  const nowSeconds = options.nowSeconds ?? Math.floor(Date.now() / 1000);
  if (Math.abs(nowSeconds - parsed.timestamp) > MAX_CLOCK_SKEW_SECONDS) {
    return { ok: false, response: errorResponse(401, "timestamp_out_of_range") };
  }
  const bodyResult = await readBoundedBody(request);
  if (bodyResult.response) return { ok: false, response: bodyResult.response };
  const bodyBytes = bodyResult.bytes;
  const actualBodySha256 = bytesToHex(await sha256(bodyBytes));
  if (actualBodySha256 !== parsed.values.bodySha256) {
    return { ok: false, response: errorResponse(401, "body_digest_mismatch") };
  }
  const publicKey = base64urlToBytes(publicKeyBase64url, 32);
  if (!publicKey) {
    return { ok: false, response: errorResponse(401, "invalid_public_key") };
  }
  let envelope;
  try {
    envelope = buildRequestEnvelope({
      request,
      contributorId: parsed.values.contributorId,
      keyId: parsed.values.keyId,
      timestamp: parsed.timestamp,
      nonce: parsed.values.nonce,
      bodySha256: actualBodySha256,
    });
  } catch {
    return { ok: false, response: errorResponse(400, "invalid_url_encoding") };
  }
  if (!(await verifyEnvelopeSignature(envelope, parsed.signature, publicKey))) {
    return { ok: false, response: errorResponse(401, "invalid_signature") };
  }
  if (options.consumeNonce !== false) {
    if (!(await consumeNonce(env, parsed.values.contributorId, parsed.values.nonce, nowSeconds))) {
      return { ok: false, response: errorResponse(409, "nonce_replay") };
    }
  }
  return {
    ok: true,
    bodyBytes,
    envelope,
    headers: parsed.values,
    nowSeconds,
  };
}

export async function authenticateContributorRequest(request, env, options = {}) {
  const parsed = parseHeaders(request);
  if (parsed.response) return { ok: false, response: parsed.response };
  const row = await env.DB.prepare(
    `SELECT k.public_key_base64url, k.status AS key_status,
            c.tier, c.revoked_at
       FROM contributor_keys k
       JOIN contributors c ON c.contributor_id = k.contributor_id
      WHERE k.contributor_id = ? AND k.key_id = ?`,
  ).bind(parsed.values.contributorId, parsed.values.keyId).first();
  if (!row || row.key_status !== "active") {
    return { ok: false, response: errorResponse(401, "inactive_or_unknown_key") };
  }
  if (row.revoked_at) {
    return { ok: false, response: errorResponse(403, "contributor_revoked") };
  }
  const verified = await verifyRequestWithKey(request, env, row.public_key_base64url, options);
  if (!verified.ok) return verified;
  return {
    ...verified,
    principal: {
      contributorId: parsed.values.contributorId,
      keyId: parsed.values.keyId,
      tier: row.tier,
    },
  };
}

export async function registerContributor(request, env, options = {}) {
  const publicKeyBase64url = singleHeader(request.headers, "X-Tether-Public-Key");
  if (!publicKeyBase64url) {
    return { ok: false, response: errorResponse(400, "missing_or_duplicate_public_key") };
  }
  let derived;
  try {
    derived = await deriveContributorIdentifiers(publicKeyBase64url);
  } catch {
    return { ok: false, response: errorResponse(400, "malformed_public_key") };
  }
  const parsed = parseHeaders(request);
  if (parsed.response) return { ok: false, response: parsed.response };
  if (parsed.values.contributorId !== derived.contributorId || parsed.values.keyId !== derived.keyId) {
    return { ok: false, response: errorResponse(400, "derived_id_mismatch") };
  }
  const verified = await verifyRequestWithKey(request, env, publicKeyBase64url, options);
  if (!verified.ok) return verified;
  if (verified.bodyBytes.byteLength !== 0 || verified.headers.bodySha256 !== EMPTY_SHA256) {
    return { ok: false, response: errorResponse(400, "registration_body_must_be_empty") };
  }

  const existing = await env.DB.prepare(
    `SELECT c.revoked_at, c.tier, k.public_key_base64url, k.status AS key_status
       FROM contributors c
       LEFT JOIN contributor_keys k
         ON k.contributor_id = c.contributor_id AND k.key_id = ?
      WHERE c.contributor_id = ?`,
  ).bind(derived.keyId, derived.contributorId).first();
  if (existing?.revoked_at) {
    return { ok: false, response: errorResponse(403, "contributor_revoked") };
  }
  if (existing) {
    if (existing.public_key_base64url !== publicKeyBase64url || existing.key_status !== "active") {
      return { ok: false, response: errorResponse(409, "registration_conflict") };
    }
    return { ok: true, status: 200, contributorId: derived.contributorId, keyId: derived.keyId, tier: existing.tier };
  }

  const nowIso = new Date(verified.nowSeconds * 1000).toISOString();
  await env.DB.batch([
    env.DB.prepare(
      `INSERT INTO contributors (contributor_id, tier, first_seen_at, last_active_at)
       VALUES (?, 'free', ?, ?)`,
    ).bind(derived.contributorId, nowIso, nowIso),
    env.DB.prepare(
      `INSERT INTO contributor_keys
         (key_id, contributor_id, public_key_base64url, status, created_at)
       VALUES (?, ?, ?, 'active', ?)`,
    ).bind(derived.keyId, derived.contributorId, publicKeyBase64url, verified.nowSeconds),
  ]);
  return { ok: true, status: 201, contributorId: derived.contributorId, keyId: derived.keyId, tier: "free" };
}

export function parseCanonicalJsonBody(bodyBytes) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bodyBytes);
  } catch {
    return { response: errorResponse(400, "invalid_utf8") };
  }
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    return { response: errorResponse(400, "invalid_json") };
  }
  try {
    if (canonicalizeJson(value) !== text) {
      return { response: errorResponse(400, "non_canonical_json") };
    }
  } catch {
    return { response: errorResponse(400, "invalid_json_value") };
  }
  return { value };
}

export async function rotateContributorKey(request, env, options = {}) {
  const authenticated = await authenticateContributorRequest(request, env, options);
  if (!authenticated.ok) return authenticated;
  const parsedBody = parseCanonicalJsonBody(authenticated.bodyBytes);
  if (parsedBody.response) return { ok: false, response: parsedBody.response };
  const publicKeyBase64url = parsedBody.value?.public_key;
  if (typeof publicKeyBase64url !== "string" || Object.keys(parsedBody.value).length !== 1) {
    return { ok: false, response: errorResponse(400, "invalid_rotation_body") };
  }
  let derived;
  try {
    derived = await deriveContributorIdentifiers(publicKeyBase64url);
  } catch {
    return { ok: false, response: errorResponse(400, "malformed_public_key") };
  }
  if (derived.keyId === authenticated.principal.keyId) {
    return { ok: false, response: errorResponse(400, "rotation_key_unchanged") };
  }
  const existing = await env.DB.prepare(
    `SELECT contributor_id FROM contributor_keys WHERE key_id = ?`,
  ).bind(derived.keyId).first();
  if (existing) {
    return { ok: false, response: errorResponse(409, "key_already_registered") };
  }
  const now = authenticated.nowSeconds;
  await env.DB.batch([
    env.DB.prepare(
      `UPDATE contributor_keys
          SET status = 'inactive', deactivated_at = ?
        WHERE contributor_id = ? AND key_id = ? AND status = 'active'`,
    ).bind(now, authenticated.principal.contributorId, authenticated.principal.keyId),
    env.DB.prepare(
      `INSERT INTO contributor_keys
         (key_id, contributor_id, public_key_base64url, status, created_at)
       VALUES (?, ?, ?, 'active', ?)`,
    ).bind(derived.keyId, authenticated.principal.contributorId, publicKeyBase64url, now),
  ]);
  return {
    ok: true,
    contributorId: authenticated.principal.contributorId,
    oldKeyId: authenticated.principal.keyId,
    keyId: derived.keyId,
  };
}

export const contributorAuthConstants = Object.freeze({
  AUTH_HEADERS,
  EMPTY_SHA256,
  MAX_CLOCK_SKEW_SECONDS,
  NONCE_TTL_SECONDS,
});
