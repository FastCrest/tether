import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import worker from "../worker.js";

function b64(buffer) {
  return Buffer.from(buffer).toString("base64");
}

function decodeB64Url(value) {
  return Buffer.from(value.replace(/-/g, "+").replace(/_/g, "/"), "base64");
}

function canonical(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
}

async function signingEnv({ expiresAt, revoked = null, keyId = "key_test", publicKeyOverride = null, retired = false } = {}) {
  const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
  const privatePkcs8 = await crypto.subtle.exportKey("pkcs8", pair.privateKey);
  const publicRaw = await crypto.subtle.exportKey("raw", pair.publicKey);
  const storedPublicB64 = publicKeyOverride || b64(publicRaw);
  const calls = [];
  const license = {
    customer_id: "acme",
    expires_at: expiresAt || new Date(Date.now() + 2 * 86400000).toISOString(),
    max_seats: 1,
  };
  const DB = {
    calls,
    async batch(statements) {
      for (const statement of statements) await statement.run();
      return statements.map(() => ({ success: true }));
    },
    prepare(sql) {
      const call = { sql, bindings: [], operation: null };
      calls.push(call);
      return {
        bind(...bindings) { call.bindings = bindings; return this; },
        async first() {
          call.operation = "first";
          if (sql.includes("FROM activation_codes")) {
            return { license_id: "lic_test", expires_at: new Date(Date.now() + 3600000).toISOString(), used: 0 };
          }
          if (sql.includes("SELECT license_json")) {
            return { license_json: JSON.stringify({
              license_version: 2,
              license_id: "lic_test",
              customer_id: "acme",
              tier: "pro",
              issued_at: new Date(Date.now() - 1000).toISOString(),
              expires_at: license.expires_at,
              max_seats: 1,
              hardware_binding: null,
            }) };
          }
          if (sql.includes("FROM licenses WHERE license_id")) return license;
          if (sql.includes("FROM revocation_list")) return revoked;
          if (sql.includes("FROM master_keys")) {
            return retired ? null : {
              key_id: keyId,
              public_key_b64: storedPublicB64,
              generated_at: new Date().toISOString(),
            };
          }
          if (sql.includes("FROM abuse_signals")) return { exists: 1 };
          return null;
        },
        async run() { call.operation = "run"; return { success: true, meta: { changes: 1 } }; },
      };
    },
  };
  return {
    env: { DB, PRIVATE_KEY: b64(privatePkcs8), SIGNING_KEY_ID: keyId },
    publicKey: pair.publicKey,
    calls,
  };
}

function heartbeatRequest(nonce = "MDEyMzQ1Njc4OWFiY2RlZg") {
  return new Request("https://worker.test/v1/heartbeat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      license_id: "lic_test",
      hardware_fingerprint: "fp_test",
      tether_version: "1.2.3",
      request_nonce: nonce,
    }),
  });
}

async function verifyResponse(response, publicKey) {
  assert.equal(response.status, 200);
  const value = await response.json();
  const { signature, ...payload } = value;
  const valid = await crypto.subtle.verify(
    "Ed25519",
    publicKey,
    decodeB64Url(signature),
    new TextEncoder().encode(canonical(payload)),
  );
  assert.equal(valid, true);
  assert.equal(signature.includes("="), false);
  return value;
}

test("active heartbeat returns exact signed nonce-bound attestation", async () => {
  const { env, publicKey, calls } = await signingEnv();
  const value = await verifyResponse(await worker.fetch(heartbeatRequest(), env), publicKey);
  assert.deepEqual(Object.keys(value).sort(), [
    "domain", "issued_at", "key_id", "license_id", "request_nonce",
    "signature", "status", "v", "valid_until",
  ]);
  assert.equal(value.domain, "tether.license.heartbeat");
  assert.equal(value.request_nonce, "MDEyMzQ1Njc4OWFiY2RlZg");
  assert.equal(value.status, "active");
  assert.ok(value.valid_until <= value.issued_at + 86400);
  assert.equal(calls.some((call) => call.sql.includes("INSERT INTO heartbeats")), true);
});

test("revoked and expired statuses are explicit and signed", async () => {
  const revokedEnv = await signingEnv({ revoked: { revoked_at: "now", reason: "test" } });
  const revoked = await verifyResponse(
    await worker.fetch(heartbeatRequest(), revokedEnv.env), revokedEnv.publicKey,
  );
  assert.equal(revoked.status, "revoked");

  const expiredEnv = await signingEnv({ expiresAt: new Date(Date.now() - 1000).toISOString() });
  const expired = await verifyResponse(
    await worker.fetch(heartbeatRequest(), expiredEnv.env), expiredEnv.publicKey,
  );
  assert.equal(expired.status, "expired");
});

test("malformed or padded nonces are rejected before database access", async () => {
  for (const nonce of ["short", "MDEyMzQ1Njc4OWFiY2RlZg==", "!!!!!!!!!!!!!!!!!!!!!!"]) {
    const { env, calls } = await signingEnv();
    const response = await worker.fetch(heartbeatRequest(nonce), env);
    assert.equal(response.status, 400);
    assert.equal(calls.length, 0);
  }
});

test("activation signs the exact hardware-bound v2 license", async () => {
  const { env, publicKey } = await signingEnv();
  const hardware = { gpu_uuid: "GPU-test", gpu_name: "A10G", cpu_count: 8 };
  const response = await worker.fetch(new Request(
    "https://worker.test/v1/activation/REFLEX-AAAA-BBBB-CCCC",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ hardware_binding: hardware }),
    },
  ), env);
  assert.equal(response.status, 200);
  const { license } = await response.json();
  const { signature, key_id, ...payload } = license;
  assert.equal(key_id, "key_test");
  assert.deepEqual(payload.hardware_binding, hardware);
  assert.equal(await crypto.subtle.verify(
    "Ed25519", publicKey, Buffer.from(signature, "base64"),
    new TextEncoder().encode(canonical(payload)),
  ), true);
});

test("configured signing key id must cryptographically match PRIVATE_KEY", async () => {
  const unrelated = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
  const unrelatedRaw = await crypto.subtle.exportKey("raw", unrelated.publicKey);
  const { env, calls } = await signingEnv({ publicKeyOverride: b64(unrelatedRaw) });
  const response = await worker.fetch(heartbeatRequest(), env);
  assert.equal(response.status, 500);
  assert.equal(calls.some((call) => call.sql.includes("INSERT INTO heartbeats")), false);
  assert.match((await response.json()).message, /does not match PRIVATE_KEY/);
});

test("authenticated signer check proves the configured private/public binding", async () => {
  const { env } = await signingEnv({ keyId: "key_existing" });
  env.ADMIN_TOKEN = "admin-test";
  const response = await worker.fetch(new Request("https://worker.test/admin/signer", {
    headers: { authorization: "Bearer admin-test" },
  }), env);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { verified: true, key_id: "key_existing" });
});

test("old and new configured signers both emit correctly labeled signatures", async () => {
  for (const keyId of ["key_old", "key_new"]) {
    const { env, publicKey } = await signingEnv({ keyId });
    const value = await verifyResponse(await worker.fetch(heartbeatRequest(), env), publicKey);
    assert.equal(value.key_id, keyId);
  }
});

test("retired configured signer fails closed", async () => {
  const { env } = await signingEnv({ keyId: "key_old", retired: true });
  const response = await worker.fetch(heartbeatRequest(), env);
  assert.equal(response.status, 500);
  assert.match((await response.json()).message, /missing or retired/);
});

test("fixed non-ASCII license vector matches Worker canonical bytes and signature", async () => {
  const vector = JSON.parse(readFileSync(
    new URL("../../../tests/fixtures/license_non_ascii_vector.json", import.meta.url),
    "utf8",
  ));
  const canonicalBytes = new TextEncoder().encode(canonical(vector.payload));
  assert.deepEqual(Buffer.from(canonicalBytes), Buffer.from(vector.canonical_utf8_b64, "base64"));
  const publicKey = await crypto.subtle.importKey(
    "raw", Buffer.from(vector.public_key_b64, "base64"),
    { name: "Ed25519" }, false, ["verify"],
  );
  assert.equal(await crypto.subtle.verify(
    "Ed25519", publicKey, Buffer.from(vector.signature_b64, "base64"), canonicalBytes,
  ), true);
});
