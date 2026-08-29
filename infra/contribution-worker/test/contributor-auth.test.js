import assert from "node:assert/strict";
import { createHash, createPrivateKey, sign as ed25519Sign } from "node:crypto";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import test from "node:test";

import {
  buildRequestEnvelope,
  canonicalizeJson,
  deriveContributorIdentifiers,
} from "../contributor-auth.js";
import { ContributorStorageFence } from "../contributor-storage-fence.js";
import worker from "../worker.js";

const vectors = JSON.parse(readFileSync(
  new URL("./fixtures/contributor-auth-v1.json", import.meta.url),
  "utf8",
)).vectors;
let nonceCounter = 1;

function decodeBase64url(value) {
  return Buffer.from(value, "base64url");
}

function privateKey(vector) {
  const seed = Buffer.from(vector.private_key_seed_hex, "hex");
  return createPrivateKey({
    key: Buffer.concat([
      Buffer.from("302e020100300506032b657004220420", "hex"),
      seed,
    ]),
    format: "der",
    type: "pkcs8",
  });
}

function nextNonce() {
  const bytes = Buffer.alloc(16);
  bytes.writeUInt32BE(nonceCounter++, 12);
  return bytes.toString("base64url");
}

function signedRequest(path, {
  vector = vectors[0],
  contributorId = vector.contributor_id,
  keyId = vector.key_id,
  method = "GET",
  body,
  nonce = nextNonce(),
  timestamp = Math.floor(Date.now() / 1000),
  publicKey,
  headerOverrides = {},
} = {}) {
  const url = `https://worker.test${path}`;
  if (path === "/v1/uploads/sign" && body !== undefined) {
    try {
      const value = JSON.parse(body);
      if (value && typeof value === "object" && !value.reservation_key) {
        value.reservation_key = createHash("sha256")
          .update(`${nonce}\n${body}`).digest("base64url");
        body = canonicalizeJson(value);
      }
    } catch {
      // Invalid JSON tests must reach the Worker unchanged.
    }
  }
  const bodyBytes = Buffer.from(body ?? "", "utf8");
  const bodySha256 = createHash("sha256").update(bodyBytes).digest("hex");
  const requestForEnvelope = new Request(url, { method });
  const envelope = buildRequestEnvelope({
    request: requestForEnvelope,
    contributorId,
    keyId,
    timestamp,
    nonce,
    bodySha256,
  });
  const signature = ed25519Sign(
    null,
    Buffer.from(canonicalizeJson(envelope)),
    privateKey(vector),
  ).toString("base64url");
  const headers = {
    "X-Tether-Auth-Version": "1",
    "X-Tether-Contributor-Id": contributorId,
    "X-Tether-Key-Id": keyId,
    "X-Tether-Timestamp": String(timestamp),
    "X-Tether-Nonce": nonce,
    "X-Tether-Content-SHA256": bodySha256,
    "X-Tether-Signature": signature,
    ...headerOverrides,
  };
  if (publicKey) headers["X-Tether-Public-Key"] = publicKey;
  if (body !== undefined) headers["content-type"] = "application/json";
  return new Request(url, {
    method,
    headers,
    body: body === undefined ? undefined : body,
  });
}

class D1Statement {
  constructor(database, sql, bindings = []) {
    this.database = database;
    this.sql = sql;
    this.bindings = bindings;
  }

  bind(...bindings) {
    return new D1Statement(this.database, this.sql, bindings);
  }

  async first() {
    return this.database.sqlite.prepare(this.sql).get(...this.bindings) ?? null;
  }

  async all() {
    return { results: this.database.sqlite.prepare(this.sql).all(...this.bindings) };
  }

  async run() {
    const result = this.database.sqlite.prepare(this.sql).run(...this.bindings);
    return { success: true, meta: { changes: Number(result.changes) } };
  }
}

class D1TestDatabase {
  constructor() {
    this.sqlite = new DatabaseSync(":memory:");
    this.sqlite.exec(readFileSync(new URL("../schema.sql", import.meta.url), "utf8"));
  }

  prepare(sql) {
    return new D1Statement(this, sql);
  }

  async batch(statements) {
    this.sqlite.exec("BEGIN IMMEDIATE");
    try {
      const results = [];
      for (const statement of statements) results.push(await statement.run());
      this.sqlite.exec("COMMIT");
      return results;
    } catch (error) {
      this.sqlite.exec("ROLLBACK");
      throw error;
    }
  }
}

class R2TestBucket {
  constructor() {
    this.objects = new Map();
    this.beforePut = null;
    this.afterPut = null;
    this.afterHead = null;
  }

  async put(key, body, options = {}) {
    if (this.beforePut) await this.beforePut(key);
    const bytes = new Uint8Array(body.slice(0));
    this.objects.set(key, { bytes, options });
    if (this.afterPut) await this.afterPut(key);
  }

  async head(key) {
    const object = this.objects.get(key);
    if (this.afterHead) await this.afterHead(key, object);
    if (!object) return null;
    return {
      size: object.bytes.byteLength,
      customMetadata: object.options.customMetadata || {},
    };
  }

  async delete(key) {
    this.objects.delete(key);
  }

  async list({ prefix = "", limit = 1000 } = {}) {
    const objects = Array.from(this.objects.keys())
      .filter((key) => key.startsWith(prefix))
      .slice(0, limit)
      .map((key) => ({ key }));
    return { objects, truncated: false };
  }
}

class DurableObjectTestStorage {
  constructor() {
    this.values = new Map();
  }

  async get(key) {
    return this.values.get(key);
  }

  async put(key, value) {
    this.values.set(key, value);
  }
}

class ContributorStorageTestNamespace {
  constructor(bucket) {
    this.bucket = bucket;
    this.instances = new Map();
    this.beforePayloadDispatch = null;
  }

  getByName(name) {
    if (!this.instances.has(name)) {
      this.instances.set(name, new ContributorStorageFence(
        { storage: new DurableObjectTestStorage() },
        { CURATE_BUCKET: this.bucket },
      ));
    }
    const instance = this.instances.get(name);
    return {
      fetch: async (request) => {
        if (new URL(request.url).pathname === "/payload" && this.beforePayloadDispatch) {
          await this.beforePayloadDispatch(request);
        }
        return await instance.fetch(request);
      },
    };
  }
}

function environment() {
  const bucket = new R2TestBucket();
  return {
    DB: new D1TestDatabase(),
    CURATE_BUCKET: bucket,
    CONTRIBUTOR_STORAGE: new ContributorStorageTestNamespace(bucket),
    UPLOAD_CAPABILITY_SECRET: "test-only-capability-secret-32-bytes",
  };
}

async function register(env, vector = vectors[0]) {
  return worker.fetch(signedRequest("/v1/contributors/register", {
    vector,
    method: "POST",
    publicKey: vector.public_key,
  }), env);
}

test("fixed Ed25519 vectors are RFC 8785 stable in Worker JavaScript", async () => {
  for (const vector of vectors) {
    assert.equal(canonicalizeJson(vector.envelope), vector.canonical_envelope_utf8);
    const derived = await deriveContributorIdentifiers(vector.public_key);
    assert.equal(derived.contributorId, vector.contributor_id);
    assert.equal(derived.keyId, vector.key_id);
    const key = await crypto.subtle.importKey(
      "raw",
      decodeBase64url(vector.public_key),
      { name: "Ed25519" },
      false,
      ["verify"],
    );
    assert.equal(await crypto.subtle.verify(
      "Ed25519",
      key,
      decodeBase64url(vector.signature),
      Buffer.from(vector.canonical_envelope_utf8),
    ), true);

    for (const mutate of [
      (envelope) => ({ ...envelope, method: "DELETE" }),
      (envelope) => ({ ...envelope, path: envelope.path.replace("%2F", "/") }),
      (envelope) => ({ ...envelope, body_sha256: "0".repeat(64) }),
      (envelope) => ({ ...envelope, query: [...envelope.query].reverse() }),
    ]) {
      const mutated = mutate(vector.envelope);
      if (canonicalizeJson(mutated) === vector.canonical_envelope_utf8) continue;
      assert.equal(await crypto.subtle.verify(
        "Ed25519",
        key,
        decodeBase64url(vector.signature),
        Buffer.from(canonicalizeJson(mutated)),
      ), false);
    }
  }
  assert.deepEqual(vectors[1].envelope.query, [
    ["key", "value"],
    ["key", "value2"],
    ["z", ""],
  ]);
});

test("existing-database migration expires unauthenticated legacy reservations", () => {
  const sqlite = new DatabaseSync(":memory:");
  sqlite.exec(`
    CREATE TABLE contributors (
      contributor_id TEXT PRIMARY KEY,
      tier TEXT NOT NULL,
      first_seen_at TEXT NOT NULL
    );
    CREATE TABLE uploads (
      upload_id TEXT PRIMARY KEY,
      contributor_id TEXT NOT NULL,
      r2_key TEXT NOT NULL,
      byte_size INTEGER NOT NULL,
      status TEXT NOT NULL,
      signed_at TEXT NOT NULL
    );
    INSERT INTO contributors VALUES ('legacy', 'free', '2026-01-01');
    INSERT INTO uploads VALUES ('old', 'legacy', 'legacy/key', 1, 'pending', '2026-01-01');
  `);
  sqlite.exec(readFileSync(
    new URL("../migrations/2026-08-28-contributor-auth-v1.sql", import.meta.url),
    "utf8",
  ));
  assert.equal(sqlite.prepare(
    "SELECT status FROM uploads WHERE upload_id = 'old'",
  ).get().status, "expired");
  const columns = new Set(sqlite.prepare("PRAGMA table_info(uploads)").all().map((row) => row.name));
  for (const name of [
    "content_sha256",
    "manifest_sha256",
    "capability_sha256",
    "attempt_id",
    "completion_id",
    "reservation_key",
    "request_sha256",
  ]) assert.equal(columns.has(name), true, name);
});

test("registration proves possession, is idempotent, and rejects replay", async () => {
  const env = environment();
  const noProof = await worker.fetch(signedRequest("/v1/contributors/register", {
    method: "POST",
    publicKey: vectors[0].public_key,
    headerOverrides: { "X-Tether-Signature": vectors[1].signature },
  }), env);
  assert.equal(noProof.status, 401);
  assert.equal(env.DB.sqlite.prepare("SELECT COUNT(*) AS count FROM contributors").get().count, 0);

  const single = signedRequest("/v1/contributors/register", {
    method: "POST",
    publicKey: vectors[0].public_key,
  });
  const duplicateHeaders = new Headers(single.headers);
  duplicateHeaders.append(
    "X-Tether-Timestamp",
    single.headers.get("X-Tether-Timestamp"),
  );
  const duplicate = await worker.fetch(new Request(single.url, {
    method: "POST",
    headers: duplicateHeaders,
  }), env);
  assert.equal(duplicate.status, 401);

  const response = await register(env);
  assert.equal(response.status, 201);
  assert.deepEqual(await response.json(), {
    contributor_id: vectors[0].contributor_id,
    key_id: vectors[0].key_id,
    tier: "free",
  });

  assert.equal((await register(env)).status, 200);
  const replayNonce = nextNonce();
  assert.equal((await worker.fetch(signedRequest("/v1/contributors/register", {
    method: "POST",
    publicKey: vectors[0].public_key,
    nonce: replayNonce,
  }), env)).status, 200);
  const replay = await worker.fetch(signedRequest("/v1/contributors/register", {
    method: "POST",
    publicKey: vectors[0].public_key,
    nonce: replayNonce,
  }), env);
  assert.equal(replay.status, 409);
  assert.equal((await replay.json()).error, "nonce_replay");
});

test("body, path, timestamp, padding, and header substitution fail closed", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);

  const stale = await worker.fetch(signedRequest(
    `/v1/contributors/${vectors[0].contributor_id}/stats`,
    { timestamp: 1 },
  ), env);
  assert.equal(stale.status, 401);

  const padded = await worker.fetch(signedRequest(
    `/v1/contributors/${vectors[0].contributor_id}/stats`,
    { headerOverrides: { "X-Tether-Nonce": `${nextNonce()}=` } },
  ), env);
  assert.equal(padded.status, 400);

  const mismatch = await worker.fetch(signedRequest(
    `/v1/contributors/${vectors[0].contributor_id}/stats`,
    { headerOverrides: { "X-Tether-Key-Id": vectors[1].key_id } },
  ), env);
  assert.equal(mismatch.status, 401);

  const original = signedRequest("/v1/uploads/complete", {
    method: "POST",
    body: canonicalizeJson({ upload_id: "upl_original" }),
  });
  const changedBody = new Request(original.url, {
    method: original.method,
    headers: original.headers,
    body: canonicalizeJson({ upload_id: "upl_changed" }),
  });
  assert.equal((await worker.fetch(changedBody, env)).status, 401);
});

test("rotation preserves contributor identity and rejects the inactive key", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const body = canonicalizeJson({ public_key: vectors[1].public_key });
  const rotated = await worker.fetch(signedRequest("/v1/contributors/rotate", {
    method: "POST",
    body,
  }), env);
  assert.equal(rotated.status, 200);
  const payload = await rotated.json();
  assert.equal(payload.contributor_id, vectors[0].contributor_id);
  assert.equal(payload.key_id, vectors[1].key_id);

  const path = `/v1/contributors/${vectors[0].contributor_id}/stats`;
  assert.equal((await worker.fetch(signedRequest(path), env)).status, 401);
  assert.equal((await worker.fetch(signedRequest(path, {
    vector: vectors[1],
    contributorId: vectors[0].contributor_id,
    keyId: vectors[1].key_id,
  }), env)).status, 200);
});

test("revoke status is read-only and visible only to its owner or admin", async () => {
  const env = environment();
  assert.equal((await register(env, vectors[0])).status, 201);
  assert.equal((await register(env, vectors[1])).status, 201);
  env.DB.sqlite.prepare(
    `INSERT INTO revoke_requests
       (request_id, contributor_id, requested_at, scope, status)
     VALUES ('rev_owner', ?, ?, 'all', 'pending')`,
  ).run(vectors[0].contributor_id, new Date().toISOString());

  const path = "/v1/revoke/cascade-status/rev_owner";
  const owner = await worker.fetch(signedRequest(path), env);
  assert.equal(owner.status, 200);
  assert.equal((await owner.json()).request_id, "rev_owner");
  const foreign = await worker.fetch(signedRequest(path, { vector: vectors[1] }), env);
  assert.equal(foreign.status, 403);
  const unauthenticated = await worker.fetch(new Request(`https://worker.test${path}`), env);
  assert.equal(unauthenticated.status, 401);
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM revoke_requests WHERE request_id = 'rev_owner'",
  ).get().status, "pending");
});

test("stats, reservation, PUT capability, and completion are principal-bound", async () => {
  const env = environment();
  assert.equal((await register(env, vectors[0])).status, 201);
  assert.equal((await register(env, vectors[1])).status, 201);

  const content = Buffer.from("{\"episode\":1}\n", "utf8");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "test-anonymizer-1",
    content_sha256: createHash("sha256").update(content).digest("hex"),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 1, face: 0, name: 2 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "test-scanner-1",
  };
  const signBody = canonicalizeJson({
    byte_size: content.byteLength,
    contributor_id: vectors[1].contributor_id,
    file_name: "session.jsonl",
    manifest,
    tier: "enterprise",
  });
  const signed = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: signBody,
    timestamp: now,
  }), env);
  assert.equal(signed.status, 200, JSON.stringify(await signed.clone().json()));
  const reservation = await signed.json();
  assert.match(reservation.r2_key, new RegExp(`^free-contributors/${vectors[0].contributor_id}/`));
  assert.match(reservation.upload_capability, /^[A-Za-z0-9_-]{43}$/);

  const wrongCapability = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${Buffer.alloc(32, 7).toString("base64url")}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  }), env);
  assert.equal(wrongCapability.status, 401);
  assert.equal(env.CURATE_BUCKET.objects.size, 0);

  const makePut = () => new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  });
  const concurrentPuts = await Promise.all([
    worker.fetch(makePut(), env),
    worker.fetch(makePut(), env),
  ]);
  assert.deepEqual(concurrentPuts.map((response) => response.status).sort(), [200, 409]);
  const successfulPut = concurrentPuts.find((response) => response.status === 200);
  assert.match((await successfulPut.json()).r2_key, /\/attempts\/att_[0-9a-f]{32}$/);
  assert.equal(env.CURATE_BUCKET.objects.size, 1);

  const completeBody = canonicalizeJson({ upload_id: reservation.upload_id });
  const crossPrincipal = await worker.fetch(signedRequest("/v1/uploads/complete", {
    vector: vectors[1],
    method: "POST",
    body: completeBody,
  }), env);
  assert.equal(crossPrincipal.status, 403);

  const completed = await worker.fetch(signedRequest("/v1/uploads/complete", {
    method: "POST",
    body: completeBody,
  }), env);
  assert.equal(completed.status, 200);
  assert.equal((await completed.json()).status, "completed");

  const duplicateComplete = await worker.fetch(signedRequest("/v1/uploads/complete", {
    method: "POST",
    body: completeBody,
  }), env);
  assert.equal(duplicateComplete.status, 200);
  assert.equal((await duplicateComplete.json()).idempotent, true);

  const ownStats = await worker.fetch(signedRequest(
    `/v1/contributors/${vectors[0].contributor_id}/stats`,
  ), env);
  assert.equal(ownStats.status, 200);
  const stats = await ownStats.json();
  assert.equal(stats.total_uploads, 1);
  assert.equal(stats.total_bytes, content.byteLength);
  assert.equal(stats.total_episodes, 0);

  env.ADMIN_TOKEN = "test-admin-token";
  const adminStats = await worker.fetch(new Request(
    `https://worker.test/v1/contributors/${vectors[0].contributor_id}/stats`,
    { headers: { Authorization: "Bearer test-admin-token" } },
  ), env);
  assert.equal(adminStats.status, 200);

  const foreignStats = await worker.fetch(signedRequest(
    `/v1/contributors/${vectors[1].contributor_id}/stats`,
  ), env);
  assert.equal(foreignStats.status, 403);
});

test("manifest, size, and digest validation reject before authoritative completion", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const now = Math.floor(Date.now() / 1000);
  const invalid = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: 1, file_name: "x.jsonl" }),
    timestamp: now,
  }), env);
  assert.equal(invalid.status, 400);

  const oversizedManifest = {
    anonymizer_version: "a",
    content_sha256: "0".repeat(64),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const oversized = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({
      byte_size: 104857601,
      file_name: "x.jsonl",
      manifest: oversizedManifest,
    }),
    timestamp: now,
  }), env);
  assert.equal(oversized.status, 413, JSON.stringify(await oversized.clone().json()));

  const insert = env.DB.sqlite.prepare(
    `INSERT INTO uploads
       (upload_id, contributor_id, r2_key, byte_size, media_type, content_sha256,
        manifest_sha256, capability_sha256, expires_at, status, signed_at,
        signed_at_epoch)
     VALUES (?, ?, ?, 1, 'application/jsonl', ?, ?, ?, ?, 'expired', ?, ?)`,
  );
  for (let index = 0; index < 60; index += 1) {
    insert.run(
      `upl_rate_${index}`,
      vectors[0].contributor_id,
      `free-contributors/${vectors[0].contributor_id}/rate-${index}`,
      "0".repeat(64),
      "1".repeat(64),
      "2".repeat(64),
      now + 600,
      new Date(now * 1000).toISOString(),
      now,
    );
  }
  const rateLimited = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({
      byte_size: 1,
      file_name: "rate.jsonl",
      manifest: oversizedManifest,
    }),
    timestamp: now,
  }), env);
  assert.equal(rateLimited.status, 429);
  assert.equal((await rateLimited.json()).error, "sign_rate_limit_exceeded");
});

test("parquet reservations preserve the allowlisted media type in R2", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("PAR1test", "utf8");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "test-anonymizer-1",
    content_sha256: createHash("sha256").update(content).digest("hex"),
    domain: "tether.anonymization.manifest",
    media_type: "application/x-parquet",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "test-scanner-1",
  };
  const signed = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({
      byte_size: content.byteLength,
      file_name: "episode.parquet",
      manifest,
    }),
    timestamp: now,
  }), env);
  assert.equal(signed.status, 200, JSON.stringify(await signed.clone().json()));
  const reservation = await signed.json();
  const uploaded = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  }), env);
  assert.equal(uploaded.status, 200, JSON.stringify(await uploaded.clone().json()));
  const row = env.DB.sqlite.prepare(
    "SELECT r2_key, media_type FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(row.media_type, "application/x-parquet");
  assert.equal(env.CURATE_BUCKET.objects.get(row.r2_key).options.httpMetadata.contentType,
    "application/x-parquet");
});

test("PUT requires exact declared length and rejects a signed digest mismatch", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("x", "utf8");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "a",
    content_sha256: "0".repeat(64),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signed = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: 1, file_name: "digest.jsonl", manifest }),
    timestamp: now,
  }), env);
  assert.equal(signed.status, 200);
  const reservation = await signed.json();

  const missingLength = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: { Authorization: `Upload ${reservation.upload_capability}` },
    body: content,
  }), env);
  assert.equal(missingLength.status, 411);

  const wrongDigest = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": "1",
    },
    body: content,
  }), env);
  assert.equal(wrongDigest.status, 400);
  assert.equal((await wrongDigest.json()).error, "content_digest_mismatch");
  assert.equal(env.CURATE_BUCKET.objects.size, 0);
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id).status, "rejected");
});

test("signed completion expires an unclaimed persisted reservation", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("persisted reservation");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "a",
    content_sha256: createHash("sha256").update(content).digest("hex"),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signed = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({
      byte_size: content.byteLength,
      file_name: "expired-persisted.jsonl",
      manifest,
    }),
    timestamp: now,
  }), env);
  assert.equal(signed.status, 200);
  const reservation = await signed.json();

  const realDateNow = Date.now;
  let completed;
  try {
    Date.now = () => realDateNow() + 11 * 60 * 1000;
    completed = await worker.fetch(signedRequest("/v1/uploads/complete", {
      method: "POST",
      body: canonicalizeJson({ upload_id: reservation.upload_id }),
    }), env);
  } finally {
    Date.now = realDateNow;
  }
  assert.equal(completed.status, 410);
  assert.equal((await completed.json()).error, "upload_reservation_expired");
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id).status, "expired");
  assert.equal(env.CURATE_BUCKET.objects.size, 0);
});

test("scheduled recovery finalizes only matching attempt-specific objects", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("{\"recover\":true}\n");
  const now = Math.floor(Date.now() / 1000);
  const digest = createHash("sha256").update(content).digest("hex");
  const manifest = {
    anonymizer_version: "a",
    content_sha256: digest,
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signed = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: content.byteLength, file_name: "recover.jsonl", manifest }),
    timestamp: now,
  }), env);
  const reservation = await signed.json();
  const row = env.DB.sqlite.prepare(
    "SELECT * FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  const attemptId = "att_00000000000000000000000000000001";
  const attemptKey = `${row.r2_key}/attempts/${attemptId}`;
  env.DB.sqlite.prepare(
    "UPDATE uploads SET status = 'uploading', attempt_id = ?, attempt_started_at = 1 WHERE upload_id = ?",
  ).run(attemptId, reservation.upload_id);
  await env.CURATE_BUCKET.put(attemptKey, content, {
    customMetadata: {
      attempt_id: attemptId,
      content_sha256: digest,
      manifest_sha256: row.manifest_sha256,
      upload_id: reservation.upload_id,
    },
  });

  const work = [];
  worker.scheduled({}, env, { waitUntil: (promise) => work.push(promise) });
  await Promise.all(work);
  const recovered = env.DB.sqlite.prepare(
    "SELECT status, r2_key FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(recovered.status, "uploaded");
  assert.equal(recovered.r2_key, attemptKey);
  assert.equal(env.CURATE_BUCKET.objects.size, 1);
});

test("original PUT cannot delete an attempt recovered and completed before its final CAS", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("{\"race\":true}\n");
  const now = Math.floor(Date.now() / 1000);
  const digest = createHash("sha256").update(content).digest("hex");
  const manifest = {
    anonymizer_version: "a",
    content_sha256: digest,
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signResponse = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: content.byteLength, file_name: "race.jsonl", manifest }),
    timestamp: now,
  }), env);
  const reservation = await signResponse.json();

  env.CURATE_BUCKET.afterPut = async () => {
    env.CURATE_BUCKET.afterPut = null;
    // Make the just-written attempt stale, then deterministically run recovery
    // before the originating request reaches its final uploaded CAS.
    env.DB.sqlite.prepare(
      "UPDATE uploads SET attempt_started_at = 1 WHERE upload_id = ?",
    ).run(reservation.upload_id);
    const work = [];
    worker.scheduled({}, env, { waitUntil: (promise) => work.push(promise) });
    await Promise.all(work);
    const completeBody = canonicalizeJson({ upload_id: reservation.upload_id });
    const completed = await worker.fetch(signedRequest("/v1/uploads/complete", {
      method: "POST",
      body: completeBody,
    }), env);
    assert.equal(completed.status, 200, JSON.stringify(await completed.clone().json()));
  };

  const response = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  }), env);
  assert.equal(response.status, 409);
  const row = env.DB.sqlite.prepare(
    "SELECT status, r2_key FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(row.status, "completed");
  assert.equal(env.CURATE_BUCKET.objects.has(row.r2_key), true);
  assert.equal(env.CURATE_BUCKET.objects.size, 1);
});

test("invalid recovery inspection cannot delete an owner-finalized attempt", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("{\"owner\":true}\n");
  const now = Math.floor(Date.now() / 1000);
  const digest = createHash("sha256").update(content).digest("hex");
  const manifest = {
    anonymizer_version: "a",
    content_sha256: digest,
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signResponse = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: content.byteLength, file_name: "owner-race.jsonl", manifest }),
    timestamp: now,
  }), env);
  const reservation = await signResponse.json();
  const row = env.DB.sqlite.prepare(
    "SELECT * FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  const attemptId = "att_00000000000000000000000000000002";
  const attemptKey = `${row.r2_key}/attempts/${attemptId}`;
  env.DB.sqlite.prepare(
    "UPDATE uploads SET status = 'uploading', attempt_id = ?, attempt_started_at = 1 WHERE upload_id = ?",
  ).run(attemptId, reservation.upload_id);
  await env.CURATE_BUCKET.put(attemptKey, Buffer.from("invalid"), {
    customMetadata: {
      attempt_id: attemptId,
      content_sha256: "0".repeat(64),
      manifest_sha256: row.manifest_sha256,
      upload_id: reservation.upload_id,
    },
  });

  env.CURATE_BUCKET.afterHead = async (key) => {
    if (key !== attemptKey) return;
    env.CURATE_BUCKET.afterHead = null;
    // Recovery has already observed the invalid object. The owner now
    // overwrites it with the valid body and finalizes before recovery can
    // claim/reset the inspected attempt.
    await env.CURATE_BUCKET.put(attemptKey, content, {
      customMetadata: {
        attempt_id: attemptId,
        content_sha256: digest,
        manifest_sha256: row.manifest_sha256,
        upload_id: reservation.upload_id,
      },
    });
    env.DB.sqlite.prepare(
      `UPDATE uploads SET status = 'uploaded', r2_key = ?, uploaded_at = ?
        WHERE upload_id = ? AND status = 'uploading' AND attempt_id = ?`,
    ).run(attemptKey, now, reservation.upload_id, attemptId);
  };

  const work = [];
  worker.scheduled({}, env, { waitUntil: (promise) => work.push(promise) });
  await Promise.all(work);
  const finalized = env.DB.sqlite.prepare(
    "SELECT status, r2_key FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(finalized.status, "uploaded");
  assert.equal(finalized.r2_key, attemptKey);
  assert.equal(env.CURATE_BUCKET.objects.has(attemptKey), true);
  assert.equal(env.CURATE_BUCKET.objects.size, 1);
});

test("terminal purge storage-fences a claimed PUT beyond the former stale lease", async () => {
  const env = environment();
  env.ADMIN_TOKEN = "test-admin-token";
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("{\"purge_race\":true}\n");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "a",
    content_sha256: createHash("sha256").update(content).digest("hex"),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signResponse = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({ byte_size: content.byteLength, file_name: "purge-race.jsonl", manifest }),
    timestamp: now,
  }), env);
  const reservation = await signResponse.json();

  let releasePut;
  let signalPutEntered;
  const putEntered = new Promise((resolve) => { signalPutEntered = resolve; });
  const putGate = new Promise((resolve) => { releasePut = resolve; });
  env.CONTRIBUTOR_STORAGE.beforePayloadDispatch = async () => {
    env.CONTRIBUTOR_STORAGE.beforePayloadDispatch = null;
    signalPutEntered();
    await putGate;
  };
  const putPromise = worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  }), env);
  await putEntered;

  const adminHeaders = {
    Authorization: "Bearer test-admin-token",
    "Content-Type": "application/json",
  };
  const revoke = await worker.fetch(new Request("https://worker.test/v1/revoke/cascade", {
    method: "POST",
    headers: adminHeaders,
    body: JSON.stringify({ contributor_id: vectors[0].contributor_id, scope: "all" }),
  }), env);
  assert.equal(revoke.status, 200);
  const requestId = (await revoke.json()).request_id;

  // Advance beyond the removed five-minute stale-marker shortcut while the
  // live request is still blocked before the serialized storage boundary.
  const realDateNow = Date.now;
  let terminalPurge;
  try {
    Date.now = () => realDateNow() + 10 * 60 * 1000;
    terminalPurge = await worker.fetch(new Request(
      `https://worker.test/admin/cascade-execute/${requestId}`,
      { method: "POST", headers: adminHeaders },
    ), env);
  } finally {
    Date.now = realDateNow;
  }
  assert.equal(terminalPurge.status, 200);
  assert.equal((await terminalPurge.json()).overall_status, "completed");
  const fenced = env.DB.sqlite.prepare(
    "SELECT status, attempt_id FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(fenced.status, "purged");
  assert.equal(fenced.attempt_id, null);
  assert.equal(env.CURATE_BUCKET.objects.size, 0);

  releasePut();
  const losingPut = await putPromise;
  assert.equal(losingPut.status, 409);
  assert.equal((await losingPut.json()).error, "contributor_purged_during_upload");
  const cleaned = env.DB.sqlite.prepare(
    "SELECT status, attempt_id FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id);
  assert.equal(cleaned.status, "purged");
  assert.equal(cleaned.attempt_id, null);
  assert.equal(env.CURATE_BUCKET.objects.size, 0);

  const work = [];
  worker.scheduled({}, env, { waitUntil: (promise) => work.push(promise) });
  await Promise.all(work);
  assert.equal(env.CURATE_BUCKET.objects.size, 0);
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id).status, "purged");
});

test("completion HEAD followed by purge reports purged after its lost CAS", async () => {
  const env = environment();
  env.ADMIN_TOKEN = "test-admin-token";
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("{\"completion_purge\":true}\n");
  const now = Math.floor(Date.now() / 1000);
  const manifest = {
    anonymizer_version: "a",
    content_sha256: createHash("sha256").update(content).digest("hex"),
    domain: "tether.anonymization.manifest",
    media_type: "application/jsonl",
    removed_fields: { email: 0, face: 0, name: 0 },
    scan_timestamp: now,
    schema_version: 1,
    scanner_version: "s",
  };
  const signResponse = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST",
    body: canonicalizeJson({
      byte_size: content.byteLength,
      file_name: "completion-purge.jsonl",
      manifest,
    }),
    timestamp: now,
  }), env);
  const reservation = await signResponse.json();
  const uploaded = await worker.fetch(new Request(reservation.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${reservation.upload_capability}`,
      "Content-Length": String(content.byteLength),
    },
    body: content,
  }), env);
  assert.equal(uploaded.status, 200);

  env.CURATE_BUCKET.afterHead = async () => {
    env.CURATE_BUCKET.afterHead = null;
    const adminHeaders = {
      Authorization: "Bearer test-admin-token",
      "Content-Type": "application/json",
    };
    const revoke = await worker.fetch(new Request("https://worker.test/v1/revoke/cascade", {
      method: "POST",
      headers: adminHeaders,
      body: JSON.stringify({ contributor_id: vectors[0].contributor_id, scope: "all" }),
    }), env);
    assert.equal(revoke.status, 200);
    const requestId = (await revoke.json()).request_id;
    const purged = await worker.fetch(new Request(
      `https://worker.test/admin/cascade-execute/${requestId}`,
      { method: "POST", headers: adminHeaders },
    ), env);
    assert.equal(purged.status, 200);
    assert.equal((await purged.json()).overall_status, "completed");
  };

  const completeBody = canonicalizeJson({ upload_id: reservation.upload_id });
  const completed = await worker.fetch(signedRequest("/v1/uploads/complete", {
    method: "POST",
    body: completeBody,
  }), env);
  assert.equal(completed.status, 410);
  assert.equal((await completed.json()).error, "upload_purged");
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM uploads WHERE upload_id = ?",
  ).get(reservation.upload_id).status, "purged");
  assert.equal(env.CURATE_BUCKET.objects.size, 0);
  const contributor = env.DB.sqlite.prepare(
    "SELECT total_bytes, total_uploads FROM contributors WHERE contributor_id = ?",
  ).get(vectors[0].contributor_id);
  assert.equal(contributor.total_bytes, 0);
  assert.equal(contributor.total_uploads, 0);
});

test("scheduled purge accepts arbitrary historical Pro IDs and one failure cannot starve later cascades", async () => {
  const env = environment();
  const requestedAt = "2000-01-01T00:00:00.000Z";
  const legacyId = "alice@bigco.com";
  const brokenId = "invalid/path";
  for (const contributorId of [brokenId, legacyId]) {
    env.DB.sqlite.prepare(
      `INSERT INTO contributors
         (contributor_id, tier, first_seen_at, last_active_at, revoked_at)
       VALUES (?, 'free', ?, ?, ?)`,
    ).run(contributorId, requestedAt, requestedAt, requestedAt);
    env.DB.sqlite.prepare(
      `INSERT INTO revoke_requests
         (request_id, contributor_id, requested_at, scope, status,
          derived_rebuild_completed_at, buyer_notification_completed_at)
       VALUES (?, ?, ?, 'all', 'pending', ?, ?)`,
    ).run(
      contributorId === brokenId ? "rev_broken" : "rev_legacy",
      contributorId,
      requestedAt,
      requestedAt,
      requestedAt,
    );
  }
  const legacyKey = `pro-contributors/${legacyId}/2025-01-01/legacy.jsonl`;
  await env.CURATE_BUCKET.put(legacyKey, Buffer.from("legacy"));

  const errors = [];
  const originalConsoleError = console.error;
  console.error = (message) => errors.push(String(message));
  try {
    const work = [];
    worker.scheduled({}, env, { waitUntil: (promise) => work.push(promise) });
    await Promise.all(work);
  } finally {
    console.error = originalConsoleError;
  }

  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM revoke_requests WHERE request_id = 'rev_broken'",
  ).get().status, "pending");
  assert.equal(env.DB.sqlite.prepare(
    "SELECT status FROM revoke_requests WHERE request_id = 'rev_legacy'",
  ).get().status, "completed");
  assert.equal(env.CURATE_BUCKET.objects.has(legacyKey), false);
  assert.equal(errors.some((line) => line.includes("rev_broken")), true);

  // Legacy identifiers are purge-only. Even an internal caller cannot use a
  // historical arbitrary-ID namespace to put payload bytes through the fence.
  const legacyStorage = env.CONTRIBUTOR_STORAGE.getByName(legacyId);
  const rejectedWrite = await legacyStorage.fetch(new Request(
    "https://storage.internal/payload",
    {
      method: "PUT",
      headers: {
        "X-Tether-Contributor-Id": legacyId,
        "X-Tether-R2-Key": legacyKey,
        "X-Tether-R2-Metadata": "{}",
      },
      body: "must-not-write",
    },
  ));
  assert.equal(rejectedWrite.status, 400);
  assert.equal(env.CURATE_BUCKET.objects.has(legacyKey), false);
  const rejectedReservation = await worker.fetch(signedRequest("/v1/uploads/sign", {
    contributorId: legacyId,
    method: "POST",
    body: canonicalizeJson({}),
  }), env);
  assert.equal(rejectedReservation.status, 400);
  for (const unsafeId of ["../alice", "alice\\bigco"]) {
    const unsafe = env.CONTRIBUTOR_STORAGE.getByName(unsafeId);
    const response = await unsafe.fetch(new Request("https://storage.internal/purge", {
      method: "POST", headers: { "X-Tether-Contributor-Id": unsafeId },
    }));
    assert.equal(response.status, 400);
  }
});

test("reservation key replay returns one usable reservation without double quota", async () => {
  const env = environment();
  assert.equal((await register(env)).status, 201);
  const content = Buffer.from("idempotent reservation");
  const reservationKey = Buffer.alloc(32, 9).toString("base64url");
  const body = canonicalizeJson({
    byte_size: content.length,
    file_name: "idempotent.jsonl",
    reservation_key: reservationKey,
    manifest: {
      anonymizer_version: "tether-anonymizer-v1",
      content_sha256: createHash("sha256").update(content).digest("hex"),
      domain: "tether.anonymization.manifest",
      media_type: "application/jsonl",
      removed_fields: { email: 0, face: 0, name: 0 },
      scan_timestamp: Math.floor(Date.now() / 1000),
      scanner_version: "tether-scanner-v1",
      schema_version: 1,
    },
  });
  const first = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST", body,
  }), env);
  const replay = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST", body,
  }), env);
  assert.equal(first.status, 200);
  assert.equal(replay.status, 200);
  const one = await first.json();
  const two = await replay.json();
  assert.equal(two.idempotent, true);
  assert.equal(two.upload_id, one.upload_id);
  assert.equal(two.upload_capability, one.upload_capability);
  assert.equal(env.DB.sqlite.prepare(
    "SELECT COUNT(*) AS count FROM uploads WHERE contributor_id = ?",
  ).get(vectors[0].contributor_id).count, 1);
  const put = await worker.fetch(new Request(two.put_url, {
    method: "PUT",
    headers: {
      Authorization: `Upload ${two.upload_capability}`,
      "Content-Length": String(content.length),
      "Content-Type": "application/jsonl",
    },
    body: content,
  }), env);
  assert.equal(put.status, 200);
  const completed = await worker.fetch(signedRequest("/v1/uploads/complete", {
    method: "POST", body: canonicalizeJson({ upload_id: one.upload_id }),
  }), env);
  assert.equal(completed.status, 200);
  env.DAILY_UPLOADS_LIMIT = 1;
  const saturatedReplay = await worker.fetch(signedRequest("/v1/uploads/sign", {
    method: "POST", body, timestamp: Math.floor(Date.now() / 1000) + 240,
  }), env);
  assert.equal(saturatedReplay.status, 200);
  assert.equal((await saturatedReplay.json()).upload_id, one.upload_id);
  assert.equal(env.DB.sqlite.prepare("SELECT COUNT(*) AS count FROM uploads").get().count, 1);

  const raceEnv = environment();
  assert.equal((await register(raceEnv)).status, 201);
  const changed = { ...JSON.parse(body), file_name: "different.jsonl" };
  const [raceOne, raceTwo] = await Promise.all([
    worker.fetch(signedRequest("/v1/uploads/sign", { method: "POST", body }), raceEnv),
    worker.fetch(signedRequest("/v1/uploads/sign", {
      method: "POST", body: canonicalizeJson(changed),
    }), raceEnv),
  ]);
  assert.deepEqual([raceOne.status, raceTwo.status].sort(), [200, 409]);
  assert.equal(raceEnv.DB.sqlite.prepare("SELECT COUNT(*) AS count FROM uploads").get().count, 1);
});
