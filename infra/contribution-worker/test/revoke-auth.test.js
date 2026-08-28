import assert from "node:assert/strict";
import test from "node:test";

import worker from "../worker.js";

const ADMIN_TOKEN = "test-admin-token-with-enough-entropy";

function request(path, { method = "GET", token, body } = {}) {
  const headers = {};
  if (token !== undefined) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["content-type"] = "application/json";
  return new Request(`https://worker.test${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function revokeRow(overrides = {}) {
  return {
    request_id: "rev_test",
    contributor_id: "contributor_test",
    requested_at: "2099-01-01T00:00:00.000Z",
    scope: "all",
    status: "pending",
    tombstone_at: null,
    r2_purge_started_at: null,
    r2_purge_completed_at: null,
    r2_objects_purged: 0,
    derived_rebuild_completed_at: "2099-01-01T00:00:00.000Z",
    buyer_notification_completed_at: "2099-01-01T00:00:00.000Z",
    completed_at: null,
    ...overrides,
  };
}

function dbMock({ first = null, all = { results: [] } } = {}) {
  const calls = [];
  return {
    calls,
    prepare(sql) {
      const call = { sql, bindings: [], operation: null };
      calls.push(call);
      return {
        bind(...bindings) {
          call.bindings = bindings;
          return this;
        },
        async first() {
          call.operation = "first";
          return typeof first === "function" ? first(call) : first;
        },
        async all() {
          call.operation = "all";
          return typeof all === "function" ? all(call) : all;
        },
        async run() {
          call.operation = "run";
          return { success: true };
        },
      };
    },
  };
}

test("unauthenticated revoke is rejected before any database mutation", async () => {
  const DB = dbMock();
  const response = await worker.fetch(request("/v1/revoke/cascade", {
    method: "POST",
    body: { contributor_id: "victim", scope: "all" },
  }), { DB, ADMIN_TOKEN });

  assert.equal(response.status, 401);
  assert.deepEqual(await response.json(), { error: "unauthorized" });
  assert.equal(DB.calls.length, 0);
});

test("every HTTP cascade mutation rejects a missing admin bearer", async () => {
  const cases = [
    ["/v1/revoke/cascade", { contributor_id: "victim", scope: "all" }],
    ["/admin/manual-purge", { contributor_id: "victim" }],
    ["/admin/cascade-execute/rev_test", undefined],
  ];
  for (const [path, body] of cases) {
    const DB = dbMock();
    const response = await worker.fetch(request(path, {
      method: "POST",
      body,
    }), { DB, ADMIN_TOKEN });
    assert.equal(response.status, 401, path);
    assert.equal(DB.calls.length, 0, path);
  }
});

test("incorrect bearer tokens of different lengths are rejected", async () => {
  for (const token of ["x", `${ADMIN_TOKEN}-wrong`, ""]) {
    const DB = dbMock();
    const response = await worker.fetch(request("/v1/revoke/cascade", {
      method: "POST",
      token,
      body: { contributor_id: "victim", scope: "all" },
    }), { DB, ADMIN_TOKEN });
    assert.equal(response.status, 401);
    assert.equal(DB.calls.length, 0);
  }
});

test("correct bearer token authorizes an idempotent revoke enqueue", async () => {
  const DB = dbMock();
  const response = await worker.fetch(request("/v1/revoke/cascade", {
    method: "POST",
    token: ADMIN_TOKEN,
    body: { contributor_id: "contributor_test", scope: "all" },
  }), { DB, ADMIN_TOKEN });

  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.contributor_id, "contributor_test");
  assert.match(payload.request_id, /^rev_[0-9a-f]{32}$/);
  assert.equal(DB.calls.filter((call) => call.operation === "run").length, 2);
});

test("cascade status GET is read-only even when a stage is due", async () => {
  const DB = dbMock({
    first: revokeRow({ requested_at: "2000-01-01T00:00:00.000Z" }),
  });
  const response = await worker.fetch(
    request("/v1/revoke/cascade-status/rev_test"),
    { DB },
  );

  assert.equal(response.status, 200);
  assert.equal(DB.calls.length, 1);
  assert.equal(DB.calls[0].operation, "first");
  assert.match(DB.calls[0].sql, /^SELECT \* FROM revoke_requests/);
});

test("scheduled event uses the private progression path", async () => {
  const DB = dbMock({ all: { results: [] } });
  const work = [];
  worker.scheduled({}, { DB }, { waitUntil: (promise) => work.push(promise) });

  assert.equal(work.length, 1);
  await Promise.all(work);
  assert.equal(DB.calls.length, 1);
  assert.equal(DB.calls[0].operation, "all");
  assert.match(DB.calls[0].sql, /status != 'completed'/);
});
