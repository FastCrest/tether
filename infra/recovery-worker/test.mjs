import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
const source = await readFile(new URL("./worker.js", import.meta.url), "utf8");
const { default: worker } = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);
const token = "test-recovery-token-with-enough-entropy";
test("authentication rejects requests before accessing contribution data", async () => {
  for (const env of [{}, { RECOVERY_TOKEN: token }])
    assert.equal(
      (
        await worker.fetch(
          new Request("https://example.invalid/backup", { method: "POST" }),
          env,
        )
      ).status,
      401,
    );
});
test("changing source prevents snapshot publication", async () => {
  let lists = 0,
    writes = 0;
  const env = {
    RECOVERY_TOKEN: token,
    SOURCE: {
      async list() {
        return {
          objects:
            ++lists === 1 ? [] : [{ key: "changed", etag: "new", size: 1 }],
          truncated: false,
        };
      },
    },
    DB: {
      prepare() {
        return {
          async all() {
            return { results: [] };
          },
        };
      },
      async batch() {
        return Array.from({ length: 6 }, () => ({ results: [] }));
      },
    },
    BACKUPS: {
      async put() {
        writes++;
      },
    },
  };
  const response = await worker.fetch(
    new Request("https://example.invalid/backup", {
      method: "POST",
      headers: { authorization: "Bearer " + token },
    }),
    env,
  );
  assert.equal(response.status, 500);
  assert.equal(writes, 0);
});
test("coordinator attempts AccSync even when the Tether capture fails", async () => {
  let invoked = 0;
  const env = {
    RECOVERY_TOKEN: token,
    ACCSYNC_BACKUP_TRIGGER_TOKEN: "separate-token",
    SOURCE: {
      async list() {
        throw Error("unavailable");
      },
    },
    ACCSYNC_BACKUP: {
      async fetch(request, init) {
        invoked++;
        assert.equal(init.headers.authorization, "Bearer separate-token");
        return Response.json({ complete: true });
      },
    },
  };
  const response = await worker.fetch(
    new Request("https://example.invalid/run-backups", {
      method: "POST",
      headers: { authorization: "Bearer " + token },
    }),
    env,
  );
  assert.equal(response.status, 500);
  assert.equal(invoked, 1);
});
