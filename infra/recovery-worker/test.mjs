import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, mkdtemp, writeFile, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
// Native Workers primitives; Node substitutes only the stream/hash implementations.
crypto.DigestStream = class extends WritableStream {
  constructor() {
    const hash = createHash("sha256");
    let resolve;
    const digest = new Promise((r) => {
      resolve = r;
    });
    super({
      write(chunk) {
        hash.update(chunk);
      },
      close() {
        resolve(hash.digest());
      },
    });
    this.digest = digest;
  }
};
globalThis.FixedLengthStream = class extends TransformStream {
  constructor(size) {
    let seen = 0;
    super({
      transform(chunk, controller) {
        seen += chunk.byteLength;
        if (seen > size) throw Error("size exceeded");
        controller.enqueue(chunk);
      },
      flush() {
        assert.equal(seen, size);
      },
    });
  }
};
const source = await readFile(new URL("./worker.js", import.meta.url), "utf8");
const { default: worker } = await import(
  "data:text/javascript;base64," + Buffer.from(source).toString("base64")
);
const token = "test-recovery-token-with-enough-entropy";
const tables = [
  "contributors",
  "uploads",
  "daily_uploads",
  "revoke_requests",
  "contributor_keys",
  "contributor_nonces",
];
const request = (path, method = "GET") =>
  new Request("https://example.invalid" + path, {
    method,
    headers: { authorization: "Bearer " + token },
  });
function bucket() {
  const entries = new Map();
  let serial = 0;
  return {
    entries,
    async put(key, body, options = {}) {
      const previous = entries.get(key);
      if (
        (options.onlyIf?.etagDoesNotMatch === "*" && previous) ||
        (options.onlyIf?.etagMatches &&
          options.onlyIf.etagMatches !== previous?.etag)
      )
        return null;
      const bytes =
        body instanceof ReadableStream
          ? new Uint8Array(await new Response(body).arrayBuffer())
          : new TextEncoder().encode(body);
      const item = { bytes, size: bytes.length, etag: String(++serial) };
      entries.set(key, item);
      return item;
    },
    async get(key) {
      const item = entries.get(key);
      if (!item) return null;
      return {
        ...item,
        body: new Blob([item.bytes]).stream(),
        arrayBuffer: async () => item.bytes,
        json: async () => JSON.parse(new TextDecoder().decode(item.bytes)),
      };
    },
  };
}
function environment(count = 0, size = 0) {
  const items = Array.from({ length: count }, (_, i) => ({
    key: "data/" + i,
    version: "version-" + i,
    etag: "etag-" + i,
    size,
  }));
  return {
    RECOVERY_TOKEN: token,
    ACCSYNC_BACKUP_TRIGGER_TOKEN: "separate-token",
    MAINTENANCE_TRIGGER_TOKEN: "maintenance-token",
    BACKUPS: bucket(),
    SOURCE: {
      async list() {
        return { objects: items, truncated: false };
      },
      async get(key, options) {
        const item = items.find((v) => v.key === key);
        assert.equal(options.onlyIf.etagMatches, item.etag);
        let remaining = item.size;
        return {
          ...item,
          body: new ReadableStream({
            pull(controller) {
              if (!remaining) return controller.close();
              const length = Math.min(remaining, 65536);
              remaining -= length;
              controller.enqueue(new Uint8Array(length).fill(42));
            },
          }),
          arrayBuffer() {
            throw Error("source bodies must stream");
          },
        };
      },
    },
    DB: {
      prepare() {
        return {
          async all() {
            return {
              results: tables.map((name) => ({
                type: "table",
                name,
                sql: `CREATE TABLE ${name}(id INTEGER PRIMARY KEY)`,
              })),
            };
          },
        };
      },
      async batch() {
        return tables.map(() => ({ results: [] }));
      },
    },
    ACCSYNC_BACKUP: {
      async fetch(_url, init) {
        assert.equal(init.headers.authorization, "Bearer separate-token");
        return Response.json({ complete: true, key: "accsync-snapshot" });
      },
    },
    CONTRIBUTIONS: {
      async fetch(url, init) {
        assert.equal(new URL(url).pathname, "/internal/maintenance");
        assert.equal(init.headers.authorization, "Bearer maintenance-token");
        return Response.json({ complete: true, revocationsExecuted: false });
      },
    },
  };
}
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
test("streamed 45-object 10.5 MiB backup downloads, restores and rejects corruption", async () => {
  const env = environment(45, 245760),
    response = await worker.fetch(request("/backup", "POST"), env);
  assert.equal(response.status, 200);
  const receipt = await response.json();
  assert.equal(receipt.objectBytes, 11059200);
  const manifest = await (
    await worker.fetch(request("/snapshot?key=" + receipt.key), env)
  ).json();
  assert.equal(manifest.version, 2);
  assert.equal(manifest.objects.length, 45);
  assert.ok(manifest.objects.every((o) => !o.bodyBase64 && o.version));
  const dir = await mkdtemp(join(tmpdir(), "tether-recovery-"));
  try {
    await writeFile(join(dir, "snapshot.json"), JSON.stringify(manifest));
    await mkdir(join(dir, "objects"));
    for (let i = 0; i < manifest.objects.length; i++) {
      const r = await worker.fetch(
        request("/object?key=" + receipt.key + "&index=" + i),
        env,
      );
      assert.equal(r.status, 200);
      await writeFile(
        join(dir, "objects", String(i).padStart(6, "0")),
        new Uint8Array(await r.arrayBuffer()),
      );
    }
    const script = new URL("./restore.py", import.meta.url).pathname;
    const result = JSON.parse(
      execFileSync(
        "python3",
        [
          script,
          join(dir, "snapshot.json"),
          receipt.sha256,
          join(dir, "restored"),
        ],
        { encoding: "utf8" },
      ),
    );
    assert.equal(result.objects, 45);
    assert.equal(result.tables, 6);
    await writeFile(join(dir, "objects/000000"), "corrupt");
    assert.throws(() =>
      execFileSync(
        "python3",
        [script, join(dir, "snapshot.json"), receipt.sha256, join(dir, "bad")],
        { stdio: "pipe" },
      ),
    );
    for (const suffix of ["&index=../status", "&index=999", "&index=00"])
      assert.notEqual(
        (
          await worker.fetch(
            request("/object?key=" + receipt.key + suffix),
            env,
          )
        ).status,
        200,
      );
    assert.equal(
      (
        await worker.fetch(
          request("/object?key=status/latest.json&index=0"),
          env,
        )
      ).status,
      404,
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("changing source version prevents manifest publication", async () => {
  const env = environment(1, 16);
  let lists = 0;
  const list = env.SOURCE.list;
  env.SOURCE.list = async () => {
    const result = await list();
    return {
      ...result,
      objects: result.objects.map((o) => ({
        ...o,
        version: ++lists > 1 ? "changed" : o.version,
      })),
    };
  };
  assert.equal(
    (await worker.fetch(request("/backup", "POST"), env)).status,
    500,
  );
  assert.ok(
    ![...env.BACKUPS.entries.keys()].some((k) => k.endsWith("snapshot.json")),
  );
  const status = await (await worker.fetch(request("/status"), env)).json();
  assert.equal(status.jobs.tether.latest.success, false);
  assert.equal(status.jobs.tether.fresh, false);
});
test("scheduled failures recorded while both other jobs still run and await completion", async () => {
  const env = environment();
  env.SOURCE.list = async () => {
    throw Error("source_unavailable");
  };
  await assert.rejects(worker.scheduled({}, env), /one_or_more/);
  const status = await (await worker.fetch(request("/status"), env)).json();
  assert.equal(status.jobs.tether.lastFailure.origin, "scheduled");
  assert.equal(status.jobs.accsync.lastSuccess.origin, "scheduled");
  assert.equal(status.jobs.maintenance.lastSuccess.origin, "scheduled");
  assert.equal(status.jobs.accsync.fresh, true);
});
test("freshness tracks last successful backup even when a later job fails", async () => {
  const env = environment();
  assert.equal(
    (await worker.fetch(request("/backup", "POST"), env)).status,
    200,
  );
  env.SOURCE.list = async () => {
    throw Error("source_unavailable");
  };
  assert.equal(
    (await worker.fetch(request("/backup", "POST"), env)).status,
    500,
  );
  let status = await (await worker.fetch(request("/status"), env)).json();
  assert.equal(status.jobs.tether.latest.success, false);
  assert.equal(status.jobs.tether.fresh, true);
  const state = JSON.parse(
    new TextDecoder().decode(
      env.BACKUPS.entries.get("status/latest.json").bytes,
    ),
  );
  state.jobs.tether.lastSuccess.finishedAt = "2020-01-01T00:00:00Z";
  await env.BACKUPS.put("status/latest.json", JSON.stringify(state));
  status = await (await worker.fetch(request("/status"), env)).json();
  assert.equal(status.jobs.tether.fresh, false);
});
test("v1 embedded-object restores stay compatible", async () => {
  const env = environment();
  const v1 = {
    version: 1,
    complete: true,
    capturedAt: new Date().toISOString(),
    schema: (await env.DB.prepare().all()).results.slice(0, 4),
    tables: Object.fromEntries(tables.slice(0, 4).map((t) => [t, []])),
    objects: [
      {
        key: "legacy",
        size: 5,
        bodyBase64: Buffer.from("hello").toString("base64"),
        sha256: createHash("sha256").update("hello").digest("hex"),
      },
    ],
  };
  const body = JSON.stringify(v1),
    dir = await mkdtemp(join(tmpdir(), "tether-legacy-"));
  try {
    await writeFile(join(dir, "snapshot.json"), body);
    const result = JSON.parse(
      execFileSync(
        "python3",
        [
          new URL("./restore.py", import.meta.url).pathname,
          join(dir, "snapshot.json"),
          createHash("sha256").update(body).digest("hex"),
          join(dir, "restore"),
        ],
        { encoding: "utf8" },
      ),
    );
    assert.equal(result.objects, 1);
    assert.equal(result.tables, 4);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("slow backup storage backpressures source without queuing full media", async () => {
  const env = environment(1, 8 * 1024 * 1024);
  let pulls = 0,
    release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  env.SOURCE.get = async () => ({
    key: "data/0",
    version: "version-0",
    etag: "etag-0",
    size: 8 * 1024 * 1024,
    body: new ReadableStream({
      pull(controller) {
        pulls++;
        if (pulls > 128) return controller.close();
        controller.enqueue(new Uint8Array(65536));
      },
    }),
  });
  const put = env.BACKUPS.put;
  env.BACKUPS.put = async (key, body, options) => {
    if (key.includes("/objects/")) await gate;
    return put(key, body, options);
  };
  const capture = worker.fetch(request("/backup", "POST"), env);
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.ok(pulls < 10, `queued ${pulls} source chunks while storage paused`);
  release();
  assert.equal((await capture).status, 200);
});
