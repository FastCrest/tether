import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdtemp, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execFileSync } from "node:child_process";
import worker, { RecoveryCapture } from "./worker.js";
import { Capture, CAPACITY, TABLES } from "./capture.js";
crypto.DigestStream = class extends WritableStream {
  constructor() {
    const hash = createHash("sha256");
    let resolve, reject;
    const digest = new Promise((a, b) => {
      resolve = a;
      reject = b;
    });
    super({
      write(chunk) {
        hash.update(chunk);
      },
      close() {
        resolve(hash.digest());
      },
      abort(error) {
        reject(error);
      },
    });
    this.digest = digest;
  }
};
globalThis.FixedLengthStream = class extends TransformStream {
  constructor(size) {
    let seen = 0;
    super({
      transform(chunk, output) {
        seen += chunk.length;
        if (seen > size) throw Error("size exceeded");
        output.enqueue(chunk);
      },
      flush() {
        assert.equal(seen, size);
      },
    });
  }
};
const token = "private-test-token";
const sha = (body) => createHash("sha256").update(body).digest("hex");
function bytesStream(size, value = 42) {
  let remaining = size;
  return new ReadableStream({
    pull(output) {
      if (!remaining) return output.close();
      const n = Math.min(65536, remaining);
      remaining -= n;
      output.enqueue(new Uint8Array(n).fill(value));
    },
  });
}
function storage() {
  const values = new Map();
  return {
    values,
    alarm: null,
    async get(key) {
      if (Array.isArray(key))
        return new Map(
          key
            .filter((k) => values.has(k))
            .map((k) => [k, structuredClone(values.get(k))]),
        );
      return structuredClone(values.get(key));
    },
    async put(key, value) {
      assert.ok(
        Buffer.byteLength(JSON.stringify(value)) < 128 * 1024,
        "DO value capacity",
      );
      values.set(key, structuredClone(value));
    },
    async setAlarm(at) {
      this.alarm = at;
    },
    async deleteAlarm() {
      this.alarm = null;
    },
  };
}
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
      let item;
      if (body instanceof ReadableStream) {
        let size = 0;
        const hash = createHash("sha256");
        for await (const chunk of body) {
          assert.ok(chunk.every((b) => b === 42));
          size += chunk.length;
          hash.update(chunk);
        }
        item = { size, pattern: 42, sha256: hash.digest("hex") };
      } else {
        const bytes = new TextEncoder().encode(body);
        item = { size: bytes.length, bytes, sha256: sha(bytes) };
      }
      item.etag = String(++serial);
      entries.set(key, item);
      return structuredClone(item);
    },
    async get(key) {
      const item = entries.get(key);
      if (!item) return null;
      return {
        ...item,
        body: item.bytes
          ? new Blob([item.bytes]).stream()
          : bytesStream(item.size, item.pattern),
        async arrayBuffer() {
          if (!item.bytes) throw Error("media must stream");
          return item.bytes;
        },
        async json() {
          return JSON.parse(new TextDecoder().decode(item.bytes));
        },
      };
    },
  };
}
function environment(count = 0, size = 0, rowCount = 0) {
  const objects = Array.from({ length: count }, (_, i) => ({
    key: "source/" + String(i).padStart(6, "0"),
    etag: "etag-" + i,
    version: "version-" + i,
    size,
  }));
  const dbRows = Object.fromEntries(
    TABLES.map((t) => [
      t,
      Array.from({ length: t === "contributors" ? rowCount : 0 }, (_, i) => ({
        __backup_rowid: i + 1,
        id: i + 1,
      })),
    ]),
  );
  const env = {
    RECOVERY_TOKEN: token,
    ACCSYNC_BACKUP_TRIGGER_TOKEN: "accsync-test-token",
    MAINTENANCE_TRIGGER_TOKEN: "maintenance-test-token",
    BACKUPS: bucket(),
    objects,
    dbRows,
    SOURCE: {
      async list({ limit, cursor } = {}) {
        const start = Number(cursor || 0);
        return {
          objects: structuredClone(objects.slice(start, start + limit)),
          truncated: start + limit < objects.length,
          cursor: String(start + limit),
        };
      },
      async get(key, options) {
        const item = objects.find((o) => o.key === key);
        if (!item || item.etag !== options.onlyIf.etagMatches) return null;
        const range = options.range || { offset: 0, length: item.size };
        return { ...item, range, body: bytesStream(range.length) };
      },
    },
    DB: {
      prepare(query) {
        return {
          bind(...args) {
            this.args = args;
            return this;
          },
          async all() {
            if (query.includes("sqlite_schema"))
              return {
                results: TABLES.map((name) => ({
                  type: "table",
                  name,
                  tbl_name: name,
                  sql: `CREATE TABLE ${name}(id INTEGER PRIMARY KEY)`,
                })),
              };
            const name = query.match(/FROM "([^"]+)"/)[1],
              args = this.args;
            const after = args.length === 1 ? null : args[0],
              limit = args.at(-1);
            return {
              results: structuredClone(
                dbRows[name]
                  .filter((r) => after === null || r.__backup_rowid > after)
                  .slice(0, limit),
              ),
            };
          },
        };
      },
    },
    CONTRIBUTIONS: {
      async fetch() {
        return Response.json({
          status: "ok",
          purges_executed: 0,
          uploads: { recovered: 0 },
          nonces_pruned: 0,
        });
      },
    },
    ACCSYNC_BACKUP: {
      calls: 0,
      async fetch() {
        this.calls++;
        return Response.json({ complete: true, key: "accsync-snapshot" });
      },
    },
  };
  const state = storage(),
    instance = new RecoveryCapture({ storage: state }, env);
  env.CAPTURE = {
    idFromName() {
      return "single";
    },
    get() {
      return {
        fetch(url, init) {
          return instance.fetch(new Request(url, init));
        },
      };
    },
  };
  return { env, state, instance };
}
const request = (path, method = "GET") =>
  new Request("https://backup.example" + path, {
    method,
    headers: { authorization: "Bearer " + token },
  });
async function drain(instance, state, maximum = 15000) {
  for (let i = 0; i < maximum; i++) {
    const job = await state.get("job");
    if (["complete", "failed"].includes(job.phase)) return job;
    await instance.alarm();
  }
  throw Error("test did not settle");
}
async function started(fixture, coordinator = false) {
  const result = await worker.fetch(
    request(coordinator ? "/run-backups" : "/backup", "POST"),
    fixture.env,
  );
  assert.equal(result.status, 202);
  return drain(fixture.instance, fixture.state);
}
test("authentication guards all capture/data/status routes", async () => {
  for (const path of [
    "/backup",
    "/advance",
    "/status",
    "/snapshot?key=bad",
    "/chunk?key=bad",
  ])
    assert.equal(
      (await worker.fetch(new Request("https://backup.example" + path), {}))
        .status,
      401,
    );
});
test("durable paginated capture exceeds 200 objects, 512 MiB total and 10k rows without buffering media", async () => {
  const fixture = environment(211, 3 * 1024 * 1024, 10051),
    job = await started(fixture);
  assert.equal(job.phase, "complete");
  assert.equal(job.receipt.objects, 211);
  assert.equal(job.receipt.objectBytes, 663748608);
  assert.equal(job.receipt.tables.contributors, 10051);
  assert.ok(job.rowPages > 40);
  assert.ok(job.inventoryPages > 4);
  assert.ok(
    [...fixture.env.BACKUPS.entries.keys()].filter((k) =>
      k.includes("/indexes/"),
    ).length > 1,
  );
  assert.equal(fixture.state.alarm, null);
});
test("one object beyond 128 MiB is copied in bounded range chunks", async () => {
  const fixture = environment(1, 129 * 1024 * 1024),
    job = await started(fixture);
  assert.equal(job.phase, "complete");
  const chunks = [...fixture.env.BACKUPS.entries.entries()].filter(([k]) =>
    k.includes("/media/"),
  );
  assert.equal(chunks.length, 9);
  assert.ok(chunks.every(([, v]) => v.size <= CAPACITY.chunkBytes));
});
test("lost checkpoint after immutable chunk write resumes idempotently", async () => {
  const fixture = environment(1, 100),
    engine = new Capture(fixture.state, fixture.env);
  await engine.start("manual", false);
  await fixture.instance.alarm();
  const before = await fixture.state.get("job");
  assert.equal(before.phase, "copy");
  let lost = true;
  const put = fixture.state.put.bind(fixture.state);
  fixture.state.put = async (key, value) => {
    if (lost && key === "job" && value.completedObjects === 1) {
      lost = false;
      throw Error("simulated_eviction");
    }
    return put(key, value);
  };
  await assert.rejects(engine.step(), /simulated_eviction/);
  fixture.state.put = put;
  const job = await drain(fixture.instance, fixture.state);
  assert.equal(job.phase, "complete");
  assert.equal(job.receipt.objects, 1);
  assert.equal(
    [...fixture.env.BACKUPS.entries.keys()].filter((k) => k.includes("/media/"))
      .length,
    1,
  );
});
test("new source version fails closed even when ETag/content is unchanged", async () => {
  const fixture = environment(1, 10);
  await worker.fetch(request("/backup", "POST"), fixture.env);
  await fixture.instance.alarm();
  fixture.env.objects[0].version = "changed";
  const job = await drain(fixture.instance, fixture.state);
  assert.equal(job.phase, "failed");
  assert.equal(job.lastError, "source_version_changed");
  assert.ok(
    ![...fixture.env.BACKUPS.entries.keys()].some((k) =>
      k.endsWith("snapshot.json"),
    ),
  );
});
test("table mutation during verification prevents publication", async () => {
  const fixture = environment(0, 0, 300);
  await worker.fetch(request("/backup", "POST"), fixture.env);
  while ((await fixture.state.get("job")).phase !== "verify_rows")
    await fixture.instance.alarm();
  fixture.env.dbRows.contributors[0].id = 900;
  const job = await drain(fixture.instance, fixture.state);
  assert.equal(job.phase, "failed");
  assert.equal(job.lastError, "source_tables_changed_during_backup");
});
test("new inventory object during copy prevents publication", async () => {
  const fixture = environment(1, 10);
  await worker.fetch(request("/backup", "POST"), fixture.env);
  await fixture.instance.alarm();
  fixture.env.objects.push({
    key: "source/new",
    size: 0,
    version: "new",
    etag: "new",
  });
  const job = await drain(fixture.instance, fixture.state);
  assert.equal(job.phase, "failed");
  assert.equal(job.lastError, "source_inventory_changed_during_backup");
});
test("coordinator settles maintenance before capture and independently runs AccSync on maintenance failure", async () => {
  const fixture = environment(0, 0, 1);
  fixture.env.CONTRIBUTIONS.fetch = async () =>
    new Response("unavailable", { status: 503 });
  const job = await started(fixture, true);
  assert.equal(job.phase, "complete");
  assert.equal(fixture.env.ACCSYNC_BACKUP.calls, 1);
  const status = await (
    await worker.fetch(request("/status"), fixture.env)
  ).json();
  assert.equal(status.jobs.maintenance.lastFailure.error, "service_http_503");
  assert.equal(status.jobs.tether.lastSuccess.receipt.version, 3);
  assert.equal(status.jobs.accsync.lastSuccess.receipt.complete, true);
});
test("alarm survives transient R2 failure; progress does not become a successful backup", async () => {
  const fixture = environment(1, 10);
  await worker.fetch(request("/backup", "POST"), fixture.env);
  const list = fixture.env.SOURCE.list;
  let first = true;
  fixture.env.SOURCE.list = async (...args) => {
    if (first) {
      first = false;
      throw Error("temporary_unavailable");
    }
    return list(...args);
  };
  await fixture.instance.alarm();
  let status = await (
    await worker.fetch(request("/status"), fixture.env)
  ).json();
  assert.equal(status.jobs.tether.latest.success, null);
  assert.equal(status.jobs.tether.fresh, false);
  assert.ok(fixture.state.alarm);
  assert.equal(
    (await drain(fixture.instance, fixture.state)).phase,
    "complete",
  );
});
test("scheduled completion proof survives later manual capture", async () => {
  const fixture = environment();
  await worker.scheduled({}, fixture.env);
  await drain(fixture.instance, fixture.state);
  let status = await (
      await worker.fetch(request("/status"), fixture.env)
    ).json(),
    proof = status.jobs.tether.lastScheduledSuccess;
  assert.equal(proof.origin, "scheduled");
  await started(fixture);
  status = await (await worker.fetch(request("/status"), fixture.env)).json();
  assert.deepEqual(status.jobs.tether.lastScheduledSuccess, proof);
  assert.equal(status.jobs.tether.lastSuccess.origin, "manual");
});
test("v3 scoped downloads and local restore verify shards and chunks", async () => {
  const fixture = environment(3, 65536, 501),
    job = await started(fixture);
  const dir = await mkdtemp(join(tmpdir(), "tether-v3-"));
  try {
    for (const name of ["indexes", "shards", "media"])
      await mkdir(join(dir, name));
    const root = await (
      await worker.fetch(
        request("/snapshot?key=" + job.receipt.key),
        fixture.env,
      )
    ).json();
    await writeFile(join(dir, "snapshot.json"), JSON.stringify(root));
    const routes = {};
    async function fetch(path, file) {
      const response = await worker.fetch(request(path), fixture.env);
      assert.equal(response.status, 200);
      const bytes = Buffer.from(await response.arrayBuffer());
      routes[path] = bytes.toString("base64");
      if (file) await writeFile(join(dir, file), bytes);
      return bytes;
    }
    await fetch("/snapshot?key=" + encodeURIComponent(job.receipt.key));
    for (let i = 0; i < root.indexes.length; i++) {
      const query =
        "key=" + encodeURIComponent(job.receipt.key) + "&index=" + i;
      const listing = JSON.parse(
        await fetch(
          "/index?" + query,
          "indexes/" + String(i).padStart(8, "0") + ".json",
        ),
      );
      for (let n = 0; n < listing.entries.length; n++) {
        const item = listing.entries[n],
          suffix = query + "&entry=" + n;
        const shard = JSON.parse(
          await fetch(
            "/shard?" + suffix,
            "shards/" +
              String(i).padStart(8, "0") +
              "-" +
              String(n).padStart(8, "0") +
              ".json",
          ),
        );
        if (item.kind === "object")
          for (let c = 0; c < shard.chunks.length; c++)
            await fetch(
              "/chunk?" + suffix + "&chunk=" + c,
              "media/" +
                String(i).padStart(8, "0") +
                "-" +
                String(n).padStart(8, "0") +
                "-" +
                String(c).padStart(8, "0"),
            );
      }
    }
    const script = new URL("./restore.py", import.meta.url).pathname;
    const receipt = JSON.parse(
      execFileSync(
        "python3",
        [
          script,
          join(dir, "snapshot.json"),
          job.receipt.sha256,
          join(dir, "restore"),
        ],
        { encoding: "utf8" },
      ),
    );
    assert.equal(receipt.rows, 501);
    assert.equal(receipt.objects, 3);
    await writeFile(join(dir, "routes.json"), JSON.stringify(routes));
    const downloader = new URL("./download.py", import.meta.url).pathname;
    const downloadResult = execFileSync(
      "python3",
      [
        "-c",
        `
import sys, pathlib, importlib.util, json, io, base64, os
sys.path.insert(0, ${JSON.stringify(new URL("./", import.meta.url).pathname)})
spec=importlib.util.spec_from_file_location("downloader",${JSON.stringify(downloader)})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
routes=json.loads(pathlib.Path(${JSON.stringify(join(dir, "routes.json"))}).read_text())
class Opener:
 def open(self,request,timeout):
  assert request.get_header("Authorization")=="Bearer private-test-token"
  assert request.get_header("User-agent")=="Tether recovery operator"
  return io.BytesIO(base64.b64decode(routes[request.full_url.removeprefix("https://backup.example")]))
module.urllib.request.build_opener=lambda *args:Opener()
os.environ["RECOVERY_TOKEN"]="private-test-token"
print(json.dumps(module.download("https://backup.example",${JSON.stringify(job.receipt.key)},${JSON.stringify(job.receipt.sha256)},${JSON.stringify(join(dir, "download"))})))
`,
      ],
      { encoding: "utf8" },
    );
    assert.equal(JSON.parse(downloadResult).verified, true);
    const restoreDownloaded = JSON.parse(
      execFileSync(
        "python3",
        [
          script,
          join(dir, "download/snapshot.json"),
          job.receipt.sha256,
          join(dir, "download-restored"),
        ],
        { encoding: "utf8" },
      ),
    );
    assert.equal(restoreDownloaded.objects, 3);
    const mediaFile = (await import("node:fs/promises")).readdir;
    const files = await mediaFile(join(dir, "media"));
    await writeFile(join(dir, "media", files[0]), "corrupt");
    assert.throws(() =>
      execFileSync(
        "python3",
        [
          script,
          join(dir, "snapshot.json"),
          job.receipt.sha256,
          join(dir, "corrupt"),
        ],
        { stdio: "pipe" },
      ),
    );
    for (const path of [
      "/chunk?key=" + job.receipt.key + "&index=0&entry=0&chunk=999",
      "/index?key=status/latest.json&index=0",
      "/shard?key=" + job.receipt.key + "&index=../status",
    ])
      assert.notEqual(
        (await worker.fetch(request(path), fixture.env)).status,
        200,
      );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
test("v1/v2 snapshots remain downloadable and restorable", async () => {
  for (const version of [1, 2]) {
    const fixture = environment(),
      dir = await mkdtemp(join(tmpdir(), "tether-old-"));
    try {
      const key = "snapshots/20261025T000000Z-old/snapshot.json",
        body = Buffer.from("hello"),
        entry = {
          key: "old",
          size: 5,
          sha256: sha(body),
          ...(version === 1
            ? { bodyBase64: body.toString("base64") }
            : { backupKey: "snapshots/20261025T000000Z-old/objects/000000" }),
        };
      const schema = (await fixture.env.DB.prepare("sqlite_schema").all())
        .results;
      const manifest = {
          version,
          complete: true,
          capturedAt: new Date().toISOString(),
          schema,
          tables: Object.fromEntries(TABLES.map((t) => [t, []])),
          objects: [entry],
        },
        raw = JSON.stringify(manifest);
      await fixture.env.BACKUPS.put(key, raw);
      await writeFile(join(dir, "snapshot.json"), raw);
      if (version === 2) {
        await mkdir(join(dir, "objects"));
        await writeFile(join(dir, "objects/000000"), body);
        fixture.env.BACKUPS.entries.set(entry.backupKey, {
          size: 5,
          bytes: body,
          etag: "legacy",
        });
        assert.equal(
          (
            await worker.fetch(
              request("/object?key=" + key + "&index=0"),
              fixture.env,
            )
          ).status,
          200,
        );
      }
      const receipt = JSON.parse(
        execFileSync(
          "python3",
          [
            new URL("./restore.py", import.meta.url).pathname,
            join(dir, "snapshot.json"),
            sha(raw),
            join(dir, "restore"),
          ],
          { encoding: "utf8" },
        ),
      );
      assert.equal(receipt.objects, 1);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }
});

test("root publication survives a lost final checkpoint without changing its pinned hash", async () => {
  const fixture = environment(1, 10),
    engine = new Capture(fixture.state, fixture.env);
  await engine.start("manual", false);
  while ((await fixture.state.get("job")).phase !== "publish")
    await fixture.instance.alarm();
  let lost = true;
  const put = fixture.state.put.bind(fixture.state);
  fixture.state.put = async (key, value) => {
    if (lost && key === "job" && value.phase === "complete") {
      lost = false;
      throw Error("lost_final_checkpoint");
    }
    return put(key, value);
  };
  await assert.rejects(engine.step(), /lost_final_checkpoint/);
  const key = [...fixture.env.BACKUPS.entries.keys()].find((k) =>
    k.endsWith("snapshot.json"),
  );
  const before = fixture.env.BACKUPS.entries.get(key).sha256;
  fixture.state.put = put;
  const final = await drain(fixture.instance, fixture.state);
  assert.equal(final.phase, "complete");
  assert.equal(final.receipt.sha256, before);
});
test("status-write failure after completion retries publication status without refreshing backup age", async () => {
  const fixture = environment();
  await worker.fetch(request("/backup", "POST"), fixture.env);
  while ((await fixture.state.get("job")).phase !== "publish")
    await fixture.instance.alarm();
  let fail = true;
  const put = fixture.env.BACKUPS.put;
  fixture.env.BACKUPS.put = async (key, body, options) => {
    if (fail && key === "status/latest.json") {
      fail = false;
      throw Error("temporary_status_failure");
    }
    return put(key, body, options);
  };
  await assert.rejects(fixture.instance.alarm(), /temporary_status_failure/);
  assert.equal((await fixture.state.get("job")).phase, "complete");
  const completed = await fixture.state.get("job");
  await new Promise((resolve) => setTimeout(resolve, 15));
  await fixture.instance.alarm();
  const first = await (
    await worker.fetch(request("/status"), fixture.env)
  ).json();
  assert.equal(first.jobs.tether.lastSuccess.finishedAt, completed.finishedAt);
  await new Promise((resolve) => setTimeout(resolve, 5));
  await worker.fetch(request("/advance", "POST"), fixture.env);
  const second = await (
    await worker.fetch(request("/status"), fixture.env)
  ).json();
  assert.equal(
    first.jobs.tether.lastSuccess.finishedAt,
    second.jobs.tether.lastSuccess.finishedAt,
  );
});

test("an existing durable checkpoint always rearms its missing alarm", async () => {
  const fixture = environment();
  const engine = new Capture(fixture.state, fixture.env);
  const original = await engine.start("manual", false);
  fixture.state.alarm = null;
  const resumed = await engine.start("manual", false);
  assert.equal(resumed.id, original.id);
  assert.ok(fixture.state.alarm);
  assert.equal(
    (await drain(fixture.instance, fixture.state)).phase,
    "complete",
  );
});
