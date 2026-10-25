import { Capture, CAPACITY, hash, readJSON } from "./capture.js";
const encoder = new TextEncoder(),
  STATUS_KEY = "status/latest.json";
const headers = { "cache-control": "no-store" };
async function authorized(request, env) {
  if (!env.RECOVERY_TOKEN) return false;
  const [a, b] = await Promise.all([
    hash(encoder.encode(request.headers.get("authorization") || "")),
    hash(encoder.encode("Bearer " + env.RECOVERY_TOKEN)),
  ]);
  let difference = 0;
  for (let i = 0; i < a.length; i++)
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return difference === 0;
}
async function readStatus(env) {
  const o = await env.BACKUPS.get(STATUS_KEY);
  return { object: o, state: o ? await o.json() : { version: 1, jobs: {} } };
}
async function record(env, name, result, progress = false) {
  for (let retry = 0; retry < 5; retry++) {
    const { object, state } = await readStatus(env),
      previous = state.jobs[name] || {};
    state.jobs[name] = {
      ...previous,
      latest: result,
      ...(progress
        ? { progress: result.receipt }
        : {
            progress: null,
            [result.success ? "lastSuccess" : "lastFailure"]: result,
          }),
      ...(result.origin === "scheduled"
        ? {
            lastScheduledAttempt: result,
            ...(!progress && result.success
              ? { lastScheduledSuccess: result }
              : {}),
          }
        : {}),
    };
    if (
      await env.BACKUPS.put(STATUS_KEY, JSON.stringify(state), {
        onlyIf: object
          ? { etagMatches: object.etag }
          : { etagDoesNotMatch: "*" },
        httpMetadata: { contentType: "application/json" },
      })
    )
      return;
  }
  throw Error("status_write_conflict");
}
const code = (error) =>
  /^[a-z0-9_]{1,100}$/.test(error.message) ? error.message : "operation_failed";
async function invoke(binding, token, path) {
  if (!binding || !token) throw Error("service_binding_or_token_missing");
  const response = await binding.fetch("https://backup.internal" + path, {
    method: "POST",
    headers: { authorization: "Bearer " + token },
    redirect: "manual",
    signal: AbortSignal.timeout(90000),
  });
  if (!response.ok) throw Error("service_http_" + response.status);
  const receipt = await response.json();
  if (
    path === "/internal/maintenance"
      ? receipt.status !== "ok" || receipt.purges_executed !== 0
      : receipt.complete !== true
  )
    throw Error("service_invalid_receipt");
  return receipt;
}
async function serviceJob(env, name, origin, operation) {
  const startedAt = new Date().toISOString();
  try {
    const receipt = await operation();
    await record(env, name, {
      success: true,
      origin,
      startedAt,
      finishedAt: new Date().toISOString(),
      receipt,
    });
    return receipt;
  } catch (error) {
    await record(env, name, {
      success: false,
      origin,
      startedAt,
      finishedAt: new Date().toISOString(),
      error: code(error),
    });
    throw error;
  }
}
const progress = (job) =>
  job
    ? {
        id: job.id,
        phase: job.phase,
        origin: job.origin,
        startedAt: job.startedAt,
        updatedAt: job.updatedAt,
        objects: job.objects,
        completedObjects: job.completedObjects || 0,
        objectBytes: job.objectBytes,
        copiedBytes: job.copiedBytes,
        tables: job.rows,
        rowPages: job.rowPages,
        verificationPage: job.verifyPage,
        attempts: job.attempts,
        lastError: job.lastError,
        receipt: job.receipt,
      }
    : null;
export class RecoveryCapture {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.engine = new Capture(ctx.storage, env);
    this.tail = Promise.resolve();
  }
  exclusive(operation) {
    const result = this.tail.then(operation, operation);
    this.tail = result.catch(() => {});
    return result;
  }
  async publishStatus(job) {
    const result = {
      origin: job.origin,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt || job.updatedAt || job.startedAt,
      success:
        job.phase === "complete" ? true : job.phase === "failed" ? false : null,
      inProgress: !["complete", "failed"].includes(job.phase),
      ...(job.phase === "failed"
        ? { error: job.lastError }
        : { receipt: job.phase === "complete" ? job.receipt : progress(job) }),
    };
    await record(this.env, "tether", result, result.inProgress);
  }
  async advance() {
    let job = await this.ctx.storage.get("job");
    if (!job) return job;
    if (["complete", "failed"].includes(job.phase)) {
      if (!job.statusRecorded) {
        await this.ctx.storage.setAlarm(Date.now() + 60000);
        await this.publishStatus(job);
        job.statusRecorded = true;
        await this.ctx.storage.put("job", job);
      }
      await this.ctx.storage.deleteAlarm();
      return job;
    }
    // Persist the next alarm before work, so eviction/timeouts after an immutable
    // write still retry the same checkpoint. The engine's keys are deterministic.
    await this.ctx.storage.setAlarm(Date.now() + 60000);
    try {
      job = await this.engine.step(async () => {
        await serviceJob(this.env, "maintenance", job.origin, () =>
          invoke(
            this.env.CONTRIBUTIONS,
            this.env.MAINTENANCE_TRIGGER_TOKEN,
            "/internal/maintenance",
          ),
        ).catch(() => {});
      });
    } catch (error) {
      job = await this.ctx.storage.get("job");
      job.attempts++;
      job.lastError = code(error);
      job.updatedAt = new Date().toISOString();
      const terminal =
        /source_|capacity|precision|deadline|checksum_mismatch|retry_mismatch/.test(
          job.lastError,
        ) || job.attempts >= 8;
      if (terminal) {
        job.phase = "failed";
        job.finishedAt = job.updatedAt;
      }
      await this.ctx.storage.put("job", job);
      if (!terminal)
        await this.ctx.storage.setAlarm(
          Date.now() + Math.min(300000, 1000 * 2 ** job.attempts),
        );
    }
    await this.publishStatus(job);
    if (["complete", "failed"].includes(job.phase)) {
      job.statusRecorded = true;
      await this.ctx.storage.put("job", job);
      await this.ctx.storage.deleteAlarm();
    }
    return job;
  }
  async fetch(request) {
    return this.exclusive(async () => {
      const url = new URL(request.url);
      if (url.pathname === "/start") {
        const job = await this.engine.start(
          url.searchParams.get("origin") === "scheduled"
            ? "scheduled"
            : "manual",
          url.searchParams.get("maintenance") === "1",
        );
        await this.publishStatus(job);
        return Response.json(progress(job));
      }
      if (url.pathname === "/advance")
        return Response.json(progress(await this.advance()));
      return Response.json(progress(await this.ctx.storage.get("job")));
    });
  }
  async alarm() {
    return this.exclusive(() => this.advance());
  }
}
function durable(env) {
  if (!env.CAPTURE) throw Error("capture_durable_binding_missing");
  return env.CAPTURE.get(env.CAPTURE.idFromName("contribution-backup-v3"));
}
async function coordinate(env, origin) {
  const settled = await Promise.allSettled([
    durable(env)
      .fetch("https://capture.internal/start?maintenance=1&origin=" + origin, {
        method: "POST",
      })
      .then((r) => {
        if (!r.ok) throw Error("capture_start_failed");
        return r.json();
      }),
    serviceJob(env, "accsync", origin, () =>
      invoke(env.ACCSYNC_BACKUP, env.ACCSYNC_BACKUP_TRIGGER_TOKEN, "/backup"),
    ),
  ]);
  const jobs = settled.map((r, i) => ({
    project: ["tether", "accsync"][i],
    success:
      r.status === "fulfilled"
        ? i === 0 && !r.value.receipt
          ? null
          : true
        : false,
    ...(r.status === "fulfilled"
      ? { receipt: r.value }
      : { error: code(r.reason) }),
  }));
  return {
    complete: jobs.every((j) => j.success === true),
    inProgress: jobs.some((j) => j.success === null),
    jobs,
  };
}
const validSnapshot = (key) =>
  /^snapshots\/[A-Za-z0-9TZ-]+\/snapshot\.json$/.test(key ?? "");
const index = (value) =>
  /^(0|[1-9][0-9]{0,5})$/.test(value ?? "") ? Number(value) : -1;
async function download(request, env, url) {
  const key = url.searchParams.get("key"),
    object = await env.BACKUPS.get(key);
  if (!object) return new Response("Missing", { status: 404, headers });
  if (url.pathname === "/snapshot")
    return new Response(object.body, {
      headers: { ...headers, "content-type": "application/json" },
    });
  const manifest = await object.json();
  if (!manifest.complete)
    return new Response("Incomplete", { status: 400, headers });
  const prefix = key.slice(0, -"snapshot.json".length);
  let descriptor;
  if (manifest.version === 2 && url.pathname === "/object") {
    const n = index(url.searchParams.get("index")),
      item = manifest.objects[n];
    if (item?.backupKey === prefix + "objects/" + String(n).padStart(6, "0"))
      descriptor = {
        key: item.backupKey,
        size: item.size,
        sha256: item.sha256,
      };
  } else if (manifest.version === 3) {
    const n = index(url.searchParams.get("index")),
      entry = index(url.searchParams.get("entry")),
      chunk = index(url.searchParams.get("chunk"));
    const page = manifest.indexes[n];
    if (
      page?.key !==
      prefix + "indexes/" + String(n).padStart(8, "0") + ".json"
    )
      return new Response("Missing", { status: 404, headers });
    if (url.pathname === "/index") descriptor = page;
    else {
      const listing = await readJSON(env, page),
        item = listing.entries[entry];
      if (
        !item ||
        !["table", "object"].includes(item.kind) ||
        !item.key.startsWith(
          prefix + (item.kind === "table" ? "tables/" : "objects/"),
        )
      )
        return new Response("Missing", { status: 404, headers });
      if (url.pathname === "/shard") descriptor = item;
      else if (url.pathname === "/chunk" && item.kind === "object") {
        const media = await readJSON(env, item),
          candidate = media.chunks[chunk];
        if (
          candidate?.key.startsWith(prefix + "media/") &&
          /^media\/[0-9]{8}\/[0-9]{8}$/.test(candidate.key.slice(prefix.length))
        )
          descriptor = candidate;
      }
    }
  }
  if (!descriptor) return new Response("Missing", { status: 404, headers });
  const read = await env.BACKUPS.get(descriptor.key);
  if (!read || read.size !== descriptor.size)
    return new Response("Missing", { status: 404, headers });
  return new Response(read.body, {
    headers: {
      ...headers,
      "content-type": "application/octet-stream",
      "x-content-sha256": descriptor.sha256,
    },
  });
}
export default {
  async fetch(request, env) {
    if (!(await authorized(request, env)))
      return new Response("Unauthorized", { status: 401 });
    try {
      const url = new URL(request.url);
      if (request.method === "POST" && url.pathname === "/run-backups") {
        const result = await coordinate(env, "manual");
        return Response.json(result, {
          status: result.jobs.some((j) => j.success === false)
            ? 500
            : result.inProgress
              ? 202
              : 200,
          headers,
        });
      }
      if (
        request.method === "POST" &&
        ["/backup", "/advance"].includes(url.pathname)
      ) {
        const response = await durable(env).fetch(
          "https://capture.internal/" +
            (url.pathname === "/backup" ? "start?origin=manual" : "advance"),
          { method: "POST" },
        );
        const result = await response.json();
        return Response.json(
          result?.receipt || {
            complete: false,
            inProgress:
              result && !["failed", "complete"].includes(result.phase),
            progress: result,
          },
          {
            status:
              result?.phase === "failed"
                ? 500
                : result?.phase === "complete"
                  ? 200
                  : 202,
            headers,
          },
        );
      }
      if (request.method === "GET" && url.pathname === "/status") {
        const { state } = await readStatus(env),
          now = Date.now();
        for (const name of ["tether", "accsync", "maintenance"]) {
          const item = (state.jobs[name] ||= {});
          item.ageSeconds = item.lastSuccess
            ? Math.max(
                0,
                Math.floor(
                  (now - Date.parse(item.lastSuccess.finishedAt)) / 1000,
                ),
              )
            : null;
          item.fresh = item.ageSeconds !== null && item.ageSeconds <= 26 * 3600;
        }
        return Response.json(
          {
            ...state,
            checkedAt: new Date(now).toISOString(),
            capacity: CAPACITY,
          },
          { headers },
        );
      }
      if (
        request.method === "GET" &&
        ["/snapshot", "/object", "/index", "/shard", "/chunk"].includes(
          url.pathname,
        ) &&
        validSnapshot(url.searchParams.get("key"))
      )
        return download(request, env, url);
      return new Response("Not found", { status: 404, headers });
    } catch {
      return Response.json(
        { complete: false, error: "recovery_failed_inspect_operator_status" },
        { status: 500, headers },
      );
    }
  },
  async scheduled(_event, env) {
    const result = await coordinate(env, "scheduled");
    if (result.jobs.some((j) => j.success === false))
      throw Error("one_or_more_recovery_jobs_failed");
  },
};
