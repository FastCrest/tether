const TABLES = [
  "contributors",
  "uploads",
  "daily_uploads",
  "revoke_requests",
  "contributor_keys",
  "contributor_nonces",
];
const encoder = new TextEncoder();
const LIMITS = {
  objects: 200,
  objectBytes: 128 * 1024 * 1024,
  totalBytes: 512 * 1024 * 1024,
  rows: 10000,
  manifestBytes: 8 * 1024 * 1024,
};
const STATUS_KEY = "status/latest.json";
const hex = (bytes) =>
  Array.from(new Uint8Array(bytes), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
const hash = async (bytes) => hex(await crypto.subtle.digest("SHA-256", bytes));
async function streamHash(body) {
  const digest = new crypto.DigestStream("SHA-256");
  await body.pipeTo(digest);
  return hex(await digest.digest);
}
async function inventory(env) {
  let cursor;
  const objects = [];
  do {
    const page = await env.SOURCE.list({
      limit: 1000,
      cursor,
      include: ["httpMetadata", "customMetadata"],
    });
    objects.push(...page.objects);
    if (objects.length > LIMITS.objects)
      throw Error("object_count_budget_exceeded");
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return objects;
}
const identity = (items) =>
  JSON.stringify(
    items
      .map(({ key, etag, version, size }) => ({ key, etag, version, size }))
      .sort((a, b) => a.key.localeCompare(b.key)),
  );
async function capture(env) {
  const capturedAt = new Date().toISOString();
  const prefix =
    "snapshots/" +
    capturedAt.replace(/[-:]/g, "").replace(/\.\d{3}/, "") +
    "-" +
    crypto.randomUUID() +
    "/";
  const before = await inventory(env);
  const objectBytes = before.reduce((n, o) => n + o.size, 0);
  if (
    objectBytes > LIMITS.totalBytes ||
    before.some((o) => o.size > LIMITS.objectBytes)
  )
    throw Error("object_bytes_budget_exceeded");
  const getSchema = async () =>
    (
      await env.DB.prepare(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_cf_%' ORDER BY type,name",
      ).all()
    ).results;
  const schema = await getSchema();
  const getRows = async () => {
    const r = await env.DB.batch(
      TABLES.map((t) =>
        env.DB.prepare('SELECT * FROM "' + t + '" LIMIT ' + (LIMITS.rows + 1)),
      ),
    );
    if (r.some((v) => v.results.length > LIMITS.rows))
      throw Error("row_budget_exceeded");
    return Object.fromEntries(TABLES.map((t, i) => [t, r[i].results]));
  };
  const tables = await getRows(),
    objects = [];
  for (const item of before) {
    const o = await env.SOURCE.get(item.key, {
      onlyIf: { etagMatches: item.etag },
    });
    if (!o?.body || o.version !== item.version || o.size !== item.size)
      throw Error("source_version_changed");
    // One backpressured pipeline hashes and copies; tee() would let the faster
    // consumer queue an entire large object while the upload is slower.
    const backupKey =
      prefix + "objects/" + String(objects.length).padStart(6, "0");
    const digest = new crypto.DigestStream("SHA-256");
    const writer = digest.getWriter();
    const hashing = new TransformStream({
      async transform(chunk, controller) {
        await writer.write(chunk);
        controller.enqueue(chunk);
      },
      async flush() {
        await writer.close();
      },
    });
    const fixed = new FixedLengthStream(item.size);
    const controller = new AbortController();
    const copying = o.body
      .pipeThrough(hashing)
      .pipeTo(fixed.writable, { signal: controller.signal })
      .catch(async (error) => {
        await writer.abort(error).catch(() => {});
        throw error;
      });
    let written, sha256;
    try {
      [written, , sha256] = await Promise.all([
        env.BACKUPS.put(backupKey, fixed.readable, {
          onlyIf: { etagDoesNotMatch: "*" },
          httpMetadata: o.httpMetadata,
          customMetadata: { sourceVersion: o.version },
        }),
        copying,
        digest.digest.then(hex),
      ]);
    } catch (error) {
      controller.abort(error);
      throw error;
    }
    if (!written || written.size !== item.size)
      throw Error("backup_copy_failed");
    const read = await env.BACKUPS.get(backupKey);
    if (
      !read?.body ||
      read.size !== item.size ||
      (await streamHash(read.body)) !== sha256
    )
      throw Error("backup_object_readback_mismatch");
    objects.push({
      key: item.key,
      version: item.version,
      etag: item.etag,
      size: item.size,
      httpMetadata: item.httpMetadata,
      customMetadata: item.customMetadata,
      sha256,
      backupKey,
    });
  }
  if (
    identity(before) !== identity(await inventory(env)) ||
    JSON.stringify(tables) !== JSON.stringify(await getRows()) ||
    JSON.stringify(schema) !== JSON.stringify(await getSchema())
  )
    throw Error("source_changed_during_backup");
  const snapshot = {
    version: 2,
    complete: true,
    capturedAt,
    schema,
    tables,
    objects,
  };
  const body = JSON.stringify(snapshot);
  if (encoder.encode(body).length > LIMITS.manifestBytes)
    throw Error("snapshot_bytes_budget_exceeded");
  const key = prefix + "snapshot.json",
    sha256 = await hash(encoder.encode(body));
  const written = await env.BACKUPS.put(key, body, {
    onlyIf: { etagDoesNotMatch: "*" },
    httpMetadata: { contentType: "application/json" },
    customMetadata: { sha256 },
  });
  if (!written) throw Error("snapshot_creation_failed");
  const read = await env.BACKUPS.get(key);
  if (!read || (await hash(await read.arrayBuffer())) !== sha256)
    throw Error("backup_readback_mismatch");
  return {
    complete: true,
    version: 2,
    key,
    sha256,
    tables: Object.fromEntries(TABLES.map((t) => [t, tables[t].length])),
    objects: objects.length,
    objectBytes,
    capturedAt,
  };
}
async function authorized(request, env) {
  if (!env.RECOVERY_TOKEN) return false;
  const [a, b] = await Promise.all([
    hash(encoder.encode(request.headers.get("authorization") ?? "")),
    hash(encoder.encode("Bearer " + env.RECOVERY_TOKEN)),
  ]);
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
async function readStatus(env) {
  const o = await env.BACKUPS.get(STATUS_KEY);
  return { object: o, state: o ? await o.json() : { version: 1, jobs: {} } };
}
async function record(env, job, result) {
  for (let retry = 0; retry < 5; retry++) {
    const { object, state } = await readStatus(env);
    const previous = state.jobs[job] || {};
    state.jobs[job] = {
      ...previous,
      latest: result,
      [result.success ? "lastSuccess" : "lastFailure"]: result,
      ...(result.origin === "scheduled"
        ? {
            lastScheduledAttempt: result,
            ...(result.success ? { lastScheduledSuccess: result } : {}),
          }
        : {}),
    };
    const written = await env.BACKUPS.put(STATUS_KEY, JSON.stringify(state), {
      onlyIf: object ? { etagMatches: object.etag } : { etagDoesNotMatch: "*" },
      httpMetadata: { contentType: "application/json" },
    });
    if (written) return;
  }
  throw Error("status_write_conflict");
}
async function job(env, name, origin, operation) {
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
    // Persist only a bounded error code, never arbitrary provider messages/credentials.
    const code = /^[a-z0-9_]{1,100}$/.test(error.message)
      ? error.message
      : "operation_failed";
    await record(env, name, {
      success: false,
      origin,
      startedAt,
      finishedAt: new Date().toISOString(),
      error: code,
    });
    throw Error(code);
  }
}
async function invoke(binding, token, path, missing) {
  if (!binding || !token) throw Error(missing);
  const r = await binding.fetch("https://backup.internal" + path, {
    method: "POST",
    headers: { authorization: "Bearer " + token },
    redirect: "manual",
    signal: AbortSignal.timeout(90000),
  });
  if (!r.ok) throw Error("service_http_" + r.status);
  const receipt = await r.json();
  if (path === "/internal/maintenance") {
    if (receipt.status !== "ok" || receipt.purges_executed !== 0)
      throw Error("maintenance_invalid_receipt");
  } else if (receipt.complete !== true) throw Error("service_incomplete");
  return receipt;
}
async function coordinate(env, origin) {
  // Housekeeping writes upload/nonce rows. Let it settle before capture's
  // consistency window, while still attempting both backups after any failure.
  const maintenance = await Promise.allSettled([
    job(env, "maintenance", origin, () =>
      invoke(
        env.CONTRIBUTIONS,
        env.MAINTENANCE_TRIGGER_TOKEN,
        "/internal/maintenance",
        "maintenance_binding_or_token_missing",
      ),
    ),
  ]);
  const backups = await Promise.allSettled([
    job(env, "tether", origin, () => capture(env)),
    job(env, "accsync", origin, () =>
      invoke(
        env.ACCSYNC_BACKUP,
        env.ACCSYNC_BACKUP_TRIGGER_TOKEN,
        "/backup",
        "accsync_binding_or_token_missing",
      ),
    ),
  ]);
  const settled = [...backups, ...maintenance];
  const jobs = settled.map((r, i) => ({
    project: ["tether", "accsync", "maintenance"][i],
    success: r.status === "fulfilled",
    ...(r.status === "fulfilled"
      ? { receipt: r.value }
      : { error: r.reason.message }),
  }));
  for (const result of jobs)
    console.log(
      JSON.stringify({ event: "recovery_backup_job", origin, ...result }),
    );
  return { complete: settled.every((r) => r.status === "fulfilled"), jobs };
}
const validSnapshot = (key) =>
  /^snapshots\/[A-Za-z0-9TZ-]+\/snapshot\.json$/.test(key ?? "");
export default {
  async fetch(request, env) {
    if (!(await authorized(request, env)))
      return new Response("Unauthorized", { status: 401 });
    const headers = { "cache-control": "no-store" };
    try {
      const u = new URL(request.url);
      if (request.method === "POST" && u.pathname === "/run-backups") {
        const result = await coordinate(env, "manual");
        return Response.json(result, {
          status: result.complete ? 200 : 500,
          headers,
        });
      }
      if (request.method === "POST" && u.pathname === "/backup")
        return Response.json(
          await job(env, "tether", "manual", () => capture(env)),
          { headers },
        );
      if (request.method === "GET" && u.pathname === "/status") {
        const { state } = await readStatus(env);
        const now = Date.now();
        for (const name of ["tether", "accsync", "maintenance"]) {
          const entry = (state.jobs[name] ||= {});
          entry.ageSeconds = entry.lastSuccess
            ? Math.max(
                0,
                Math.floor(
                  (now - Date.parse(entry.lastSuccess.finishedAt)) / 1000,
                ),
              )
            : null;
          entry.fresh =
            entry.ageSeconds !== null && entry.ageSeconds <= 26 * 3600;
        }
        return Response.json(
          { ...state, checkedAt: new Date(now).toISOString(), limits: LIMITS },
          { headers },
        );
      }
      if (
        request.method === "GET" &&
        ["/snapshot", "/object"].includes(u.pathname) &&
        validSnapshot(u.searchParams.get("key"))
      ) {
        const key = u.searchParams.get("key"),
          o = await env.BACKUPS.get(key);
        if (!o) return new Response("Missing", { status: 404, headers });
        if (u.pathname === "/snapshot")
          return new Response(o.body, {
            headers: { ...headers, "content-type": "application/json" },
          });
        const manifest = await o.json(),
          index = u.searchParams.get("index");
        if (
          manifest.version !== 2 ||
          manifest.complete !== true ||
          !/^(0|[1-9][0-9]{0,2})$/.test(index ?? "")
        )
          return new Response("Invalid object", { status: 400, headers });
        const item = manifest.objects[Number(index)];
        const expected =
          key.slice(0, -"snapshot.json".length) +
          "objects/" +
          String(index).padStart(6, "0");
        if (!item || item.backupKey !== expected)
          return new Response("Missing", { status: 404, headers });
        const object = await env.BACKUPS.get(item.backupKey);
        if (!object || object.size !== item.size)
          return new Response("Missing", { status: 404, headers });
        return new Response(object.body, {
          headers: {
            ...headers,
            "content-type": "application/octet-stream",
            "x-content-sha256": item.sha256,
          },
        });
      }
      return new Response("Not found", { status: 404, headers });
    } catch {
      return Response.json(
        { complete: false, error: "backup_failed_inspect_operator_status" },
        { status: 500, headers },
      );
    }
  },
  async scheduled(_event, env) {
    const result = await coordinate(env, "scheduled");
    if (!result.complete) throw Error("one_or_more_recovery_jobs_failed");
  },
};
