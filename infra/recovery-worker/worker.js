const TABLES = ["contributors", "uploads", "daily_uploads", "revoke_requests", "contributor_keys", "contributor_nonces"];
const encoder = new TextEncoder();
const hash = async (bytes) =>
  Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
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
    if (objects.length > 40) throw Error("object_count_budget_exceeded");
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);
  return objects;
}
async function capture(env) {
  const before = await inventory(env);
  if (before.reduce((n, o) => n + o.size, 0) > 4 * 1024 * 1024)
    throw Error("object_bytes_budget_exceeded");
  const schema = (
    await env.DB.prepare(
      "SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_cf_%' ORDER BY type,name",
    ).all()
  ).results;
  const getRows = async () => {
    const r = await env.DB.batch(
      TABLES.map((t) =>
        env.DB.prepare('SELECT * FROM "' + t + '" LIMIT 10001'),
      ),
    );
    if (r.some((v) => v.results.length > 10000))
      throw Error("row_budget_exceeded");
    return Object.fromEntries(TABLES.map((t, i) => [t, r[i].results]));
  };
  const tables = await getRows(),
    objects = [];
  for (const item of before) {
    const o = await env.SOURCE.get(item.key, {
      onlyIf: { etagMatches: item.etag },
    });
    if (!o?.body) throw Error("source_version_changed");
    const bytes = new Uint8Array(await o.arrayBuffer());
    if (bytes.length !== item.size) throw Error("source_size_changed");
    let binary = "";
    for (let i = 0; i < bytes.length; i += 8192)
      binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    objects.push({
      ...item,
      sha256: await hash(bytes),
      bodyBase64: btoa(binary),
    });
  }
  const identity = (items) =>
    JSON.stringify(
      items
        .map(({ key, etag, size }) => ({ key, etag, size }))
        .sort((a, b) => a.key.localeCompare(b.key)),
    );
  if (
    identity(before) !== identity(await inventory(env)) ||
    JSON.stringify(tables) !== JSON.stringify(await getRows())
  )
    throw Error("source_changed_during_backup");
  const snapshot = {
    version: 1,
    complete: true,
    capturedAt: new Date().toISOString(),
    schema,
    tables,
    objects,
  };
  const body = JSON.stringify(snapshot);
  if (encoder.encode(body).length > 8 * 1024 * 1024)
    throw Error("snapshot_bytes_budget_exceeded");
  const key =
    "snapshots/" +
    snapshot.capturedAt.replace(/[-:]/g, "").replace(/\.\d{3}/, "") +
    "-" +
    crypto.randomUUID() +
    "/snapshot.json";
  const sha256 = await hash(encoder.encode(body));
  await env.BACKUPS.put(key, body, {
    onlyIf: { etagDoesNotMatch: "*" },
    httpMetadata: { contentType: "application/json" },
    customMetadata: { sha256 },
  });
  const read = await env.BACKUPS.get(key);
  if (!read || (await hash(await read.arrayBuffer())) !== sha256)
    throw Error("backup_readback_mismatch");
  return {
    complete: true,
    key,
    sha256,
    tables: Object.fromEntries(TABLES.map((t) => [t, tables[t].length])),
    objects: objects.length,
    objectBytes: before.reduce((n, o) => n + o.size, 0),
    capturedAt: snapshot.capturedAt,
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
async function coordinate(env) {
  const jobs = [capture(env)];
  jobs.push(
    (async () => {
      if (!env.ACCSYNC_BACKUP_TRIGGER_TOKEN)
        throw Error("accsync_trigger_token_missing");
      if (!env.ACCSYNC_BACKUP) throw Error("accsync_service_binding_missing");
      const r = await env.ACCSYNC_BACKUP.fetch(
        "https://backup.internal/backup",
        {
          method: "POST",
          headers: {
            authorization: "Bearer " + env.ACCSYNC_BACKUP_TRIGGER_TOKEN,
          },
          redirect: "manual",
          signal: AbortSignal.timeout(90000),
        },
      );
      if (!r.ok) throw Error("accsync_backup_http_" + r.status);
      return r.json();
    })(),
  );
  const settled = await Promise.allSettled(jobs);
  for (let i = 0; i < settled.length; i++) {
    const r = settled[i];
    console.log(
      JSON.stringify({
        event: "recovery_backup_job",
        project: i === 0 ? "tether" : "accsync",
        success: r.status === "fulfilled",
        ...(r.status === "fulfilled" ? r.value : { reason: r.reason.message }),
      }),
    );
  }
  if (settled.some((r) => r.status === "rejected"))
    throw Error("one_or_more_backups_failed");
  return settled.map((r) => r.value);
}
export default {
  async fetch(request, env) {
    if (!(await authorized(request, env)))
      return new Response("Unauthorized", { status: 401 });
    try {
      const u = new URL(request.url);
      if (request.method === "POST" && u.pathname === "/run-backups")
        return Response.json(
          { complete: true, jobs: await coordinate(env) },
          { headers: { "cache-control": "no-store" } },
        );
      if (request.method === "POST" && u.pathname === "/backup")
        return Response.json(await capture(env), {
          headers: { "cache-control": "no-store" },
        });
      if (
        request.method === "GET" &&
        u.pathname === "/snapshot" &&
        /^snapshots\/[A-Za-z0-9TZ-]+\/snapshot\.json$/.test(
          u.searchParams.get("key") ?? "",
        )
      ) {
        const o = await env.BACKUPS.get(u.searchParams.get("key"));
        return o
          ? new Response(o.body, {
              headers: {
                "content-type": "application/json",
                "cache-control": "no-store",
              },
            })
          : new Response("Missing", { status: 404 });
      }
      return new Response("Not found", { status: 404 });
    } catch {
      return Response.json(
        { complete: false, error: "backup_failed_inspect_operator_logs" },
        { status: 500 },
      );
    }
  },
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(coordinate(env));
  },
};
