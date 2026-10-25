export const TABLES = [
  "contributors",
  "uploads",
  "daily_uploads",
  "revoke_requests",
  "contributor_keys",
  "contributor_nonces",
];
export const CAPACITY = {
  chunkBytes: 16 * 1024 * 1024,
  shardBytes: 4 * 1024 * 1024,
  inventoryPage: 50,
  rowPage: 250,
  maxObjectBytes: 64 * 1024 ** 3,
  maxShards: 10000,
  deadlineHours: 6,
};
const encoder = new TextEncoder();
const hex = (value) =>
  Array.from(new Uint8Array(value), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
export const hash = async (bytes) =>
  hex(await crypto.subtle.digest("SHA-256", bytes));
export const jsonHash = async (value) =>
  hash(encoder.encode(JSON.stringify(value)));
const number = (value) => String(value).padStart(8, "0");
const schemaQuery =
  "SELECT type,name,tbl_name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_cf_%' ORDER BY type,name";
const schema = async (env) => (await env.DB.prepare(schemaQuery).all()).results;
const objectIdentity = (objects) =>
  objects.map(({ key, etag, version, size, httpMetadata, customMetadata }) => ({
    key,
    etag,
    version,
    size,
    httpMetadata,
    customMetadata,
  }));
export async function streamHash(body) {
  const digest = new crypto.DigestStream("SHA-256");
  await body.pipeTo(digest);
  return hex(await digest.digest);
}
export async function immutableJSON(env, key, value) {
  const body = JSON.stringify(value),
    size = encoder.encode(body).length,
    sha256 = await hash(encoder.encode(body));
  if (size > CAPACITY.shardBytes)
    throw Error("manifest_shard_capacity_exceeded");
  const existing = await env.BACKUPS.get(key);
  if (existing) {
    if ((await streamHash(existing.body)) !== sha256)
      throw Error("source_changed_during_retry");
  } else {
    const written = await env.BACKUPS.put(key, body, {
      onlyIf: { etagDoesNotMatch: "*" },
      httpMetadata: { contentType: "application/json" },
      customMetadata: { sha256 },
    });
    if (!written) throw Error("immutable_write_conflict");
    const read = await env.BACKUPS.get(key);
    if (!read || (await streamHash(read.body)) !== sha256)
      throw Error("backup_readback_mismatch");
  }
  return { key, size, sha256 };
}
async function readJSON(env, descriptor) {
  if (!descriptor || descriptor.size > CAPACITY.shardBytes)
    throw Error("manifest_shard_capacity_exceeded");
  const object = await env.BACKUPS.get(descriptor.key);
  if (!object || object.size !== descriptor.size)
    throw Error("backup_shard_missing");
  const bytes = await object.arrayBuffer();
  if ((await hash(bytes)) !== descriptor.sha256)
    throw Error("backup_shard_checksum_mismatch");
  return JSON.parse(new TextDecoder().decode(bytes));
}
async function copiedChunk(env, item, key, offset, length) {
  const source = await env.SOURCE.get(item.key, {
    onlyIf: { etagMatches: item.etag },
    ...(item.size ? { range: { offset, length } } : {}),
  });
  if (
    !source?.body ||
    source.version !== item.version ||
    source.size !== item.size
  )
    throw Error("source_version_changed");
  if (
    item.size &&
    (source.range?.offset !== offset || source.range?.length !== length)
  )
    throw Error("source_range_mismatch");
  const existing = await env.BACKUPS.get(key);
  if (existing) {
    const [actual, expected] = await Promise.all([
      streamHash(existing.body),
      streamHash(source.body),
    ]);
    if (existing.size !== length || actual !== expected)
      throw Error("backup_chunk_retry_mismatch");
    return { key, size: length, sha256: actual, offset };
  }
  const digest = new crypto.DigestStream("SHA-256"),
    writer = digest.getWriter(),
    controller = new AbortController();
  const hashing = new TransformStream({
    async transform(chunk, output) {
      await writer.write(chunk);
      output.enqueue(chunk);
    },
    async flush() {
      await writer.close();
    },
  });
  const fixed = new FixedLengthStream(length);
  const copying = source.body
    .pipeThrough(hashing)
    .pipeTo(fixed.writable, { signal: controller.signal })
    .catch(async (error) => {
      await writer.abort(error).catch(() => {});
      throw error;
    });
  let written, sha256;
  try {
    [written, , sha256] = await Promise.all([
      env.BACKUPS.put(key, fixed.readable, {
        onlyIf: { etagDoesNotMatch: "*" },
        customMetadata: { sourceVersion: item.version },
      }),
      copying,
      digest.digest.then(hex),
    ]);
  } catch (error) {
    controller.abort(error);
    throw error;
  }
  const read = await env.BACKUPS.get(key);
  if (
    !written ||
    written.size !== length ||
    !read ||
    read.size !== length ||
    (await streamHash(read.body)) !== sha256
  )
    throw Error("backup_chunk_readback_mismatch");
  return { key, size: length, sha256, offset };
}
async function rows(env, table, after, limit) {
  // Maintained tables are ordinary rowid tables. Preserve rowid only as a paging
  // cursor; it is never inserted into restored application data.
  const query =
    'SELECT rowid AS "__backup_rowid", * FROM "' +
    table +
    '"' +
    (after === null ? "" : " WHERE rowid > ?") +
    " ORDER BY rowid LIMIT ?";
  const statement = env.DB.prepare(query).bind(
    ...(after === null ? [limit] : [after, limit]),
  );
  const result = (await statement.all()).results;
  if (result.some((r) => !Number.isSafeInteger(r.__backup_rowid)))
    throw Error("rowid_precision_unsupported");
  return result;
}
async function boundedRows(env, table, after) {
  let limit = CAPACITY.rowPage;
  while (true) {
    try {
      const result = await rows(env, table, after, limit);
      if (
        encoder.encode(JSON.stringify(result)).length <=
        CAPACITY.shardBytes - 1024
      )
        return { rows: result, limit };
      if (limit === 1) throw Error("single_row_capacity_exceeded");
    } catch (error) {
      if (limit === 1) throw error;
    }
    limit = Math.max(1, Math.floor(limit / 2));
  }
}
export class Capture {
  constructor(storage, env) {
    this.storage = storage;
    this.env = env;
  }
  async start(origin, maintenance) {
    const previous = await this.storage.get("job");
    if (previous && !["complete", "failed"].includes(previous.phase))
      return previous;
    const startedAt = new Date().toISOString(),
      id =
        startedAt.replace(/[-:]/g, "").replace(/\.\d{3}/, "") +
        "-" +
        crypto.randomUUID();
    const job = {
      id,
      prefix: "snapshots/" + id + "/",
      startedAt,
      capturedAt: null,
      origin,
      phase: maintenance ? "maintenance" : "inventory",
      inventoryPages: 0,
      cursor: undefined,
      copyPage: 0,
      copyItem: 0,
      offset: 0,
      chunk: 0,
      objects: 0,
      objectBytes: 0,
      copiedBytes: 0,
      tableIndex: 0,
      after: null,
      rowPages: 0,
      rows: Object.fromEntries(TABLES.map((t) => [t, 0])),
      descriptorCount: 0,
      verifyPage: 0,
      attempts: 0,
      lastError: null,
    };
    await this.storage.put("job", job);
    await this.storage.setAlarm(Date.now() + 1000);
    return job;
  }
  async descriptor(job, kind, value) {
    if (job.descriptorCount >= CAPACITY.maxShards)
      throw Error("manifest_shard_count_capacity_exceeded");
    await this.storage.put(
      job.id + ":descriptor:" + number(job.descriptorCount),
      { ...value, kind },
    );
    job.descriptorCount++;
  }
  async step(maintenance) {
    const job = await this.storage.get("job");
    if (!job || ["complete", "failed"].includes(job.phase)) return job;
    if (
      Date.now() - Date.parse(job.startedAt) >
      CAPACITY.deadlineHours * 3600000
    )
      throw Error("capture_deadline_exceeded");
    const env = this.env;
    if (job.phase === "maintenance") {
      // Maintenance records its own failure; capture must proceed independently.
      await maintenance();
      job.phase = "inventory";
    } else if (job.phase === "inventory") {
      const page = await env.SOURCE.list({
        limit: CAPACITY.inventoryPage,
        cursor: job.cursor,
        include: ["httpMetadata", "customMetadata"],
      });
      if (page.objects.some((o) => o.size > CAPACITY.maxObjectBytes))
        throw Error("single_object_capacity_exceeded");
      if (
        job.objects + page.objects.length >
        CAPACITY.maxShards - TABLES.length * 2
      )
        throw Error("inventory_object_capacity_exceeded");
      const entries = objectIdentity(page.objects);
      const descriptor = await immutableJSON(
        env,
        job.prefix + "inventory/" + number(job.inventoryPages) + ".json",
        { objects: entries },
      );
      await this.storage.put(
        job.id + ":inventory:" + number(job.inventoryPages),
        descriptor,
      );
      job.inventoryPages++;
      job.objects += entries.length;
      job.objectBytes += entries.reduce((n, o) => n + o.size, 0);
      job.cursor = page.truncated ? page.cursor : undefined;
      if (!page.truncated) job.phase = "copy";
    } else if (job.phase === "copy") {
      if (job.copyPage >= job.inventoryPages) {
        job.phase = "rows";
        job.capturedAt = new Date().toISOString();
        job.schema = await schema(env);
        if (encoder.encode(JSON.stringify(job.schema)).length > 65536)
          throw Error("schema_capacity_exceeded");
      } else {
        const page = await readJSON(
          env,
          await this.storage.get(job.id + ":inventory:" + number(job.copyPage)),
        );
        const item = page.objects[job.copyItem];
        if (!item) {
          job.copyPage++;
          job.copyItem = 0;
        } else {
          const index = job.completedObjects || 0;
          const length = Math.min(CAPACITY.chunkBytes, item.size - job.offset);
          const descriptor = await copiedChunk(
            env,
            item,
            job.prefix + "media/" + number(index) + "/" + number(job.chunk),
            job.offset,
            length,
          );
          await this.storage.put(
            job.id + ":chunk:" + number(job.chunk),
            descriptor,
          );
          job.offset += length;
          job.chunk++;
          job.copiedBytes += length;
          if (job.offset === item.size) {
            const chunks = [];
            for (let start = 0; start < job.chunk; start += 100) {
              const keys = Array.from(
                { length: Math.min(100, job.chunk - start) },
                (_, i) => job.id + ":chunk:" + number(start + i),
              );
              const values = await this.storage.get(keys);
              for (const key of keys) chunks.push(values.get(key));
            }
            const entry = { ...item, chunks };
            const objectDescriptor = await immutableJSON(
              env,
              job.prefix + "objects/" + number(index) + ".json",
              entry,
            );
            await this.descriptor(job, "object", objectDescriptor);
            job.completedObjects = index + 1;
            job.copyItem++;
            job.offset = 0;
            job.chunk = 0;
          }
        }
      }
    } else if (job.phase === "rows") {
      if (job.tableIndex === TABLES.length) {
        job.phase = "verify_rows";
        job.verifyPage = 0;
      } else {
        const table = TABLES[job.tableIndex],
          page = await boundedRows(env, table, job.after);
        const descriptor = await immutableJSON(
          env,
          job.prefix + "tables/" + number(job.rowPages) + ".json",
          { table, after: job.after, limit: page.limit, rows: page.rows },
        );
        await this.storage.put(
          job.id + ":row:" + number(job.rowPages),
          descriptor,
        );
        await this.descriptor(job, "table", { ...descriptor, table });
        job.rowPages++;
        job.rows[table] += page.rows.length;
        if (!page.rows.length) {
          job.tableIndex++;
          job.after = null;
        } else job.after = page.rows.at(-1).__backup_rowid;
      }
    } else if (job.phase === "verify_rows") {
      if (job.verifyPage === job.rowPages) {
        job.phase = "verify_inventory";
        job.verifyPage = 0;
        job.cursor = undefined;
      } else {
        const descriptor = await this.storage.get(
            job.id + ":row:" + number(job.verifyPage),
          ),
          page = await readJSON(env, descriptor);
        const actual = await rows(env, page.table, page.after, page.limit);
        if ((await jsonHash(actual)) !== (await jsonHash(page.rows)))
          throw Error("source_tables_changed_during_backup");
        job.verifyPage++;
      }
    } else if (job.phase === "verify_inventory") {
      const page = await env.SOURCE.list({
        limit: CAPACITY.inventoryPage,
        cursor: job.cursor,
        include: ["httpMetadata", "customMetadata"],
      });
      const descriptor = await this.storage.get(
        job.id + ":inventory:" + number(job.verifyPage),
      );
      if (!descriptor) throw Error("source_inventory_changed_during_backup");
      const expected = await readJSON(env, descriptor);
      if (
        (await jsonHash(objectIdentity(page.objects))) !==
        (await jsonHash(expected.objects))
      )
        throw Error("source_inventory_changed_during_backup");
      job.verifyPage++;
      job.cursor = page.truncated ? page.cursor : undefined;
      if (!page.truncated) {
        if (job.verifyPage !== job.inventoryPages)
          throw Error("source_inventory_changed_during_backup");
        job.phase = "publish";
      }
    } else if (job.phase === "publish") {
      if ((await jsonHash(job.schema)) !== (await jsonHash(await schema(env))))
        throw Error("source_schema_changed_during_backup");
      job.publicationAt ||= new Date().toISOString();
      await this.storage.put("job", job);
      const indexes = [];
      for (let start = 0; start < job.descriptorCount; start += 100) {
        const keys = Array.from(
          { length: Math.min(100, job.descriptorCount - start) },
          (_, i) => job.id + ":descriptor:" + number(start + i),
        );
        const values = await this.storage.get(keys);
        const entries = keys.map((key) => values.get(key));
        indexes.push(
          await immutableJSON(
            env,
            job.prefix + "indexes/" + number(indexes.length) + ".json",
            { entries },
          ),
        );
      }
      const manifest = {
        version: 3,
        complete: true,
        startedAt: job.startedAt,
        capturedAt: job.capturedAt,
        completedAt: job.publicationAt,
        consistency: "paginated_double_pass_not_atomic",
        schema: job.schema,
        tables: job.rows,
        objectCount: job.objects,
        objectBytes: job.objectBytes,
        indexes,
      };
      const receipt = await immutableJSON(
        env,
        job.prefix + "snapshot.json",
        manifest,
      );
      job.phase = "complete";
      job.finishedAt = new Date().toISOString();
      job.receipt = {
        complete: true,
        version: 3,
        ...receipt,
        tables: job.rows,
        objects: job.objects,
        objectBytes: job.objectBytes,
        capturedAt: job.capturedAt,
      };
    }
    job.attempts = 0;
    job.lastError = null;
    job.updatedAt = new Date().toISOString();
    await this.storage.put("job", job);
    if (job.phase !== "complete")
      await this.storage.setAlarm(Date.now() + 1000);
    return job;
  }
}
export { readJSON };
