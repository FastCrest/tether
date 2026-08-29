const CONTRIBUTOR_ID_RE = /^ctr_[0-9a-f]{32}$/;
// Historical consent receipts used free_<hardware-hash>_<random-suffix>, and
// a small number of pre-release rows used longer underscore-delimited tokens.
// The purge path accepts that server-stored namespace without allowing path
// separators. Payload writes remain Auth-v1 ctr_* only.
// Explicit internal purge compatibility for historical Pro customer IDs.
// No slash/control characters are accepted, and this regex is never used by
// payload writes or public reservation authorization.
const LEGACY_PURGE_ID_RE = /^[^\u0000-\u0020/\\]{1,128}$/;
const SAFE_KEY_RE = /^[A-Za-z0-9._/-]+$/;
const R2_LIST_PAGE_SIZE = 1000;

/**
 * Per-contributor Durable Object that serializes payload writes and purges.
 *
 * D1 owns authorization/lifecycle state. This object owns the storage ordering
 * boundary that D1 and R2 cannot provide across services: a purge either runs
 * after an admitted write and deletes it, or runs first and persistently
 * refuses the later write. No wall-clock lease is used to infer completion.
 */
export class ContributorStorageFence {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.tail = Promise.resolve();
  }

  async fetch(request) {
    const operation = () => this.handle(request);
    const result = this.tail.then(operation, operation);
    this.tail = result.catch(() => undefined);
    return await result;
  }

  async handle(request) {
    const contributorId = request.headers.get("X-Tether-Contributor-Id") || "";
    const path = new URL(request.url).pathname;
    const isPayload = request.method === "PUT" && path === "/payload";
    const isPurge = request.method === "POST" && path === "/purge";
    const validContributor = isPayload
      ? CONTRIBUTOR_ID_RE.test(contributorId)
      : isPurge && (
        CONTRIBUTOR_ID_RE.test(contributorId)
        || LEGACY_PURGE_ID_RE.test(contributorId)
      );
    if (!validContributor) {
      return jsonResponse(400, { error: "invalid_storage_fence_contributor" });
    }
    const owner = await this.state.storage.get("contributor_id");
    if (owner && owner !== contributorId) {
      return jsonResponse(403, { error: "storage_fence_owner_mismatch" });
    }
    if (!owner) await this.state.storage.put("contributor_id", contributorId);

    if (isPayload) {
      return await this.writePayload(request, contributorId);
    }
    if (isPurge) {
      return await this.purgeContributor(contributorId);
    }
    return jsonResponse(404, { error: "storage_fence_operation_not_found" });
  }

  async writePayload(request, contributorId) {
    if (await this.state.storage.get("purged")) {
      return jsonResponse(409, { error: "contributor_storage_purged" });
    }
    const key = request.headers.get("X-Tether-R2-Key") || "";
    const allowedPrefixes = [
      `free-contributors/${contributorId}/`,
      `pro-contributors/${contributorId}/`,
      `enterprise-contributors/${contributorId}/`,
    ];
    if (!SAFE_KEY_RE.test(key) || !allowedPrefixes.some((prefix) => key.startsWith(prefix))) {
      return jsonResponse(400, { error: "invalid_storage_fence_key" });
    }
    let customMetadata;
    try {
      customMetadata = JSON.parse(request.headers.get("X-Tether-R2-Metadata") || "");
    } catch {
      return jsonResponse(400, { error: "invalid_storage_fence_metadata" });
    }
    const bytes = new Uint8Array(await request.arrayBuffer());
    await this.env.CURATE_BUCKET.put(key, bytes, {
      httpMetadata: { contentType: request.headers.get("Content-Type") || "application/octet-stream" },
      customMetadata,
    });
    return jsonResponse(200, { stored: true });
  }

  async purgeContributor(contributorId) {
    // Persist the terminal fence before listing. Because write and purge calls
    // execute through the same per-instance queue, no admitted R2.put can run
    // concurrently with or after this operation.
    await this.state.storage.put("purged", true);
    const prefixes = [
      `free-contributors/${contributorId}/`,
      `pro-contributors/${contributorId}/`,
      `enterprise-contributors/${contributorId}/`,
    ];
    let objectsPurged = 0;
    for (const prefix of prefixes) {
      let cursor;
      while (true) {
        const listed = await this.env.CURATE_BUCKET.list({
          prefix,
          limit: R2_LIST_PAGE_SIZE,
          cursor,
        });
        for (const object of listed.objects || []) {
          await this.env.CURATE_BUCKET.delete(object.key);
          objectsPurged += 1;
        }
        if (!listed.truncated) break;
        cursor = listed.cursor;
      }
    }
    const cumulative = Number(await this.state.storage.get("objects_purged") || 0)
      + objectsPurged;
    await this.state.storage.put("objects_purged", cumulative);
    return jsonResponse(200, { objects_purged: cumulative, purged: true });
  }
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
