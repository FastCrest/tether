/**
 * Reflex episode data upload endpoint — Cloudflare Worker + R2.
 *
 * Legacy read-only statistics service. Upload authority moved to
 * contribution-worker; both former upload routes return 410 without reading
 * the request body or touching storage.
 *
 * R2 layout: reflex-raw-episodes/{contributor_hash}/{date}/{episode_id}.parquet
 *
 * Endpoints:
 *   POST   /v1/episodes/upload        — retired (410; use contribution-worker)
 *   POST   /v1/episodes/upload-url    — retired (410; use contribution-worker)
 *   GET    /v1/contributor/{hash}/stats — retired (410; use contribution-worker)
 *   GET    /v1/stats                   — retired (410; use contribution-worker)
 *   GET    /healthz                    — health check
 *
 * Deploy:
 *   cd infra/data-worker
 *   wrangler d1 create reflex-data
 *   wrangler r2 bucket create reflex-raw-episodes
 *   # Update wrangler.toml with database_id + bucket binding
 *   wrangler d1 execute reflex-data --file=schema.sql
 *   wrangler deploy
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;

    // Health check
    if (request.method === "GET" && path === "/healthz") {
      return jsonResponse({ status: "ok", service: "reflex-data-worker" });
    }

    // Global stats
    if (request.method === "GET" && path === "/v1/stats") {
      return retiredUploadResponse();
    }

    // Per-contributor stats
    const contributorMatch = path.match(/^\/v1\/contributor\/([a-f0-9]{16})\/stats$/);
    if (request.method === "GET" && contributorMatch) {
      return retiredUploadResponse();
    }

    // Direct upload
    if (request.method === "POST" && path === "/v1/episodes/upload") {
      return retiredUploadResponse();
    }

    // Presigned upload URL
    if (request.method === "POST" && path === "/v1/episodes/upload-url") {
      return retiredUploadResponse();
    }

    return new Response("Not Found", { status: 404 });
  },
};


function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function retiredUploadResponse() {
  return jsonResponse({
    error: "upload_endpoint_retired",
    message: "Uploads are accepted only by contribution-worker using Contributor Authentication v1.",
    service: "contribution-worker",
    register_endpoint: "/v1/contributors/register",
    sign_endpoint: "/v1/uploads/sign",
  }, 410);
}
