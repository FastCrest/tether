/**
 * Reflex telemetry endpoint — Cloudflare Worker.
 *
 * Receives heartbeat POSTs from `pip install fastcrest-tether` deployments
 * running with a valid Pro license OR free-tier telemetry enabled.
 * Validates the payload shape, writes one row per heartbeat to D1,
 * and returns 204 No Content.
 *
 * Deploy:
 *   cd infra/telemetry-worker
 *   npm install -g wrangler
 *   wrangler login
 *   wrangler d1 create reflex-telemetry
 *   # Copy the resulting database_id into wrangler.toml
 *   wrangler d1 execute reflex-telemetry --file=schema.sql
 *   wrangler d1 execute reflex-telemetry --file=migrations/001_free_tier_fields.sql
 *   wrangler deploy
 *   # Optional: bind a custom domain at telemetry.fastcrest.workers.dev
 *
 * Privacy posture (locked Phase 1):
 * - We log Cf-Connecting-IP at the worker level (Cloudflare default) but
 *   never write it to D1. The IP is dropped after request handling.
 * - We never log /act payloads, customer data, or model weights — the
 *   Reflex client side intentionally omits these from the heartbeat.
 * - The org_hash field is SHA256(customer_id)[:16] (Pro) or
 *   SHA256(machine_fingerprint)[:16] (free). Reverse-mapping
 *   requires the billing DB (Pro) or physical machine access (free).
 */

const SCHEMA_VERSION_ACCEPTED = 1;

export default {
  async fetch(request, env, ctx) {
    if (request.method === "GET" && new URL(request.url).pathname === "/healthz") {
      return new Response(JSON.stringify({ status: "ok", schema: SCHEMA_VERSION_ACCEPTED }), {
        headers: { "Content-Type": "application/json" },
      });
    }

    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    const url = new URL(request.url);
    if (url.pathname !== "/v1/heartbeat") {
      return new Response("Not Found", { status: 404 });
    }

    let payload;
    try {
      payload = await request.json();
    } catch (e) {
      return new Response(JSON.stringify({ error: "invalid_json" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Accept tether_version (current clients) OR reflex_version (legacy
    // reflex-vla clients) and normalize to tether_version. Without this the
    // renamed client (which sends tether_version) hit "missing_field" 400 on
    // every heartbeat — 100% telemetry loss. The D1 column stays reflex_version
    // for data continuity.
    if (payload.tether_version == null && payload.reflex_version != null) {
      payload.tether_version = payload.reflex_version;
    }

    // Schema validation. v1 fields are LOCKED — additive-only Phase 2.
    const required = ["schema_version", "license_id", "org_hash", "workload", "tether_version", "timestamp"];
    for (const field of required) {
      if (!(field in payload)) {
        return new Response(
          JSON.stringify({ error: "missing_field", field }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    if (payload.schema_version !== SCHEMA_VERSION_ACCEPTED) {
      return new Response(
        JSON.stringify({
          error: "schema_version_mismatch",
          accepted: SCHEMA_VERSION_ACCEPTED,
          received: payload.schema_version,
        }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    // Defensive size limits. Anything outside these is suspect and
    // probably an injection attempt; reject without storing.
    if (
      typeof payload.license_id !== "string" || payload.license_id.length > 256 ||
      typeof payload.org_hash !== "string" || payload.org_hash.length !== 16 ||
      typeof payload.tether_version !== "string" || payload.tether_version.length > 64 ||
      typeof payload.timestamp !== "string" || payload.timestamp.length > 64 ||
      typeof payload.workload !== "object" || payload.workload === null
    ) {
      return new Response(JSON.stringify({ error: "invalid_field_shape" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    const vlaFamily = String(payload.workload.vla_family || "unknown").slice(0, 64);
    const hardwareTier = String(payload.workload.hardware_tier || "unknown").slice(0, 64);

    // Determine tier: free-tier uses license_id="free"
    const tier = payload.license_id === "free" ? "free" : (payload.tier || "pro");

    // Extract optional free-tier fields (safe defaults)
    const modelName = String(payload.model_name || "unknown").slice(0, 128);
    const hardwareDetail = String(payload.hardware_detail || "unknown").slice(0, 128);
    const latencyP50 = typeof payload.latency_p50 === "number" ? payload.latency_p50 : null;
    const latencyP95 = typeof payload.latency_p95 === "number" ? payload.latency_p95 : null;
    const latencyP99 = typeof payload.latency_p99 === "number" ? payload.latency_p99 : null;
    const errorCount24h = typeof payload.error_count_24h === "number" ? payload.error_count_24h : 0;
    const safetyViolationCount24h = typeof payload.safety_violation_count_24h === "number" ? payload.safety_violation_count_24h : 0;
    const episodeCount24h = typeof payload.episode_count_24h === "number" ? payload.episode_count_24h : 0;
    const actionDim = typeof payload.action_dim === "number" ? payload.action_dim : null;
    const embodiment = String(payload.embodiment || "unknown").slice(0, 64);
    const denoiseSteps = typeof payload.denoise_steps === "number" ? payload.denoise_steps : null;
    const inferenceMode = String(payload.inference_mode || "unknown").slice(0, 64);

    // Insert into D1. PII-safe — no IP, no customer name, no payload.
    if (env.DB) {
      try {
        await env.DB.prepare(
          `INSERT INTO heartbeats (license_id, org_hash, vla_family, hardware_tier, reflex_version, client_timestamp, server_timestamp, model_name, hardware_detail, latency_p50, latency_p95, latency_p99, error_count_24h, safety_violation_count_24h, episode_count_24h, action_dim, embodiment, denoise_steps, inference_mode, tier)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
          .bind(
            payload.license_id,
            payload.org_hash,
            vlaFamily,
            hardwareTier,
            payload.tether_version,  // D1 column stays `reflex_version` (continuity)
            payload.timestamp,
            new Date().toISOString(),
            modelName,
            hardwareDetail,
            latencyP50,
            latencyP95,
            latencyP99,
            errorCount24h,
            safetyViolationCount24h,
            episodeCount24h,
            actionDim,
            embodiment,
            denoiseSteps,
            inferenceMode,
            tier
          )
          .run();
      } catch (e) {
        // Storage failure shouldn't kill the client; return 204 anyway.
        // Log to Cloudflare Workers metrics for ops visibility.
        console.error("D1 insert failed:", e.message);
      }
    }

    return new Response(null, { status: 204 });
  },
};
