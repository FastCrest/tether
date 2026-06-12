# Tether Pro telemetry endpoint

Cloudflare Worker that receives anonymized heartbeat POSTs from `tether` deployments running under a Pro license. Stores one row per heartbeat in D1 for aggregate analysis.

## What gets received

Per-heartbeat JSON payload (locked Phase 1 schema, see `src/tether/pro/telemetry.py:HeartbeatPayload`):

```json
{
  "schema_version": 1,
  "license_id": "<JWT customer_id>",
  "org_hash": "<sha256(customer_id)[:16]>",
  "workload": {"vla_family": "pi05", "hardware_tier": "a100"},
  "reflex_version": "0.8.0",
  "timestamp": "2026-05-03T12:00:00.000000Z"
}
```

What's intentionally NOT sent:
- `/act` payloads (images, instructions, robot state)
- Robot trajectories or actions
- Model weights or embeddings
- Customer org name (only the SHA256 tag)
- IP addresses (Cloudflare logs `Cf-Connecting-IP` at the edge but the worker does not write it to D1)

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| `POST` | `/v1/heartbeat` | Accepts a heartbeat payload, returns 204 on success |
| `GET`  | `/healthz`     | Returns `{"status": "ok", "schema": 1}` |
| any    | other          | 404 / 405 |

## Deploy

```bash
cd infra/telemetry-worker

# One-time setup
npm install -g wrangler
wrangler login
wrangler d1 create reflex-telemetry
# → copy the resulting `database_id` into wrangler.toml

# Apply the schema
wrangler d1 execute reflex-telemetry --file=schema.sql

# Deploy the worker
wrangler deploy

# (Optional) bind a custom domain
# Edit wrangler.toml, uncomment the `routes` block, redeploy
```

After deploy, the default URL is `https://tether-telemetry.<your-subdomain>.workers.dev`. The client default is `DEFAULT_TELEMETRY_ENDPOINT = https://tether-telemetry.fastcrest.workers.dev/v1/heartbeat` in `src/tether/pro/telemetry.py` — update it there if your deployed subdomain differs.

## Common queries

```sql
-- How many unique active deployments in the last 7 days?
SELECT COUNT(DISTINCT org_hash) FROM heartbeats
WHERE server_timestamp > datetime('now', '-7 days');

-- Bypass-population estimate: licenses heard from vs paying customers
-- (cross-reference license_id against your billing DB)
SELECT COUNT(DISTINCT license_id) FROM heartbeats
WHERE server_timestamp > datetime('now', '-1 days');

-- Workload distribution
SELECT vla_family, hardware_tier, COUNT(*) as heartbeats
FROM heartbeats
WHERE server_timestamp > datetime('now', '-7 days')
GROUP BY vla_family, hardware_tier
ORDER BY heartbeats DESC;

-- Version distribution (which Reflex versions are deployed where)
SELECT reflex_version, COUNT(DISTINCT org_hash) as deployments
FROM heartbeats
WHERE server_timestamp > datetime('now', '-7 days')
GROUP BY reflex_version
ORDER BY deployments DESC;
```

## Customer opt-out

Customers can disable telemetry via `TETHER_NO_TELEMETRY=1` env var. The Tether client respects this before any HTTP call is attempted — see `src/tether/pro/telemetry.py:_is_disabled`.

For the customer-facing disclosure that this exists, see `docs/self_distilling_serve.md` (the Pro tier doc).
