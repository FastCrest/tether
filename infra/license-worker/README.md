# Reflex Pro license server

Cloudflare Worker that signs licenses, serves activation codes, records heartbeats, checks revocations, and detects abuse for Reflex Pro customers.

Pairs with the Phase 1 telemetry worker at `infra/telemetry-worker/` (different concerns: telemetry is anonymous usage stats; this is per-customer license + revocation infrastructure).

## What it provides

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/healthz` | GET | none | Health probe |
| `/admin/init` | POST | bearer | One-time Ed25519 keypair generation |
| `/admin/issue` | POST | bearer | Sign + store new license, return activation code |
| `/admin/revoke` | POST | bearer | Revoke a license_id |
| `/admin/list` | GET | bearer | List all licenses with status + heartbeat info |
| `/v1/pubkey` | GET | none | Return current Ed25519 public key (for offline verify) |
| `/v1/activation/:code` | GET | none | Fetch signed license by one-time code (24h TTL, single-use) |
| `/v1/heartbeat` | POST | none | Record heartbeat + check revocation/expiry |
| `/v1/revocation/:license_id` | GET | none | Check if a license is revoked |

## Deploy (one-time setup, ~25 min)

```bash
# 1. Install wrangler if you don't have it
npm install -g wrangler

# 2. Authenticate (browser OAuth)
wrangler login

# 3. Create the D1 database, copy the resulting database_id
wrangler d1 create reflex-licenses
# → paste the database_id into wrangler.toml

# 4. Apply the schema
cd infra/license-worker
wrangler d1 execute reflex-licenses --file=schema.sql

# 5. Set the admin token (generate one with `openssl rand -base64 32`)
wrangler secret put ADMIN_TOKEN
# (paste your generated token at the prompt)

# 6. (Optional) Set Slack webhook URL for new-license / revoke / sharing alerts
wrangler secret put SLACK_WEBHOOK_URL

# 7. Deploy the worker
wrangler deploy
# → note the URL (e.g. https://tether-licenses.<subdomain>.workers.dev)

# 8. Generate the Ed25519 keypair (one-time)
curl -X POST https://tether-licenses.<subdomain>.workers.dev/admin/init \
    -H "Authorization: Bearer <YOUR_ADMIN_TOKEN>"
# → response includes public_key_b64 + private_key_b64
# IMMEDIATELY:
#   a. Copy the private_key_b64 and set it as a Worker Secret:
#        echo '<private_key_b64>' | wrangler secret put PRIVATE_KEY
#   b. Copy the public_key_b64 into src/tether/pro/_public_key.py
#      (replace the BUNDLED_PUBLIC_KEY_B64 constant)
#   c. Discard the private_key_b64 from your terminal scrollback (it never
#      needs to leave wrangler again)
#
# 9. Update src/tether/pro/license.py:DEFAULT_LICENSE_ENDPOINT to your worker URL
# 10. Commit + push the public-key + endpoint changes

# Test the live worker
curl https://tether-licenses.<subdomain>.workers.dev/healthz
curl https://tether-licenses.<subdomain>.workers.dev/v1/pubkey
```

## Issue your first license

```bash
# From your laptop (admin CLI talks to the worker)
export TETHER_LICENSE_ENDPOINT=https://tether-licenses.<subdomain>.workers.dev
export REFLEX_ADMIN_TOKEN=<your_admin_token>

python -m tether.admin.issue_license \
    --customer-id alice@bigco.com \
    --tier pro \
    --expires-in 30 \
    --notes "First customer"

# → outputs: License: lic_xxx
#            Activation code: REFLEX-XXXX-XXXX-XXXX (expires in 24h)
#
# Send the activation code to the customer however you talk to them
# (Discord, DM, email, whatever — no email service required at Reflex's end).
```

## Customer redeems

```bash
# On the customer's machine
pip install --upgrade fastcrest-tether
tether pro activate REFLEX-XXXX-XXXX-XXXX
# ✓ License fetched, signature verified, written to ~/.reflex/pro.license
# ✓ Hardware bound

tether serve --pro <export_dir>  # works
```

## Revoke a license

```bash
python -m tether.admin.revoke_license \
    --license-id lic_xxx \
    --reason "Refund processed"
```

Customer's running deployment will fail its next heartbeat (within 24h) and refuse to serve.

## Privacy posture

- We log Cf-Connecting-IP at the Cloudflare edge but the worker does NOT write it to D1. Only the country code (Cf-IPCountry) is stored.
- The `hardware_fingerprint` field stored in heartbeats is a customer-computed hash (gpu_uuid + cpu_count + similar), not a raw machine identifier.
- Customer payloads (`/act` requests, model inputs, robot state) NEVER touch this worker.

## Common queries

```sql
-- Active licenses + last heartbeat
SELECT license_id, customer_id, tier, expires_at,
       (SELECT MAX(server_timestamp) FROM heartbeats WHERE heartbeats.license_id = l.license_id) AS last_heartbeat
FROM licenses l
WHERE revoked_at IS NULL
ORDER BY issued_at DESC;

-- Licenses with sharing signals in the last 30 days
SELECT s.license_id, s.signal_type, s.details, s.detected_at
FROM abuse_signals s
WHERE s.detected_at > datetime('now', '-30 days')
ORDER BY s.detected_at DESC;

-- Licenses with expired heartbeats (active license, no heartbeat in 7d)
SELECT l.license_id, l.customer_id, l.expires_at,
       (SELECT MAX(server_timestamp) FROM heartbeats WHERE heartbeats.license_id = l.license_id) AS last_heartbeat
FROM licenses l
WHERE l.revoked_at IS NULL
  AND datetime(l.expires_at) > datetime('now')
  AND (
      (SELECT MAX(server_timestamp) FROM heartbeats WHERE heartbeats.license_id = l.license_id) IS NULL
      OR (SELECT MAX(server_timestamp) FROM heartbeats WHERE heartbeats.license_id = l.license_id) < datetime('now', '-7 days')
  );
```

## Key rotation (Phase 2 — not implemented yet)

The schema supports key rotation via the `master_keys.retired_at` column, but the rotation endpoint (`POST /admin/rotate`) isn't built yet. When you need it:

1. Generate a new Ed25519 keypair (new POST /admin/init variant)
2. New licenses get signed with the new key
3. Old key stays valid for verification (grace period)
4. Customer-side bundled key gets a list of N trusted keys instead of one
5. Eventually retire the old key when no licenses signed with it remain

Plan to revisit when you have ~50 active licenses or a security incident requires rotation.
