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
| `/admin/signer` | GET | bearer | Cryptographically verify `SIGNING_KEY_ID` matches `PRIVATE_KEY` |
| `/v1/pubkey` | GET | none | Return current Ed25519 public key (for offline verify) |
| `/v1/activation/:code` | POST | one-time code | Bind hardware and fetch signed v2 license (24h TTL, single-use) |
| `/v1/heartbeat` | POST | nonce-bound | Record heartbeat + return signed status attestation |
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
# → note the URL (e.g. https://reflex-licenses.<subdomain>.workers.dev)

# 8. Generate the Ed25519 keypair (one-time)
curl -X POST https://reflex-licenses.<subdomain>.workers.dev/admin/init \
    -H "Authorization: Bearer <YOUR_ADMIN_TOKEN>"
# → response includes public_key_b64 + private_key_b64
# IMMEDIATELY:
#   a. Copy the private_key_b64 and set it as a Worker Secret:
#        echo '<private_key_b64>' | wrangler secret put PRIVATE_KEY
#   b. Bind the private key to the exact D1 public-key row:
#        echo '<key_id>' | wrangler secret put SIGNING_KEY_ID
#   c. Add key_id → public_key_b64 to TRUSTED_PUBLIC_KEYS_B64
#   d. Discard the private_key_b64 from your terminal scrollback (it never
#      needs to leave wrangler again)
#
# 9. Update src/tether/pro/license.py:DEFAULT_LICENSE_ENDPOINT to your worker URL
# 10. Commit + push the public-key + endpoint changes

# Test the live worker
curl https://reflex-licenses.<subdomain>.workers.dev/healthz
curl https://reflex-licenses.<subdomain>.workers.dev/v1/pubkey
```

### Existing-install migration

Older installs may already have one active D1 public-key row and a matching
`PRIVATE_KEY`, but no `SIGNING_KEY_ID`. Migrate that install before deploying
the stricter Worker:

1. Query the active rows directly with `wrangler d1 execute reflex-licenses
   --remote --json --command "SELECT key_id, public_key_b64 FROM master_keys
   WHERE retired_at IS NULL"`.
2. Compare the row's public key with the key already shipped in
   `TRUSTED_PUBLIC_KEYS_B64`. If more than one row is active, do not guess: set
   `TETHER_SIGNING_KEY_ID` to the row known to match the existing private key.
3. Set that id with `wrangler secret put SIGNING_KEY_ID`, then deploy.
4. Call authenticated `GET /admin/signer`. A 200 response with
   `{ "verified": true, "key_id": "..." }` proves the configured private key
   matches the selected non-retired D1 row. Any mismatch fails closed.

`deploy.sh` performs these steps automatically for the normal one-active-key
case. It refuses an ambiguous multi-key migration or an active public row with
no `PRIVATE_KEY`; it never creates a replacement private key for an existing
public row. Set `REFLEX_ADMIN_TOKEN` before running it so the script can verify
the coupled signer immediately after deployment; it refuses to migrate an
existing signer without that verification credential.

## Issue your first license

```bash
# From your laptop (admin CLI talks to the worker)
export TETHER_LICENSE_ENDPOINT=https://reflex-licenses.<subdomain>.workers.dev
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

Customer's running deployment will fail its next heartbeat and refuse to serve.

## Signed heartbeat contract

The client sends a fresh unpadded-base64url 16-byte `request_nonce`. The Worker
returns an Ed25519-signed `tether.license.heartbeat` v1 attestation containing
the echoed nonce, `license_id`, active signing `key_id`, Unix `issued_at`,
`valid_until`, and status (`active`, `expired`, or `revoked`; clients also
understand `suspended`). Active validity is capped at 24 hours and at the
license expiry. Released clients accept five minutes of clock skew, persist
only verified attestations, and fail paid requests closed when the signed
deadline plus skew elapses.

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

## Key rotation

The schema supports key rotation via the `master_keys.retired_at` column, but the rotation endpoint (`POST /admin/rotate`) isn't built yet. When you need it:

1. Generate the new Ed25519 pair through an audited rotation procedure and
   insert its public half as a new non-retired `master_keys` row. Keep the old
   row active.
2. Ship both old and new public keys in `TRUSTED_PUBLIC_KEYS_B64` and release
   that client trust overlap before changing the signer.
3. Set `PRIVATE_KEY` and `SIGNING_KEY_ID` to the new coupled pair in one
   controlled deployment window, then require authenticated `/admin/signer`
   to return the new id with `verified: true`. Roll back both secrets together
   if verification fails.
4. Keep the old public key trusted throughout the maximum license/attestation
   overlap.
5. Retire the old D1 row and remove its client trust only after that overlap
   has elapsed.

Plan to revisit when you have ~50 active licenses or a security incident requires rotation.
