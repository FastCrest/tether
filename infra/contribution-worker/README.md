# reflex-contribution-worker

Cloudflare Worker for the Curate wedge — handles upload signing, contributor
stats, and revoke-cascade requests for the data contribution program.

Sibling workers (each in its own folder under `infra/`):
- `license-worker` — Pro license issuance + revocation (deployed)
- `telemetry-worker` — opt-in telemetry (built, not yet deployed)
- `contribution-worker` — this one (deployed at `https://reflex-contributions.fastcrest.workers.dev`)

## Endpoints

### Health
- `GET /healthz` → `{ status: "ok" }`

### Admin (`Authorization: Bearer <ADMIN_TOKEN>`)
- `POST /admin/init-bucket` — sanity-check the R2 binding
- `GET  /admin/contributors` — list contributors + stats
- `POST /admin/manual-purge` — trigger cascade for a specific contributor
- `POST /admin/cascade-execute/:request_id` — force cascade progression
- `POST /v1/revoke/cascade` — mark a contributor for purge (temporary admin-only containment)

### Contributor Authentication v1
- `POST /v1/contributors/register` — prove possession and enroll an Ed25519 public key
- `POST /v1/contributors/rotate` — atomically rotate from the current active key
- `POST /v1/uploads/sign` — create an authenticated upload reservation
- `PUT  /v1/uploads/put/:upload_id` — consume its one-time upload capability
- `POST /v1/uploads/complete` — authenticated, owner-only completion
- `GET  /v1/revoke/cascade-status/:request_id` — owner/admin-only state read; never advances it
- `GET  /v1/contributors/:id/stats` — authenticated, owner-only totals

Timed revoke stages advance through the private Cloudflare cron trigger every
five minutes. Public `GET` requests never perform a tombstone or R2 deletion.

## Storage

- **D1** binding `DB` (`reflex-contributions`): contributors, contributor_keys,
  contributor_nonces, uploads, daily_uploads, and revoke_requests. Schema in
  `schema.sql`.
- **R2** binding `CURATE_BUCKET` (`reflex-curate`): contribution payloads
  under `<tier>-contributors/<contributor_id>/<utc_date>/<file_name>`.

## Rate limits

Phase 1 defaults (configurable via env vars):
- `DAILY_BYTES_LIMIT`: 10 GB / contributor / UTC day
- `DAILY_UPLOADS_LIMIT`: 1000 uploads / contributor / UTC day
- 60 signed reservations per contributor in a rolling hour
- 100 MiB maximum object size
- 10-minute reservation/capability lifetime

Cloudflare's built-in DDoS protection applies on the public endpoints.

## Authentication and upload protocol

Every contributor request is signed over an RFC 8785 envelope containing the
exact uppercase method, exact URL pathname, sorted duplicate-preserving query,
exact request-body SHA-256, Unix timestamp, and 16-byte nonce. Header values
must match that envelope. The Worker permits five minutes of clock skew and
atomically rejects nonce reuse for ten minutes.

Registration supplies `X-Tether-Public-Key`; contributor and key IDs are
derived from the raw public-key digest and the same request signature proves
possession. New contributors are always assigned server tier `free`. A client
body cannot choose an identity, quota owner, or tier. Pro/Enterprise assignment
remains unavailable until a signed v2 license binding is verified server-side.
Private keys remain local, owner-readable only, and must never be stored in a
consent receipt, log, or request.

The Python clients create `~/.tether/contributor-auth-v1.json` with mode 0600
on first use. `TETHER_CONTRIBUTOR_CREDENTIALS` may point to another owner-only
credential file for test or managed-device environments; it must not point to
a checked-in file or a shared secret store export.

Each queued payload also has a hidden mode-0600 upload-state sidecar. It stores
the validated upload ID, capability, expiry, and reconciliation phase so a
process restart signed-completes the existing reservation before considering a
new one. Expired capabilities are redacted while the upload ID is retained for
safe signed status/completion reconciliation. A permanent owner-only advisory
lock inode elects one process across load, reservation, PUT, completion, and
state publication; the operating system releases the lock on process death and
the inode is never stale-deleted. Confirmed completion leaves a capability-
redacted marker bound to that source file's filesystem generation, so an
already-waiting worker reuses the result while a later replacement file safely
supersedes it under the same lock. Server-confirmed unclaimed expiry removes
the old reservation before a replacement is issued.

`/v1/uploads/sign` accepts a canonical JSON body with `file_name`, exact
`byte_size`, and a signed anonymization `manifest`. It returns a random 32-byte
capability once; D1 stores only its SHA-256. PUT atomically claims
`pending -> uploading`, writes only to an attempt-specific R2 key, verifies the
exact length and content digest, and conditionally finalizes `uploaded`.
Completion is separately contributor-signed, owner-bound, and validates the
stored R2 size/digest metadata before `uploaded -> completed`.

Both checked-in Python uploaders keep the returned reservation through bounded
transient retries. A lost PUT response is reconciled with signed completion;
only an explicitly still-pending reservation may reuse its same capability.
A lost completion response retries that idempotent completion with exponential
backoff. Neither path blindly creates a second reservation or object.
Before the first `/sign`, clients persist a random 32-byte reservation key in
the owner-only sidecar and sign it as part of the canonical body. The Worker
enforces one `(contributor_id, reservation_key)` row and derives the capability
with a dedicated HMAC secret, so a lost sign response or process crash replays
the same still-usable reservation without consuming quota twice.
The sidecar also persists the exact canonical `/sign` body, including its scan
timestamp. Restarts resend those identical bytes even after wall-clock change.
The Worker stores that signed-body SHA-256 and performs the owner/key lookup
before freshness or quota evaluation, so replay after PUT/completion or quota
saturation returns the original reservation; a changed body with the same key
fails closed.
Every replay return, including a concurrent `INSERT OR IGNORE` loser, compares
the fixed-width request digest in constant time before capability derivation.
Quota admission happens only in the conditional insert; a loser always rereads
the idempotency row before any quota response. If an old request was never
committed, the Worker's definitive `stale_manifest` lets the client atomically
persist one fresh key/timestamp/body under the same source lock and retry. A
committed stale request is found first and continues to replay the old body.

Curate rotates an active daily JSONL to a stable upload snapshot while holding
the collector's pathname lock; later appends create/remain in the original
queue path. Pro uses one episode-manifest lock independent of whether the
queued generation is JSONL or Parquet. Under that lock it reads the manifest
first; the persisted file name, byte size, and SHA-256 select the only eligible
data generation. Hash and size are rechecked before upload and terminal move.
A replacement atomically publishes its manifest and removes the superseded
format and reservation sidecar, so JSONL↔Parquet replacement cannot upload an
older sibling or strand the newer generation.

Revocation fences every contributor upload to `purged` before R2 enumeration.
The internal purge-only path accepts historical non-empty customer IDs such as
email addresses, while rejecting separators, traversal and control characters.
That compatibility grammar is never used for reservation or payload writes,
which remain restricted to cryptographic `ctr_*` principals.
Payload writes and terminal purge pass through one SQLite-backed Durable Object
per contributor. If an admitted write arrives first, purge runs after it and
deletes it; if purge arrives first, its persistent fence rejects the later
write before R2. No wall-clock lease is used to infer that a live request has
ended. Scheduled recovery only selects `uploading` rows and cannot revive a
fenced upload.

The anonymization manifest is evidence/accountability that a named scanner and
anonymizer ran. It is not semantic proof that all PII is absent.

Destructive revoke operations are excluded from this soft-auth model. As an
emergency containment measure, they require `Authorization: Bearer
<ADMIN_TOKEN>` and use a fixed-length digest comparison. A contributor-facing
revoke flow must not be re-enabled until it binds the requested contributor ID
to proof of possession.

## Deploying (when ready)

The recommended path is the checked-in deployment script. It distinguishes a
fresh D1 database from an existing one, applies tracked additive migrations and
verifies every required column before deploying Worker code, and is safe to
rerun. A partially applied migration fails closed for operator repair.

```bash
./deploy.sh
```

Manual database preparation, if the script cannot be used:

```bash
# 1. Create the D1 database
wrangler d1 create reflex-contributions
# Copy the resulting database_id into wrangler.toml

# 2. Create the R2 bucket
wrangler r2 bucket create reflex-curate

# 3a. Fresh/empty database only: apply the full schema
wrangler d1 execute reflex-contributions --file=./schema.sql --remote

# 3b. Existing legacy database instead: take a backup, apply each missing
# tracked migration exactly once, and verify the columns before code deploy.
wrangler d1 execute reflex-contributions \
  --file=./migrations/2026-05-05-revoke-cascade-stages.sql --remote
wrangler d1 execute reflex-contributions \
  --file=./migrations/2026-08-28-contributor-auth-v1.sql --remote

# 4. Set secrets
wrangler secret put ADMIN_TOKEN          # 32+ hex chars
wrangler secret put SLACK_WEBHOOK_URL    # optional

# 5. Deploy
wrangler deploy

# 6. Smoke test
curl https://reflex-contributions.fastcrest.workers.dev/healthz

# 7. Bind a custom domain (optional)
# Edit wrangler.toml routes section, then redeploy.
```

Deployment sequencing:
- Run `deploy.sh` so D1 is migrated and verified before contribution-worker
  code, then deploy the data-worker 410
  retirement. Both checked-in Python upload clients now use Contributor
  Authentication v1, and the Worker tests and cross-language vectors validate
  the protocol, but this repository slice does not authorize a production rollout.
- Never apply `schema.sql` over an existing database: its `CREATE TABLE IF NOT
  EXISTS` statements cannot add missing columns. Legacy pending reservations
  are expired by the tracked contributor-auth migration and cannot be claimed.
- Deployment refuses partial authentication schemas and verifies the key and
  nonce tables, their required columns, the active-key uniqueness/status
  constraint, and reservation/recovery indexes before publishing Worker code.
  Verification compares normalized complete table/index definitions, including
  primary keys, uniqueness, foreign keys, NOT NULL/CHECK constraints, indexed
  columns, uniqueness, and partial-index predicates; matching names alone are
  insufficient.
  Canonicalization removes whitespace adjacent to SQL punctuation, and the
  checked-in deployment regression loads the real `schema.sql` into SQLite and
  verifies its `sqlite_master`, PRAGMA column/PK/FK, and index metadata.
- Do not wire the contributor CLI to `/v1/revoke/cascade` while the emergency
  admin-only containment is active. A public revoke flow requires the signed
  proof-of-possession protocol first.

## Layout convention (per ADR decision #1)

R2 bucket layout (single bucket, prefix-segmented):

```
reflex-curate/
├── free-contributors/<contributor_id>/<YYYY-MM-DD>/<session>.jsonl
├── pro-contributors/<customer_id>/<YYYY-MM-DD>/<session>.jsonl
├── enterprise-contributors/<customer_id>/<YYYY-MM-DD>/<session>.jsonl
└── derived-datasets/v<n>/<dataset_slug>/...
```

The worker enforces the prefix when issuing reservations. A Free contributor
can never PUT to `pro-contributors/...` because the key is constructed from the
server-owned tier attached to the authenticated principal.

## Notes

- Fresh schema and existing-database migration are separate because SQLite/D1
  has no portable `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.
- Reservation rows, not caller-provided counters, are authoritative for quota.
- Revoke cascade progression runs through the Worker's private scheduled hook.
  The public status route is read-only; an administrator can force progression
  through the authenticated `/admin/cascade-execute/:request_id` route.
