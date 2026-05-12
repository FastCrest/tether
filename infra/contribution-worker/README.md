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

### Customer (Phase 1 soft-auth: `contributor_id` + `opted_in_at` body fields)
- `POST /v1/uploads/sign` — issue signed PUT URL for an upload
- `POST /v1/uploads/complete` — record successful upload, increment stats
- `POST /v1/revoke/cascade` — mark contributor for purge (30-day SLA)
- `GET  /v1/contributors/:id/stats` — return contribution totals

## Storage

- **D1** binding `DB` (`reflex-contributions`): contributors, uploads,
  daily_uploads, revoke_requests. Schema in `schema.sql`.
- **R2** binding `CURATE_BUCKET` (`reflex-curate`): contribution payloads
  under `<tier>-contributors/<contributor_id>/<utc_date>/<file_name>`.

## Rate limits

Phase 1 defaults (configurable via env vars):
- `DAILY_BYTES_LIMIT`: 10 GB / contributor / UTC day
- `DAILY_UPLOADS_LIMIT`: 1000 uploads / contributor / UTC day

Cloudflare's built-in DDoS protection applies on the public endpoints.

## Auth posture (Phase 1 → 1.5)

**Phase 1 (current):** customer endpoints accept `contributor_id` + `opted_in_at`
as body fields. The worker trusts the client. Same trust model as the
on-disk consent receipt. **Anyone with a valid `contributor_id` could upload as
that contributor** — privacy-acceptable because contributor_ids are anonymous
and uploads are subject to the same anonymization filters that ship in
`reflex.curate.uploader`.

**Phase 1.5:** add Ed25519 challenge-response. The user-side keypair lives in
the consent receipt (`consent_signature`); the worker challenges the client
with a random nonce on each /v1/uploads/sign call and refuses unless the
client signs it with the receipt's private key.

## Deploying (when ready)

```bash
# 1. Create the D1 database
wrangler d1 create reflex-contributions
# Copy the resulting database_id into wrangler.toml

# 2. Create the R2 bucket
wrangler r2 bucket create reflex-curate

# 3. Apply the schema
wrangler d1 execute reflex-contributions --file=./schema.sql --remote

# 4. Set secrets
wrangler secret put ADMIN_TOKEN          # 32+ hex chars
wrangler secret put SLACK_WEBHOOK_URL    # optional

# 5. Deploy
wrangler deploy

# 6. Smoke test
curl https://reflex-contributions.<account>.workers.dev/healthz

# 7. Bind a custom domain (optional)
# Edit wrangler.toml routes section, then redeploy.
```

After deploy:
- Update `src/reflex/curate/uploader.py:_request_signed_url` and `_put_to_r2`
  to make real httpx calls (currently raise `UploadStub`).
- Flip `live=True` in `src/reflex/runtime/server.py` curate-uploader scaffolding.
- Update `src/reflex/curate/opt_in_cli.py:_cmd_revoke` to actually POST to
  `/v1/revoke/cascade` instead of logging.

## Layout convention (per ADR decision #1)

R2 bucket layout (single bucket, prefix-segmented):

```
reflex-curate/
├── free-contributors/<contributor_id>/<YYYY-MM-DD>/<session>.jsonl
├── pro-contributors/<customer_id>/<YYYY-MM-DD>/<session>.jsonl
├── enterprise-contributors/<customer_id>/<YYYY-MM-DD>/<session>.jsonl
└── derived-datasets/v<n>/<dataset_slug>/...
```

The worker enforces the prefix when issuing signed URLs — a Free contributor
can never PUT to `pro-contributors/...` because the worker constructs the
key from the request body's `tier` field.

## Notes

- Schema designed additive-only: every new field uses `IF NOT EXISTS` and
  defaults so older rows still satisfy NOT NULL constraints.
- `daily_uploads` is the hot table on the upload path. Indexed on
  `(contributor_id, utc_date)` (PK) + `utc_date` for retention prune jobs.
- Revoke cascade runs OUT OF BAND from this worker — the worker only
  enqueues the request and flags the contributor. The actual R2 deletion
  + derived-dataset rebuild + buyer notifications are operator-driven jobs
  triggered by the Slack alert. Phase 2 automates the cascade.
