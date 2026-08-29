# data-worker upload retirement

The legacy public upload routes are permanently retired:

- `POST /v1/episodes/upload` returns `410 Gone`.
- `POST /v1/episodes/upload-url` returns `410 Gone`.
- `GET /v1/contributor/:legacy_hash/stats` returns `410 Gone`; owner stats now
  require a signed Contributor Authentication v1 request to contribution-worker.
- `GET /v1/stats` returns `410 Gone`; contribution-worker is the sole stats
  authority.

Both responses direct clients to Contributor Authentication v1 on
`contribution-worker` (`/v1/contributors/register` then `/v1/uploads/sign`).
The retired handlers do not read request bodies, write R2 objects, or mutate D1.
The Wrangler configuration intentionally exposes no D1/R2 bindings.

Do not deploy this retirement until Python upload clients have migrated and the
contribution-worker D1 authentication/reservation migration has been applied.
Repository code alone is not evidence that either Worker is deployed.
