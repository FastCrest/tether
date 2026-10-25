# Tether contribution recovery

The separate recovery Worker protects the six contribution D1 tables (including
contributor keys and replay nonces) and the complete source R2 inventory. The
contribution application still uses legacy Reflex resource names. Application
Workers have no backup-bucket binding. Recovery never writes source rows, removes
source objects, advances revocations or activates the combined legacy purge cron.
The existing safe-maintenance service call is separate from capture.

## Durable capture and observable completion

Version 3 replaces the synchronous 200-object/512 MiB/10,000-row capture with one
SQLite Durable Object, `RecoveryCapture`. Its durable alarms advance bounded work
without another cron slot, queue, paid-plan upgrade or external polling service.
The existing daily 10:15 UTC cron starts a job. The AccSync backup is independently
attempted by that coordinator; its application/provider configuration is unchanged.
The job first lets safe maintenance settle, even when maintenance reports failure,
then captures media before opening the table capture/verification window.

Each step checkpoints in Durable Object storage and schedules its next alarm.
Interrupted steps reuse deterministic immutable chunk/shard keys, rechecking
source versions and existing backup hashes; a changed retry input fails closed.
Transient errors retry with bounded backoff, at most eight consecutive failures.
Observed source version/inventory/table/schema changes fail the job. A new
`POST /backup` after a failed job starts a new immutable prefix. Partial data is
never described as a complete snapshot and expires with the normal lifecycle.

All public endpoints require the separate `RECOVERY_TOKEN` bearer credential:

- `POST /backup`: start or return the active Tether job. Returns HTTP 202 while
  pending. It does not run AccSync or maintenance.
- `POST /run-backups`: initiate maintenance plus Tether capture, and independently
  run AccSync. HTTP 202 acknowledges an active capture; it is not a backup success.
- `POST /advance`: advance one bounded step of the active job, or return its final
  receipt. This is an operator recovery/test action; alarms normally do the work.
- `GET /status`: latest outcome, preserved last success/failure and scheduled
  success/attempt for each job; active phase, counts, copied bytes, retries and
  finite capacity limits. Freshness derives only from a completed backup, never
  from an accepted request or partial job. Manual runs preserve scheduled proof.
- `GET /snapshot?key=...`: retrieve an authenticated complete root manifest.
- `/index?key=...&index=N`, `/shard?key=...&index=N&entry=M` and
  `/chunk?key=...&index=N&entry=M&chunk=C`: download only resources referenced by
  that root/index/object chain. Arbitrary R2 keys are not accepted.
- Legacy `/object?key=...&index=N` remains available for version 2 snapshots.

A scheduled kickoff returning successfully proves only accepted dispatch and the
independent AccSync attempt. Successful durable completion is recorded later with
its original `scheduled` origin. External monitoring must inspect progress,
last failures and last completed/scheduled success, and deliver alerts separately.
The endpoint itself sends no notifications. No calendar-delivery claim follows
from configuration or local tests.

## Format, consistency and finite budgets

A v3 root manifest contains schema, row/object totals and checksummed index
references. Index pages reference separate table pages and per-object manifests.
Media is copied in backpressured 16 MiB range chunks, never whole-object buffers.
Each range is pinned to its source ETag and version, hashed while copied, and
hashed again from backup storage. Individual object manifests retain source key,
version, size, metadata and ordered chunk hashes. Their parent index and root
checksums bind the complete archive.

Inventory pages hold at most 50 entries. D1 uses rowid keyset paging, initially
250 rows, reducing the page size for large rows/results. A second complete table
pass compares page hashes; a second full paginated inventory compares identities;
a final schema comparison precedes publication. Empty end pages detect growth.
All checks must pass before `snapshot.json` is conditionally published and read
back. Root/index creation is idempotent across interrupted final publication.

This is **paginated double-pass consistency, not an atomic cross-service or
point-in-time snapshot**. It detects changes observed by its copy/comparison
passes. Mutations after a page's final comparison, independent administrative
writes and a change followed by a reversion between observations are outside that
guarantee. No source-wide write fence or durable transaction is claimed. Restores
also verify SQLite integrity, foreign keys, row fingerprints and every media chunk.
Sustained write traffic may repeatedly prevent a complete archive; monitoring
must treat that as a recovery gap, not a successful backup.

Operational limits remain explicit: six-hour job deadline; 64 GiB per source
object; 10,000 total table/object manifest shards (each object consumes one, as
does each table page, including its empty end page); at most 9,988 inventory
objects before row-page accounting; 4 MiB per manifest shard; 64 KiB schema;
SQLite safe-integer rowid cursors. A very large individual D1 row or unsupported
schema/rowid requires operator action. There is no 200-object, 512 MiB-total or
10,000-rows/table memory cutoff. These finite budgets are not a promise that all
combinations finish within the deadline or provider CPU/storage quotas.

Local tests stream and verify 211 objects totaling 663,748,608 bytes (633 MiB)
and 10,051 table rows, plus a separate 129 MiB object spanning nine chunks.
The large mock bucket regenerates validated uniform bodies rather than retaining
them in memory. This is a local algorithm/streaming proof; native Cloudflare
execution, capacity, elapsed time and paid usage require separate operator receipts.
Ranged copying, full readback, immutable daily copies and progress writes consume
R2 operations/storage and Durable Object operations/duration. Existing Free quotas
still apply; exceeding usage allowances can have billing/capacity consequences.

## Retention and deployment

`tether-contributions-backups` remains private. Every v3 index/table/object/media
copy lives below the same unique `snapshots/<run>/` prefix as its root, covered by
the existing seven-day age lock and expiry after eight days. No shared-media
cleanup or reference counting is needed. Media is created before root publication;
a job can take up to six hours, so its usable retention is measured from its
earliest pieces, not eight full days after root publication. The seven-day lock
is preserved. Source objects and purge cron settings are unchanged.

Deployment adds `CAPTURE=RecoveryCapture` and the `recovery-capture-v3` SQLite DO
migration from `wrangler.toml`. It requires no D1 data migration, new cron or source
application binding. Preserve `SOURCE`, `BACKUPS`, `DB`, `ACCSYNC_BACKUP` and
`CONTRIBUTIONS` bindings and all current secrets. `MAINTENANCE_TRIGGER_TOKEN`
matches contribution `MAINTENANCE_TOKEN`; `ACCSYNC_BACKUP_TRIGGER_TOKEN` remains
separate. Do not give application Workers recovery credentials or backup access.
The DO stores checkpoint metadata and the latest job receipt; source rows remain
only in retention-controlled snapshot shards, not DO state. Previous DO checkpoint
metadata is small but currently retained; archive data follows R2 expiry.

Same-account administrator/account compromise remains outside application-bound
deletion protection. Preserve the 1,732 unverified legacy revocations in quarantine;
do not infer valid consent from their historical customer labels.

## Private download and local restore

Use only a completed publication's manifest key and SHA256. Set `RECOVERY_TOKEN`
in the operator environment without printing it. Keep all tools together: the
Python entry points import `format_v3.py` from this directory. The downloader
accepts an HTTPS origin, refuses redirects, identifies the operator in User-Agent,
and verifies the root, every index/shard and each streamed media chunk:

```sh
python3 download.py https://BACKUP_WORKER_ORIGIN SNAPSHOT_KEY PUBLICATION_SHA256 /private/new-download
python3 restore.py /private/new-download/snapshot.json PUBLICATION_SHA256 /private/new-restore
```

Both destinations must be new. Files use private permissions. V3 restoration
loads one checksummed table shard at a time into new local SQLite files and
rehydrates media from ordered, independently verified chunks. Source keys never
become filesystem paths. It validates table counts, fingerprints, SQLite
integrity/foreign keys and object totals; no production connection or revocation
execution occurs. Preserve partial local output for inspection on failure.

Version 1 embedded-body and version 2 separate-media snapshots remain supported,
including their original local layout and optional v2 `MEDIA_DIRECTORY`. V3 uses
its generated `indexes/`, `shards/` and `media/` paths beside the root manifest.
It records full reconstructed object SHA256 in `object-map.json`; server-side v3
verification pins each chunk rather than claiming a single whole-object digest.
