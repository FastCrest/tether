# Tether contribution recovery

The contribution application still uses legacy Reflex resource names. This
separate Worker backs up its six D1 tables, including contributor keys and replay nonces, and complete R2 object inventory.
Backup capture never advances revocations or writes to the source bucket or database. The separate authenticated maintenance job runs only safe stale-upload/nonce housekeeping; it never executes the combined legacy purge scheduler.

The daily 10:15 UTC coordinator first waits for safe maintenance to settle, then independently runs this backup and the AccSync
voice-backup Worker through a service binding. One failed job does not prevent
the other from running. This coordinator uses one new cron slot; AccSync shares that trigger. No plan upgrade is required.
`POST /run-backups`, `POST /backup`, `GET /status`, `GET /snapshot?key=...` and manifest-scoped `GET /object?key=...&index=N` require the
separate `RECOVERY_TOKEN`. The coordinator uses a separate
`ACCSYNC_BACKUP_TRIGGER_TOKEN` for the AccSync service. The `CONTRIBUTIONS` service binding uses `MAINTENANCE_TRIGGER_TOKEN`, matching the contribution Worker's `MAINTENANCE_TOKEN`, for `POST /internal/maintenance`. Missing configuration is an independently recorded failure; other jobs still run.

`tether-contributions-backups` is private. Its `snapshots/` prefix has a seven-day
age lock and expiry after eight days. These rules apply only to backup data;
the source bucket and contribution purge schedules are unchanged at runtime.
Do not extend backup retention past consent obligations without reviewing them.
The application has no binding to the backup bucket.

Version 2 snapshots contain schemas, all rows and a manifest of separately streamed
R2 object copies. Every copy lives below the same unique `snapshots/<run>/` prefix
as its manifest, so the seven-day lock/eight-day expiry applies to media too.
No shared media or reference-counted cleanup is needed. Objects carry their source
version, ETag, size and SHA256. Each source body passes through one backpressured SHA256/copy pipeline, then the backup is streamed back through SHA256 before publication.
The Worker never buffers a full media object. A second inventory/table/schema read rejects
source mutation, including a new source version with the same content/ETag.
Only a complete, checksum-verified manifest is published; failed-run partial copies
expire through the same eight-day lifecycle and are not restorable publications.

Capture is bounded to 200 objects, 128 MiB per object, 512 MiB of media per run,
10,000 rows per table and 8 MiB of serialized manifest. These are operational
budgets, not demonstrated production capacity. The focused local stream test
captures, downloads and restores 45 objects totaling 11,059,200 bytes. Larger
workloads require a measured, resumable job design before these limits are lifted.
The 8 MiB manifest/row budget still uses memory; object bodies do not.
The Worker uses native `crypto.DigestStream` and `FixedLengthStream`.

`GET /status` returns each job's latest attempt, last successful result and last
failure, origin (`manual` or `scheduled`), timestamps, age and `fresh` (success
within 26 hours). Mutable status lives at `status/latest.json` outside locked
snapshots; it contains no credentials or contributor rows. Conditional writes
prevent simultaneous jobs from losing each other's status. Capture and maintenance
failures are reported separately. A later failure never erases the prior success.
The scheduled handler awaits every job, persists outcomes and throws on failure.
Status requires an external monitor to deliver alerts; the endpoint alone sends
nothing. A configured schedule is not proof of an actual scheduled dispatch.

October 24 review found 1,732 pending revocations without verified caller
identity, 1,732 contributors and two objects totaling 360 bytes. All rows and
both objects passed an isolated restore, including object rehydration after
local deletion. Preserve the queue for review; do not activate automatic purge
or infer legitimate consent from its `source=customer` note.

The backup is protected against application-bound object deletion; same-account
loss or compromise of account administrators remains outside that boundary.

## Local restore command

Use the manifest key and SHA256 from a successful publication receipt or authenticated
status. Download the manifest and its referenced media to a NEW private directory.
Set `RECOVERY_TOKEN` in the operator environment without printing it. The downloader
accepts only an HTTPS origin, refuses redirects and verifies every checksum:

```sh
python3 download.py https://BACKUP_WORKER_ORIGIN SNAPSHOT_KEY PUBLICATION_SHA256 /private/new-download
```

Then restore into another NEW directory:

```sh
python3 restore.py /private/new-download/snapshot.json PUBLICATION_SHA256 /private/new-restore
```

The destination must not exist. The tool verifies the publication checksum before
creating files, uses private permissions, verifies the restored contents, and
never connects to production. AccSync audio verification requires ffmpeg on PATH.
Preserve any incomplete local output for inspection; do not start application
workers or replay provider operations from a restore.

The upload-auth rollout adds contributor_keys and contributor_nonces. New
snapshots cover all six tables; restore.py accepts both the earlier four-table
and the current six-table format. The live upload protocol is the existing
Contributor Authentication v1 implementation from main, deployed with its
tracked additive migration and upload-capability secret.

Version 1 embedded-media backups remain supported. Version 2 reads downloaded media
from `objects/000000`, `objects/000001`, etc. beside the manifest, or an explicit
optional MEDIA_DIRECTORY. Source object keys never become local filesystem paths.
Restore verifies SQLite integrity/foreign keys, row fingerprints and each object
size/hash, copying media in bounded chunks. It never connects to production.
