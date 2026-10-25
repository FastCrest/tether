# Tether contribution recovery

The contribution application still uses legacy Reflex resource names. This
separate Worker backs up its six D1 tables, including contributor keys and replay nonces, and complete R2 object inventory.
It never advances revocations or writes to the source bucket or database.

The daily 10:15 UTC coordinator independently runs this backup and the AccSync
voice-backup Worker through a service binding. One failed job does not prevent
the other from running. This coordinator uses one new cron slot; AccSync shares that trigger. No plan upgrade is required.
`POST /run-backups`, `POST /backup` and `GET /snapshot?key=...` require the
separate `RECOVERY_TOKEN`. The coordinator uses a separate
`ACCSYNC_BACKUP_TRIGGER_TOKEN` for the AccSync service.

`tether-contributions-backups` is private. Its `snapshots/` prefix has a seven-day
age lock and expiry after eight days. These rules apply only to backup data;
the source bucket and contribution purge schedules are unchanged at runtime.
Do not extend backup retention past consent obligations without reviewing them.
The application has no binding to the backup bucket.

Each complete JSON snapshot contains schemas, all rows and base64 object bodies
with SHA256 checksums. A second inventory/table read rejects concurrent mutation.
Publication uses a unique key, conditional creation and full hash readback.
Restore into a new local SQLite database and directory, verify row fingerprints,
SQLite integrity/foreign keys and every object hash. Keep snapshots private.
Never execute restored revocations or start workers from a restored copy.

Capture fails above 40 objects, 4 MiB of object bodies, 10,000 rows per table or
8 MiB of serialized snapshot. These limits cover the measured workload only.
Inspect Cloudflare scheduled-event failures and the latest complete manifest;
a configured schedule is not proof that a calendar-triggered run succeeded.

October 24 review found 1,732 pending revocations without verified caller
identity, 1,732 contributors and two objects totaling 360 bytes. All rows and
both objects passed an isolated restore, including object rehydration after
local deletion. Preserve the queue for review; do not activate automatic purge
or infer legitimate consent from its `source=customer` note.

The backup is protected against application-bound object deletion; same-account
loss or compromise of account administrators remains outside that boundary.

## Local restore command

Download the private snapshot through the authenticated Worker snapshot endpoint
or the private bucket, and use its SHA256 from the successful publication receipt.
Run from this directory:

```sh
python3 restore.py /private/snapshot.json PUBLICATION_SHA256 /private/new-restore
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
