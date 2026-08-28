-- Migration: add per-stage timestamp columns to revoke_requests.
-- Cloudflare D1 does NOT support ALTER TABLE ... ADD COLUMN IF NOT EXISTS,
-- so this migration must only be applied once per environment.
-- deploy.sh verifies that either all or none of these columns exist and applies
-- this tracked file only in the all-missing case. Do not run it blindly.
ALTER TABLE revoke_requests ADD COLUMN tombstone_at TEXT;
ALTER TABLE revoke_requests ADD COLUMN r2_purge_started_at TEXT;
ALTER TABLE revoke_requests ADD COLUMN r2_purge_completed_at TEXT;
ALTER TABLE revoke_requests ADD COLUMN derived_rebuild_completed_at TEXT;
ALTER TABLE revoke_requests ADD COLUMN buyer_notification_completed_at TEXT;
