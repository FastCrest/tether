-- Reflex contribution-worker D1 schema.
-- Tracks contributors, uploads, revoke requests, and per-contributor rate-limit
-- accounting for the Curate data-collection program.

-- Contributors. One row per contributor_id (anonymous Free or Pro customer_id).
CREATE TABLE IF NOT EXISTS contributors (
  contributor_id TEXT PRIMARY KEY,
  tier TEXT NOT NULL,                       -- "free" | "pro" | "enterprise"
  first_seen_at TEXT NOT NULL,              -- ISO 8601 UTC
  last_active_at TEXT,                      -- updated on every successful upload
  total_episodes INTEGER NOT NULL DEFAULT 0,
  total_bytes INTEGER NOT NULL DEFAULT 0,
  total_uploads INTEGER NOT NULL DEFAULT 0,
  revoked_at TEXT,                          -- non-null after /v1/revoke/cascade
  cascade_completed_at TEXT,                -- set when R2 + derived datasets purged
  notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_contributors_tier ON contributors(tier);
CREATE INDEX IF NOT EXISTS idx_contributors_revoked ON contributors(revoked_at)
  WHERE revoked_at IS NOT NULL;

-- Contributor Authentication v1 key registry. Public keys are raw Ed25519
-- keys encoded as unpadded base64url; private keys never reach this service.
CREATE TABLE IF NOT EXISTS contributor_keys (
  key_id TEXT PRIMARY KEY,
  contributor_id TEXT NOT NULL,
  public_key_base64url TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('active', 'inactive')),
  created_at INTEGER NOT NULL,
  deactivated_at INTEGER,
  FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_contributor_one_active_key
  ON contributor_keys(contributor_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_contributor_keys_owner
  ON contributor_keys(contributor_id);

-- Replays are rejected for ten minutes. The scheduled handler prunes expired
-- rows; uniqueness remains fail-closed if pruning is delayed.
CREATE TABLE IF NOT EXISTS contributor_nonces (
  contributor_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (contributor_id, nonce)
);
CREATE INDEX IF NOT EXISTS idx_contributor_nonces_expiry
  ON contributor_nonces(expires_at);

-- Uploads. One row per accepted upload.
CREATE TABLE IF NOT EXISTS uploads (
  upload_id TEXT PRIMARY KEY,               -- UUID4 generated at /v1/uploads/sign
  contributor_id TEXT NOT NULL,
  reservation_key TEXT,
  request_sha256 TEXT,
  r2_key TEXT NOT NULL,                     -- the R2 object key, e.g. free-contributors/.../sess.jsonl
  byte_size INTEGER NOT NULL,
  media_type TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  capability_sha256 TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  status TEXT NOT NULL,                     -- pending/uploading/uploaded/completed/rejected/expired/purged
  signed_at TEXT NOT NULL,                  -- when the PUT URL was issued
  signed_at_epoch INTEGER NOT NULL,
  attempt_id TEXT,
  attempt_started_at INTEGER,
  uploaded_at INTEGER,
  completed_at TEXT,
  completion_id TEXT,
  user_agent TEXT,
  source_ip TEXT,
  notes TEXT,
  FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id)
);

CREATE INDEX IF NOT EXISTS idx_uploads_contributor ON uploads(contributor_id);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
CREATE INDEX IF NOT EXISTS idx_uploads_signed_at ON uploads(signed_at);
CREATE INDEX IF NOT EXISTS idx_uploads_sign_window
  ON uploads(contributor_id, signed_at_epoch);
CREATE INDEX IF NOT EXISTS idx_uploads_stale_attempts
  ON uploads(status, attempt_started_at) WHERE status = 'uploading';
CREATE UNIQUE INDEX IF NOT EXISTS idx_uploads_reservation_key
  ON uploads(contributor_id, reservation_key) WHERE reservation_key IS NOT NULL;

-- Daily upload-bytes counters per contributor for rate limiting.
-- Phase 1: 10 GB/day default; raise on request.
CREATE TABLE IF NOT EXISTS daily_uploads (
  contributor_id TEXT NOT NULL,
  utc_date TEXT NOT NULL,                   -- "YYYY-MM-DD"
  bytes_uploaded INTEGER NOT NULL DEFAULT 0,
  uploads_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (contributor_id, utc_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_uploads_date ON daily_uploads(utc_date);

-- Revoke requests. One row per /v1/revoke/cascade call.
-- Phase 1 cascade stages (per consent-revoke_research.md):
--   T+0          revoke immediate (request row created)
--   T+5min       tombstone (sign endpoint already refuses; this stage just stamps the time)
--   T+5min+      R2 purge (paginated list+delete under <tier>-contributors/<id>/)
--   auto         derived dataset rebuild (no datasets at Phase 1; auto-complete on cascade init)
--   auto         buyer notification (no buyers at Phase 1; auto-complete on cascade init)
CREATE TABLE IF NOT EXISTS revoke_requests (
  request_id TEXT PRIMARY KEY,              -- UUID4
  contributor_id TEXT NOT NULL,
  requested_at TEXT NOT NULL,
  scope TEXT NOT NULL,                      -- "all" | "future_only"
  status TEXT NOT NULL,                     -- "pending" | "in_progress" | "completed" | "failed"
  r2_objects_purged INTEGER NOT NULL DEFAULT 0,
  derived_datasets_rebuilt INTEGER NOT NULL DEFAULT 0,
  buyer_notifications_sent INTEGER NOT NULL DEFAULT 0,
  -- Stage timestamp columns (additive in v1.1; NULL until that stage runs).
  tombstone_at TEXT,
  r2_purge_started_at TEXT,
  r2_purge_completed_at TEXT,
  derived_rebuild_completed_at TEXT,
  buyer_notification_completed_at TEXT,
  completed_at TEXT,                        -- top-level: all stages done
  notes TEXT,
  FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id)
);

-- Migration helper for existing DBs (no-op on fresh install).
-- Cloudflare D1's `IF NOT EXISTS` on ALTER doesn't exist; instead we
-- handle column-add at the application layer (worker reads NULL when
-- absent). The CREATE TABLE above already includes the new columns for
-- fresh installs.

CREATE INDEX IF NOT EXISTS idx_revoke_status ON revoke_requests(status);
CREATE INDEX IF NOT EXISTS idx_revoke_contributor ON revoke_requests(contributor_id);
