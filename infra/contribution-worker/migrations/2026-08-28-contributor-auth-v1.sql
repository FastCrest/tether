-- Contributor Authentication v1 and Upload Reservation v1.
-- Apply once to existing contribution-worker D1 databases before deploying
-- the corresponding Worker code.
-- deploy.sh makes that operation idempotent by verifying the complete expected
-- column set, applying only when all additive columns are absent, and failing
-- closed on a partially-applied migration.

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

CREATE TABLE IF NOT EXISTS contributor_nonces (
  contributor_id TEXT NOT NULL,
  nonce TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  PRIMARY KEY (contributor_id, nonce)
);
CREATE INDEX IF NOT EXISTS idx_contributor_nonces_expiry
  ON contributor_nonces(expires_at);

ALTER TABLE uploads ADD COLUMN content_sha256 TEXT;
ALTER TABLE uploads ADD COLUMN media_type TEXT NOT NULL DEFAULT 'application/jsonl';
ALTER TABLE uploads ADD COLUMN manifest_sha256 TEXT;
ALTER TABLE uploads ADD COLUMN capability_sha256 TEXT;
ALTER TABLE uploads ADD COLUMN expires_at INTEGER;
ALTER TABLE uploads ADD COLUMN signed_at_epoch INTEGER;
ALTER TABLE uploads ADD COLUMN attempt_id TEXT;
ALTER TABLE uploads ADD COLUMN attempt_started_at INTEGER;
ALTER TABLE uploads ADD COLUMN uploaded_at INTEGER;
ALTER TABLE uploads ADD COLUMN completion_id TEXT;
ALTER TABLE uploads ADD COLUMN reservation_key TEXT;
ALTER TABLE uploads ADD COLUMN request_sha256 TEXT;

CREATE INDEX IF NOT EXISTS idx_uploads_sign_window
  ON uploads(contributor_id, signed_at_epoch);
CREATE INDEX IF NOT EXISTS idx_uploads_stale_attempts
  ON uploads(status, attempt_started_at) WHERE status = 'uploading';
CREATE UNIQUE INDEX IF NOT EXISTS idx_uploads_reservation_key
  ON uploads(contributor_id, reservation_key) WHERE reservation_key IS NOT NULL;

-- Legacy pending rows predate authenticated reservations and must never be
-- claimable. Historical completed/purged rows remain readable for operators.
UPDATE uploads
   SET signed_at_epoch = CAST(strftime('%s', signed_at) AS INTEGER)
 WHERE signed_at_epoch IS NULL
   AND signed_at IS NOT NULL;

UPDATE uploads
   SET status = 'expired'
 WHERE status = 'pending'
   AND (capability_sha256 IS NULL OR expires_at IS NULL);
