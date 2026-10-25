-- Apply once before deploying the safe-maintenance Worker. Do not backfill.
-- Historical requests remain NULL and cannot purge, including admin force.
-- No existing contributor, request or object is deleted or reauthorized.
ALTER TABLE revoke_requests ADD COLUMN purge_authorized_at TEXT;
ALTER TABLE revoke_requests ADD COLUMN purge_authority TEXT;
