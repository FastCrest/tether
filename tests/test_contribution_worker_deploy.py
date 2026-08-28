from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
DEPLOY = ROOT / "infra" / "contribution-worker" / "deploy.sh"
DATABASE_ID = "39cc37c0-524c-4a7a-921e-76cc44cb1494"


def test_deploy_script_migrates_existing_d1_before_code_and_is_idempotent(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "wrangler.log"
    state = tmp_path / "schema-state"
    state.write_text("legacy\n", encoding="utf-8")

    wrangler = fake_bin / "wrangler"
    wrangler.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_WRANGLER_LOG"
case " $* " in
  *" --version "*) printf '%s\\n' 'wrangler 4.99.0';;
  *" whoami "*) printf '%s\\n' 'authenticated';;
  *" d1 list --json "*) printf '%s\\n' '[{{"name":"reflex-contributions","uuid":"{DATABASE_ID}"}}]';;
  *" r2 bucket list "*) printf '%s\\n' 'reflex-curate';;
  *" secret list "*) printf '%s\\n' '[{{"name": "ADMIN_TOKEN"}},{{"name": "UPLOAD_CAPABILITY_SECRET"}}]';;
  *" d1 execute "*" --command PRAGMA table_info(uploads) "*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == fresh* ]]; then
      printf '%s\\n' '[{{"results":[],"success":true}}]'
    elif [[ "$current" == *auth* ]]; then
      printf '%s\\n' '[{{"results":[{{"name":"upload_id"}},{{"name":"content_sha256"}},{{"name":"media_type"}},{{"name":"manifest_sha256"}},{{"name":"capability_sha256"}},{{"name":"expires_at"}},{{"name":"signed_at_epoch"}},{{"name":"attempt_id"}},{{"name":"attempt_started_at"}},{{"name":"uploaded_at"}},{{"name":"completion_id"}},{{"name":"reservation_key"}},{{"name":"request_sha256"}}],"success":true}}]'
    else
      printf '%s\\n' '[{{"results":[{{"name":"upload_id"}},{{"name":"contributor_id"}}],"success":true}}]'
    fi;;
  *" d1 execute "*" --command PRAGMA table_info(revoke_requests) "*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == fresh* ]]; then
      printf '%s\\n' '[{{"results":[],"success":true}}]'
    elif [[ "$current" == *revoke* ]]; then
      printf '%s\\n' '[{{"results":[{{"name":"request_id"}},{{"name":"tombstone_at"}},{{"name":"r2_purge_started_at"}},{{"name":"r2_purge_completed_at"}},{{"name":"derived_rebuild_completed_at"}},{{"name":"buyer_notification_completed_at"}}],"success":true}}]'
    else
      printf '%s\\n' '[{{"results":[{{"name":"request_id"}},{{"name":"contributor_id"}}],"success":true}}]'
    fi;;
  *" d1 execute "*" --command PRAGMA table_info(contributor_keys) "*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* || "$current" == *partial_table* || "$current" == *partial_index* ]]; then
      printf '%s\\n' '[{{"results":[{{"name":"key_id","type":"TEXT","notnull":0,"pk":1}},{{"name":"contributor_id","type":"TEXT","notnull":1,"pk":0}},{{"name":"public_key_base64url","type":"TEXT","notnull":1,"pk":0}},{{"name":"status","type":"TEXT","notnull":1,"pk":0}},{{"name":"created_at","type":"INTEGER","notnull":1,"pk":0}},{{"name":"deactivated_at","type":"INTEGER","notnull":0,"pk":0}}],"success":true}}]'
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *" d1 execute "*" --command PRAGMA table_info(contributor_nonces) "*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* || "$current" == *partial_index* ]]; then
      printf '%s\\n' '[{{"results":[{{"name":"contributor_id","type":"TEXT","notnull":1,"pk":1}},{{"name":"nonce","type":"TEXT","notnull":1,"pk":2}},{{"name":"expires_at","type":"INTEGER","notnull":1,"pk":0}}],"success":true}}]'
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *"name='contributor_keys'"*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* || "$current" == *partial_table* || "$current" == *partial_index* ]]; then
      if [[ "$current" == *wrong_structure* ]]; then
        sql="CREATE TABLE contributor_keys (key_id TEXT, contributor_id TEXT, public_key_base64url TEXT, status TEXT, created_at INTEGER, deactivated_at INTEGER)"
      else
        sql="CREATE TABLE contributor_keys (key_id TEXT PRIMARY KEY, contributor_id TEXT NOT NULL, public_key_base64url TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK (status IN ('active', 'inactive')), created_at INTEGER NOT NULL, deactivated_at INTEGER, FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id))"
      fi
      python3 -c 'import json,sys; print(json.dumps([{{"results":[{{"sql":sys.argv[1]}}],"success":True}}]))' "$sql"
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *"name='contributor_nonces'"*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* || "$current" == *partial_index* ]]; then
      sql="CREATE TABLE contributor_nonces (contributor_id TEXT NOT NULL, nonce TEXT NOT NULL, expires_at INTEGER NOT NULL, PRIMARY KEY (contributor_id, nonce))"
      python3 -c 'import json,sys; print(json.dumps([{{"results":[{{"sql":sys.argv[1]}}],"success":True}}]))' "$sql"
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *"name='idx_contributor_one_active_key'"*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* || "$current" == *partial_index* ]]; then
      sql="CREATE UNIQUE INDEX idx_contributor_one_active_key ON contributor_keys(contributor_id) WHERE status = 'active'"
      python3 -c 'import json,sys; print(json.dumps([{{"results":[{{"sql":sys.argv[1]}}],"success":True}}]))' "$sql"
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *"name='idx_contributor_keys_owner'"*|*"name='idx_contributor_nonces_expiry'"*|*"name='idx_uploads_sign_window'"*|*"name='idx_uploads_stale_attempts'"*|*"name='idx_uploads_reservation_key'"*)
    current=$(<"$FAKE_SCHEMA_STATE")
    if [[ "$current" == *auth* && "$current" != *partial_index* ]]; then
      case "$*" in
        *idx_contributor_keys_owner*) sql="CREATE INDEX idx_contributor_keys_owner ON contributor_keys(contributor_id)";;
        *idx_contributor_nonces_expiry*) sql="CREATE INDEX idx_contributor_nonces_expiry ON contributor_nonces(expires_at)";;
        *idx_uploads_sign_window*) sql="CREATE INDEX idx_uploads_sign_window ON uploads(contributor_id, signed_at_epoch)";;
        *idx_uploads_stale_attempts*) sql="CREATE INDEX idx_uploads_stale_attempts ON uploads(status, attempt_started_at) WHERE status = 'uploading'";;
        *idx_uploads_reservation_key*) sql="CREATE UNIQUE INDEX idx_uploads_reservation_key ON uploads(contributor_id, reservation_key) WHERE reservation_key IS NOT NULL";;
      esac
      python3 -c 'import json,sys; print(json.dumps([{{"results":[{{"sql":sys.argv[1]}}],"success":True}}]))' "$sql"
    elif [[ "$current" == *partial_index* && "$*" != *"idx_uploads_stale_attempts"* ]]; then
      sql="CREATE INDEX idx_contributor_keys_owner ON contributor_keys(contributor_id)"
      python3 -c 'import json,sys; print(json.dumps([{{"results":[{{"sql":sys.argv[1]}}],"success":True}}]))' "$sql"
    else printf '%s\\n' '[{{"results":[],"success":true}}]'; fi;;
  *" --file ./migrations/2026-05-05-revoke-cascade-stages.sql "*)
    current=$(<"$FAKE_SCHEMA_STATE"); printf '%s\\n' "${{current}} revoke" > "$FAKE_SCHEMA_STATE";;
  *" --file ./migrations/2026-08-28-contributor-auth-v1.sql "*)
    current=$(<"$FAKE_SCHEMA_STATE"); printf '%s\\n' "${{current}} auth" > "$FAKE_SCHEMA_STATE";;
  *" --file ./schema.sql "*) printf '%s\\n' 'revoke auth' > "$FAKE_SCHEMA_STATE";;
  *" deploy "*) printf '%s\\n' 'Uploaded https://reflex-contributions.fastcrest.workers.dev';;
  *) printf '%s\\n' "unexpected wrangler invocation: $*" >&2; exit 64;;
esac
""",
        encoding="utf-8",
    )
    wrangler.chmod(0o755)

    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >> "$FAKE_WRANGLER_LOG"
case "$*" in
  *"/healthz"*) printf '200';;
  *"/v1/uploads/sign"*) printf '401';;
  *) exit 64;;
esac
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)

    env = {
        **os.environ,
        "FAKE_SCHEMA_STATE": str(state),
        "FAKE_WRANGLER_LOG": str(log),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }
    for _ in range(2):
        result = subprocess.run(
            [str(DEPLOY)],
            cwd=DEPLOY.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    calls = log.read_text(encoding="utf-8").splitlines()
    revoke = [
        index for index, call in enumerate(calls) if "2026-05-05-revoke-cascade-stages.sql" in call
    ]
    auth = [
        index for index, call in enumerate(calls) if "2026-08-28-contributor-auth-v1.sql" in call
    ]
    deploys = [index for index, call in enumerate(calls) if call == "deploy"]
    assert len(revoke) == 1
    assert len(auth) == 1
    assert len(deploys) == 2
    assert revoke[0] < auth[0] < deploys[0]
    assert not any("--file ./schema.sql" in call for call in calls)
    assert sum("/healthz" in call for call in calls) == 2
    assert sum("/v1/uploads/sign" in call for call in calls) == 2

    # A separate empty-schema dry run uses schema.sql exactly once and never
    # tries legacy ALTER migrations before deployment.
    state.write_text("fresh\n", encoding="utf-8")
    log.write_text("", encoding="utf-8")
    fresh = subprocess.run(
        [str(DEPLOY)],
        cwd=DEPLOY.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    fresh_calls = log.read_text(encoding="utf-8").splitlines()
    schema_index = next(
        index for index, call in enumerate(fresh_calls) if "--file ./schema.sql" in call
    )
    deploy_index = fresh_calls.index("deploy")
    assert schema_index < deploy_index
    assert not any("/migrations/" in call for call in fresh_calls)

    for partial in ("partial_table", "revoke auth partial_index", "revoke auth wrong_structure"):
        state.write_text(f"{partial}\n", encoding="utf-8")
        log.write_text("", encoding="utf-8")
        failed = subprocess.run(
            [str(DEPLOY)],
            cwd=DEPLOY.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert failed.returncode != 0
        failed_calls = log.read_text(encoding="utf-8").splitlines()
        assert "deploy" not in failed_calls
        assert "refusing to deploy" in failed.stderr or "missing required" in failed.stderr


def test_deploy_script_is_valid_shell_and_has_no_stale_protocol_instructions() -> None:
    result = subprocess.run(
        ["bash", "-n", str(DEPLOY)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    source = DEPLOY.read_text(encoding="utf-8")
    assert "src/reflex" not in source
    assert "replace UploadStub" not in source
    assert "signed reserve -> capability PUT -> signed completion" in source
    assert "reflex-contributions.fastcrest.workers.dev" in source


def test_real_schema_has_exact_auth_semantics(tmp_path: Path) -> None:
    database = sqlite3.connect(tmp_path / "schema.db")
    database.executescript((DEPLOY.parent / "schema.sql").read_text())

    def canonical(sql: str) -> str:
        value = re.sub(r"\s+", " ", sql.strip()).lower()
        return re.sub(r"\s*([(),=])\s*", r"\1", value).removesuffix(";")

    actual_keys_sql = database.execute(
        "SELECT sql FROM sqlite_master WHERE name='contributor_keys'"
    ).fetchone()[0]
    actual_nonces_sql = database.execute(
        "SELECT sql FROM sqlite_master WHERE name='contributor_nonces'"
    ).fetchone()[0]
    assert canonical(actual_keys_sql) == canonical(
        "CREATE TABLE contributor_keys (key_id TEXT PRIMARY KEY, contributor_id TEXT NOT NULL, public_key_base64url TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK (status IN ('active', 'inactive')), created_at INTEGER NOT NULL, deactivated_at INTEGER, FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id))"
    )
    assert canonical(actual_nonces_sql) == canonical(
        "CREATE TABLE contributor_nonces (contributor_id TEXT NOT NULL, nonce TEXT NOT NULL, expires_at INTEGER NOT NULL, PRIMARY KEY (contributor_id, nonce))"
    )
    keys = database.execute("PRAGMA table_info(contributor_keys)").fetchall()
    assert [(r[1], r[2], r[3], r[5]) for r in keys] == [
        ("key_id", "TEXT", 0, 1),
        ("contributor_id", "TEXT", 1, 0),
        ("public_key_base64url", "TEXT", 1, 0),
        ("status", "TEXT", 1, 0),
        ("created_at", "INTEGER", 1, 0),
        ("deactivated_at", "INTEGER", 0, 0),
    ]
    assert database.execute("PRAGMA foreign_key_list(contributor_keys)").fetchall()[0][2:5] == (
        "contributors",
        "contributor_id",
        "contributor_id",
    )
    nonces = database.execute("PRAGMA table_info(contributor_nonces)").fetchall()
    assert [(r[1], r[2], r[3], r[5]) for r in nonces] == [
        ("contributor_id", "TEXT", 1, 1),
        ("nonce", "TEXT", 1, 2),
        ("expires_at", "INTEGER", 1, 0),
    ]
    expected_indexes = {
        "idx_contributor_one_active_key": (["contributor_id"], 1, 1),
        "idx_contributor_keys_owner": (["contributor_id"], 0, 0),
        "idx_contributor_nonces_expiry": (["expires_at"], 0, 0),
        "idx_uploads_sign_window": (["contributor_id", "signed_at_epoch"], 0, 0),
        "idx_uploads_stale_attempts": (["status", "attempt_started_at"], 0, 1),
        "idx_uploads_reservation_key": (["contributor_id", "reservation_key"], 1, 1),
    }
    for name, (columns, unique, partial) in expected_indexes.items():
        table = (
            "contributor_keys"
            if "contributor_" in name and "nonces" not in name
            else ("contributor_nonces" if "nonces" in name else "uploads")
        )
        listed = {r[1]: r for r in database.execute(f"PRAGMA index_list({table})")}
        assert listed[name][2] == unique
        assert listed[name][4] == partial
        assert [r[2] for r in database.execute(f"PRAGMA index_info({name})")] == columns
