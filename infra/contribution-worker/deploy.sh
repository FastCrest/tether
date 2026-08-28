#!/usr/bin/env bash
#
# Reflex contribution-worker — migration-safe one-command deploy.
#
# Existing databases receive each tracked additive migration before Worker
# code. Fresh databases receive the complete schema once. Column verification
# makes reruns idempotent and fails closed on partially-applied migrations.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { printf "${GREEN}\xe2\x9c\x93${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}\xe2\x9a\xa0${NC} %s\n" "$*"; }
err()  { printf "${RED}\xe2\x9c\x97${NC} %s\n" "$*" >&2; }
info() { printf "${CYAN}\xe2\x86\x92${NC} %s\n" "$*"; }

cd "$(dirname "$0")"

DB_NAME="reflex-contributions"
BUCKET_NAME="reflex-curate"
DOCUMENTED_WORKER_URL="https://reflex-contributions.fastcrest.workers.dev"
REVOKE_MIGRATION="./migrations/2026-05-05-revoke-cascade-stages.sql"
AUTH_MIGRATION="./migrations/2026-08-28-contributor-auth-v1.sql"

if ! command -v wrangler >/dev/null 2>&1; then
    info "Installing Wrangler 4 globally..."
    npm install -g wrangler@4
fi
wrangler_version=$(wrangler --version)
if [[ ! "$wrangler_version" =~ ([[:space:]]|^)4\. ]]; then
    err "Wrangler 4.x or newer is required; found: $wrangler_version"
    exit 1
fi
ok "Wrangler available: $wrangler_version"

if ! wrangler whoami >/dev/null 2>&1; then
    info "Running wrangler login (opens browser)..."
    wrangler login
fi
ok "Wrangler authenticated"

database_id() {
    wrangler d1 list --json 2>/dev/null | python3 -c '
import json, sys
name = sys.argv[1]
for database in json.load(sys.stdin):
    if database.get("name") == name:
        print(database.get("uuid") or database.get("id") or "")
        break
' "$DB_NAME"
}

DB_ID=$(database_id)
if [ -z "$DB_ID" ]; then
    info "Creating D1 database $DB_NAME..."
    wrangler d1 create "$DB_NAME"
    DB_ID=$(database_id)
    if [ -z "$DB_ID" ]; then
        err "D1 was created but its ID was not present in 'wrangler d1 list --json'"
        exit 1
    fi
    ok "D1 database created: id=$DB_ID"
else
    ok "D1 database $DB_NAME already exists (id=$DB_ID)"
fi

bucket_list=$(wrangler r2 bucket list)
if grep -qE "(^|[[:space:]])${BUCKET_NAME}([[:space:]]|$)" <<<"$bucket_list"; then
    ok "R2 bucket $BUCKET_NAME already exists"
else
    info "Creating R2 bucket $BUCKET_NAME..."
    wrangler r2 bucket create "$BUCKET_NAME"
    ok "R2 bucket created: $BUCKET_NAME"
fi

if grep -q 'database_id = "REPLACE_AFTER_CREATE"' wrangler.toml; then
    info "Patching wrangler.toml with database_id=$DB_ID"
    sed -i.bak "s/REPLACE_AFTER_CREATE/$DB_ID/" wrangler.toml
    ok "wrangler.toml updated"
else
    configured_id=$(sed -n 's/^database_id = "\([^"]*\)"/\1/p' wrangler.toml | head -1)
    if [ "$configured_id" != "$DB_ID" ]; then
        err "wrangler.toml database_id=$configured_id does not match remote $DB_NAME id=$DB_ID"
        exit 1
    fi
    ok "wrangler.toml targets the expected D1 database"
fi

d1_columns() {
    local table=$1
    wrangler d1 execute "$DB_NAME" --remote --json \
        --command "PRAGMA table_info(${table})" 2>/dev/null | python3 -c '
import json, sys
payload = json.load(sys.stdin)
for batch in payload:
    for row in batch.get("results", []):
        name = row.get("name")
        if name:
            print(name)
'
}

d1_table_signature() {
    local table=$1
    wrangler d1 execute "$DB_NAME" --remote --json \
        --command "PRAGMA table_info(${table})" 2>/dev/null | python3 -c '
import json, sys
for batch in json.load(sys.stdin):
    for row in batch.get("results", []):
        print(f"{row.get('"'"'name'"'"')}|{str(row.get('"'"'type'"'"', '"'"''"'"')).upper()}|{int(row.get('"'"'notnull'"'"', 0))}|{int(row.get('"'"'pk'"'"', 0))}")
'
}

verify_columns() {
    local table=$1
    local existing=$2
    shift 2
    local missing=()
    local column
    for column in "$@"; do
        if ! grep -Fxq "$column" <<<"$existing"; then
            missing+=("$column")
        fi
    done
    if [ "${#missing[@]}" -ne 0 ]; then
        err "D1 table $table is missing required columns: ${missing[*]}"
        exit 1
    fi
}

d1_object_sql() {
    local type=$1 name=$2
    wrangler d1 execute "$DB_NAME" --remote --json \
        --command "SELECT sql FROM sqlite_master WHERE type='${type}' AND name='${name}'" \
        2>/dev/null | python3 -c '
import json, sys
for batch in json.load(sys.stdin):
    for row in batch.get("results", []):
        if row.get("sql"):
            print(row["sql"])
'
}

require_sql_fragment() {
    local label=$1 sql=$2 fragment=$3
    if [[ "$sql" != *"$fragment"* ]]; then
        err "$label is missing required SQL constraint: $fragment"
        exit 1
    fi
}

require_exact_sql() {
    local label=$1 actual=$2 expected=$3
    local normalized_actual normalized_expected
    normalized_actual=$(python3 -c 'import re,sys; value=re.sub(r"\s+", " ", sys.argv[1].strip()).lower(); print(re.sub(r"\s*([(),=])\s*", r"\1", value).removesuffix(";"))' "$actual")
    normalized_expected=$(python3 -c 'import re,sys; value=re.sub(r"\s+", " ", sys.argv[1].strip()).lower(); print(re.sub(r"\s*([(),=])\s*", r"\1", value).removesuffix(";"))' "$expected")
    if [ "$normalized_actual" != "$normalized_expected" ]; then
        err "$label has the wrong table/index structure; refusing to deploy"
        exit 1
    fi
}

apply_additive_migration() {
    local label=$1
    local table=$2
    local migration=$3
    shift 3
    local existing present=0 column
    existing=$(d1_columns "$table")
    for column in "$@"; do
        if grep -Fxq "$column" <<<"$existing"; then
            present=$((present + 1))
        fi
    done
    if [ "$present" -eq "$#" ]; then
        ok "$label migration already satisfied"
        return
    fi
    if [ "$present" -ne 0 ]; then
        err "$label migration is partially applied ($present/$# columns present); refusing to deploy"
        exit 1
    fi
    info "Applying tracked $label migration before Worker deployment..."
    wrangler d1 execute "$DB_NAME" --remote --file "$migration"
    existing=$(d1_columns "$table")
    verify_columns "$table" "$existing" "$@"
    ok "$label migration applied and verified"
}

upload_columns=(
    content_sha256 media_type manifest_sha256 capability_sha256 expires_at
    signed_at_epoch attempt_id attempt_started_at uploaded_at completion_id
    reservation_key request_sha256
)
key_columns=(
    key_id contributor_id public_key_base64url status created_at deactivated_at
)
nonce_columns=(contributor_id nonce expires_at)
auth_indexes=(
    idx_contributor_one_active_key idx_contributor_keys_owner
    idx_contributor_nonces_expiry idx_uploads_sign_window idx_uploads_stale_attempts
    idx_uploads_reservation_key
)
revoke_columns=(
    tombstone_at r2_purge_started_at r2_purge_completed_at
    derived_rebuild_completed_at buyer_notification_completed_at
)

verify_contributor_auth_schema() {
    local keys_sql active_index_sql
    verify_columns uploads "$(d1_columns uploads)" "${upload_columns[@]}"
    verify_columns contributor_keys "$(d1_columns contributor_keys)" "${key_columns[@]}"
    verify_columns contributor_nonces "$(d1_columns contributor_nonces)" "${nonce_columns[@]}"
    require_exact_sql contributor_keys_columns "$(d1_table_signature contributor_keys)" \
      $'key_id|TEXT|0|1\ncontributor_id|TEXT|1|0\npublic_key_base64url|TEXT|1|0\nstatus|TEXT|1|0\ncreated_at|INTEGER|1|0\ndeactivated_at|INTEGER|0|0'
    require_exact_sql contributor_nonces_columns "$(d1_table_signature contributor_nonces)" \
      $'contributor_id|TEXT|1|1\nnonce|TEXT|1|2\nexpires_at|INTEGER|1|0'
    require_exact_sql contributor_keys "$(d1_object_sql table contributor_keys)" \
      "CREATE TABLE contributor_keys (key_id TEXT PRIMARY KEY, contributor_id TEXT NOT NULL, public_key_base64url TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK (status IN ('active', 'inactive')), created_at INTEGER NOT NULL, deactivated_at INTEGER, FOREIGN KEY (contributor_id) REFERENCES contributors(contributor_id))"
    require_exact_sql contributor_nonces "$(d1_object_sql table contributor_nonces)" \
      "CREATE TABLE contributor_nonces (contributor_id TEXT NOT NULL, nonce TEXT NOT NULL, expires_at INTEGER NOT NULL, PRIMARY KEY (contributor_id, nonce))"
    require_exact_sql idx_contributor_one_active_key "$(d1_object_sql index idx_contributor_one_active_key)" \
      "CREATE UNIQUE INDEX idx_contributor_one_active_key ON contributor_keys(contributor_id) WHERE status = 'active'"
    require_exact_sql idx_contributor_keys_owner "$(d1_object_sql index idx_contributor_keys_owner)" \
      "CREATE INDEX idx_contributor_keys_owner ON contributor_keys(contributor_id)"
    require_exact_sql idx_contributor_nonces_expiry "$(d1_object_sql index idx_contributor_nonces_expiry)" \
      "CREATE INDEX idx_contributor_nonces_expiry ON contributor_nonces(expires_at)"
    require_exact_sql idx_uploads_sign_window "$(d1_object_sql index idx_uploads_sign_window)" \
      "CREATE INDEX idx_uploads_sign_window ON uploads(contributor_id, signed_at_epoch)"
    require_exact_sql idx_uploads_stale_attempts "$(d1_object_sql index idx_uploads_stale_attempts)" \
      "CREATE INDEX idx_uploads_stale_attempts ON uploads(status, attempt_started_at) WHERE status = 'uploading'"
    require_exact_sql idx_uploads_reservation_key "$(d1_object_sql index idx_uploads_reservation_key)" \
      "CREATE UNIQUE INDEX idx_uploads_reservation_key ON uploads(contributor_id, reservation_key) WHERE reservation_key IS NOT NULL"
    keys_sql=$(d1_object_sql table contributor_keys)
    active_index_sql=$(d1_object_sql index idx_contributor_one_active_key)
    require_sql_fragment "contributor_keys" "$keys_sql" "CHECK (status IN ("
    require_sql_fragment "idx_contributor_one_active_key" "$active_index_sql" "CREATE UNIQUE INDEX"
}

apply_contributor_auth_migration() {
    local present=0 total=0 column index existing sql
    for table_and_columns in \
        "uploads:${upload_columns[*]}" \
        "contributor_keys:${key_columns[*]}" \
        "contributor_nonces:${nonce_columns[*]}"; do
        local table=${table_and_columns%%:*}
        local columns=${table_and_columns#*:}
        existing=$(d1_columns "$table")
        for column in $columns; do
            total=$((total + 1))
            grep -Fxq "$column" <<<"$existing" && present=$((present + 1))
        done
    done
    for index in "${auth_indexes[@]}"; do
        total=$((total + 1))
        sql=$(d1_object_sql index "$index")
        [ -n "$sql" ] && present=$((present + 1))
    done
    if [ "$present" -eq "$total" ]; then
        verify_contributor_auth_schema
        ok "Contributor Authentication v1 migration already satisfied"
        return
    fi
    if [ "$present" -ne 0 ]; then
        err "Contributor Authentication v1 schema is partial ($present/$total artifacts present); refusing to deploy"
        exit 1
    fi
    info "Applying tracked Contributor Authentication v1 migration before Worker deployment..."
    wrangler d1 execute "$DB_NAME" --remote --file "$AUTH_MIGRATION"
    verify_contributor_auth_schema
    ok "Contributor Authentication v1 migration applied and verified"
}

existing_upload_columns=$(d1_columns uploads)
if [ -z "$existing_upload_columns" ]; then
    info "Fresh D1 detected; applying complete schema.sql..."
    wrangler d1 execute "$DB_NAME" --remote --file ./schema.sql
    verify_contributor_auth_schema
    verify_columns revoke_requests "$(d1_columns revoke_requests)" "${revoke_columns[@]}"
    ok "Fresh schema applied and contributor-auth shape verified"
else
    info "Existing D1 detected; schema.sql will not be re-applied"
    apply_additive_migration \
        "revoke cascade" revoke_requests "$REVOKE_MIGRATION" "${revoke_columns[@]}"
    apply_contributor_auth_migration
fi

secret_list=$(wrangler secret list)
if grep -q '"name": "ADMIN_TOKEN"' <<<"$secret_list"; then
    ok "ADMIN_TOKEN already set"
else
    ADMIN_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    warn "Save the following generated ADMIN_TOKEN now; it will not be shown again:"
    printf "    ${YELLOW}%s${NC}\n" "$ADMIN_TOKEN"
    printf '%s\n' "$ADMIN_TOKEN" | wrangler secret put ADMIN_TOKEN
    ok "ADMIN_TOKEN set"
fi

if grep -q '"name": "UPLOAD_CAPABILITY_SECRET"' <<<"$secret_list"; then
    ok "UPLOAD_CAPABILITY_SECRET already set"
else
    UPLOAD_CAPABILITY_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    printf '%s\n' "$UPLOAD_CAPABILITY_SECRET" | wrangler secret put UPLOAD_CAPABILITY_SECRET
    ok "UPLOAD_CAPABILITY_SECRET generated and set"
fi

info "Deploying Worker only after D1 verification..."
if ! deploy_output=$(wrangler deploy 2>&1); then
    printf '%s\n' "$deploy_output" >&2
    err "Worker deployment failed after successful D1 verification"
    exit 1
fi
printf '%s\n' "$deploy_output"
ok "Worker deployed"

WORKER_URL=$(printf '%s\n' "$deploy_output" \
    | grep -oE 'https://[A-Za-z0-9-]+\.[A-Za-z0-9-]+\.workers\.dev' \
    | tail -1 || true)
if [ -z "$WORKER_URL" ]; then
    WORKER_URL=${TETHER_CONTRIBUTION_WORKER_URL:-$DOCUMENTED_WORKER_URL}
    warn "Deploy output omitted a workers.dev URL; using documented/configured $WORKER_URL"
fi

info "Smoke testing $WORKER_URL..."
health_status=$(curl --silent --show-error --max-time 15 --output /dev/null \
    --write-out '%{http_code}' "$WORKER_URL/healthz" || true)
if [ "$health_status" != "200" ]; then
    err "/healthz returned HTTP ${health_status:-transport_error}"
    exit 1
fi
ok "/healthz returned 200"

auth_status=$(curl --silent --show-error --max-time 15 --output /dev/null \
    --write-out '%{http_code}' --request POST \
    --header 'Content-Type: application/json' --data '{}' \
    "$WORKER_URL/v1/uploads/sign" || true)
if [ "$auth_status" != "401" ]; then
    err "unsigned /v1/uploads/sign returned HTTP ${auth_status:-transport_error}; expected 401"
    exit 1
fi
ok "Contributor Auth v1 sign route rejects unsigned requests"

printf '\n'
ok "Deploy complete: $WORKER_URL"
info "Checked-in Curate and Pro clients already use register -> signed reserve -> capability PUT -> signed completion."
info "Keep self-service revoke disabled until its signed owner mutation is implemented; admin containment remains available."
info "Optional: bind a custom domain in wrangler.toml and set TETHER_CONTRIBUTION_WORKER_URL for smoke tests."
