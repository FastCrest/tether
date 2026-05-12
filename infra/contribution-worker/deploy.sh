#!/usr/bin/env bash
#
# Reflex contribution-worker — one-command deploy.
#
# What this script does:
#   1. Installs wrangler globally if missing
#   2. Runs `wrangler login` (opens browser for OAuth) if not authenticated
#   3. Creates the D1 database `reflex-contributions`
#   4. Creates the R2 bucket `reflex-curate`
#   5. Patches wrangler.toml with the new database_id
#   6. Applies schema.sql to the D1 database
#   7. Generates a strong ADMIN_TOKEN and sets it as a Worker Secret
#   8. Deploys the worker
#   9. Smoke-tests /healthz
#  10. Prints the worker URL + next steps
#
# Idempotent: safe to re-run if a step fails. Skips already-completed steps.

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { printf "${GREEN}\xe2\x9c\x93${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}\xe2\x9a\xa0${NC} %s\n" "$*"; }
err()  { printf "${RED}\xe2\x9c\x97${NC} %s\n" "$*" >&2; }
info() { printf "${CYAN}\xe2\x86\x92${NC} %s\n" "$*"; }

cd "$(dirname "$0")"

# 1. wrangler install
if ! command -v wrangler >/dev/null 2>&1; then
    info "Installing wrangler globally (npm install -g wrangler)..."
    npm install -g wrangler
    ok "wrangler installed: $(wrangler --version)"
else
    ok "wrangler already installed: $(wrangler --version)"
fi

# 2. wrangler login
if ! wrangler whoami >/dev/null 2>&1; then
    info "Running wrangler login (opens browser)..."
    wrangler login
fi
ok "wrangler authenticated"

# 3. D1 database
DB_NAME="reflex-contributions"
existing_id=$(wrangler d1 list --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for db in data:
    if db.get('name') == '$DB_NAME':
        print(db.get('uuid', ''))
        break
" 2>/dev/null || echo "")
if [ -n "$existing_id" ]; then
    DB_ID="$existing_id"
    ok "D1 database $DB_NAME already exists (id=$DB_ID)"
else
    info "Creating D1 database $DB_NAME..."
    create_out=$(wrangler d1 create "$DB_NAME")
    DB_ID=$(echo "$create_out" | grep -oE 'database_id = "[a-f0-9-]+"' | head -1 | sed 's/.*"\([^"]*\)"/\1/')
    if [ -z "$DB_ID" ]; then
        err "Failed to extract database_id from wrangler output"
        echo "$create_out" >&2
        exit 1
    fi
    ok "D1 database created: id=$DB_ID"
fi

# 4. R2 bucket
BUCKET_NAME="reflex-curate"
if wrangler r2 bucket list 2>/dev/null | grep -qE "(^|[[:space:]])${BUCKET_NAME}([[:space:]]|$)"; then
    ok "R2 bucket $BUCKET_NAME already exists"
else
    info "Creating R2 bucket $BUCKET_NAME..."
    wrangler r2 bucket create "$BUCKET_NAME"
    ok "R2 bucket created: $BUCKET_NAME"
fi

# 5. Patch wrangler.toml
if grep -q 'database_id = "REPLACE_AFTER_CREATE"' wrangler.toml; then
    info "Patching wrangler.toml with database_id=$DB_ID"
    sed -i.bak "s/REPLACE_AFTER_CREATE/$DB_ID/" wrangler.toml
    ok "wrangler.toml updated"
else
    ok "wrangler.toml already has a database_id"
fi

# 6. Apply schema
info "Applying schema.sql to remote D1..."
wrangler d1 execute "$DB_NAME" --file=./schema.sql --remote
ok "Schema applied"

# 7. Set ADMIN_TOKEN
if wrangler secret list 2>/dev/null | grep -q '"name": "ADMIN_TOKEN"'; then
    ok "ADMIN_TOKEN already set (skipping)"
else
    ADMIN_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    info "Generated new ADMIN_TOKEN"
    warn "Save this admin token NOW \xe2\x80\x94 it will not be shown again:"
    echo
    printf "    ${YELLOW}%s${NC}\n" "$ADMIN_TOKEN"
    echo
    info "Setting as Worker Secret..."
    echo "$ADMIN_TOKEN" | wrangler secret put ADMIN_TOKEN
    ok "ADMIN_TOKEN set"
fi

# 8. Deploy
info "Deploying worker..."
wrangler deploy
ok "Worker deployed"

# 9. Smoke test
WORKER_URL=$(wrangler deployments list 2>/dev/null | grep -oE 'https://[a-zA-Z0-9-]+\.workers\.dev' | head -1 || echo "")
if [ -n "$WORKER_URL" ]; then
    info "Smoke testing $WORKER_URL/healthz..."
    if curl -fsS "$WORKER_URL/healthz" >/dev/null; then
        ok "/healthz returned 200"
    else
        warn "/healthz failed \xe2\x80\x94 check 'wrangler tail'"
    fi
else
    warn "Could not auto-detect worker URL; check the deploy output above"
fi

echo
ok "Deploy complete."
echo
info "Next steps:"
echo "    1. Update src/reflex/curate/uploader.py:_request_signed_url + _put_to_r2"
echo "       to make real httpx POST/PUT calls (replace UploadStub)."
echo "    2. Flip the curate uploader's live=False to live=True in"
echo "       src/reflex/runtime/server.py."
echo "    3. Update src/reflex/curate/opt_in_cli.py:_cmd_revoke to POST to"
echo "       \$WORKER_URL/v1/revoke/cascade."
echo "    4. (Optional) bind a custom domain in wrangler.toml + redeploy."
