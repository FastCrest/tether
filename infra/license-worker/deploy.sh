#!/usr/bin/env bash
#
# Reflex Pro license worker — one-command deploy.
#
# Run from your terminal where you have:
#   - npm available (already installed if you use Node.js)
#   - A Cloudflare account (free tier is enough)
#   - A web browser for the wrangler OAuth handshake
#
# What this script does:
#   1. Installs wrangler globally if missing
#   2. Runs `wrangler login` (opens your browser for OAuth)
#   3. Creates the D1 database `reflex-licenses`
#   4. Patches wrangler.toml with the new database_id
#   5. Applies the schema
#   6. Generates a strong ADMIN_TOKEN, prints it ONCE, then sets it as a Worker Secret
#   7. Optionally sets SLACK_WEBHOOK_URL
#   8. Deploys the worker
#   9. Calls /admin/init to generate the Ed25519 keypair
#   10. Sets PRIVATE_KEY as a Worker Secret
#   11. Prints the public_key_b64 + key_id you need to paste into
#       src/reflex/pro/_public_key.py + the worker URL for src/reflex/pro/activate.py
#
# After this script: edit those two Python constants, commit + push, and your
# license server is live.
#
# Idempotent: safe to re-run if a step fails. Skips already-completed steps
# where it can detect them.

set -euo pipefail

# ─── colors ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${NC} %s\n" "$*"; }
err()  { printf "${RED}✗${NC} %s\n" "$*" >&2; }
info() { printf "${CYAN}→${NC} %s\n" "$*"; }

cd "$(dirname "$0")"

# ─── 1. wrangler install ─────────────────────────────────────────────────────
if ! command -v wrangler >/dev/null 2>&1; then
    info "Installing wrangler globally (npm install -g wrangler)..."
    npm install -g wrangler
    ok "wrangler installed: $(wrangler --version)"
else
    ok "wrangler already installed: $(wrangler --version)"
fi

# ─── 2. wrangler login ───────────────────────────────────────────────────────
if ! wrangler whoami 2>/dev/null | grep -q "@"; then
    info "Opening browser for Cloudflare OAuth (wrangler login)..."
    wrangler login
fi
ACCOUNT=$(wrangler whoami 2>/dev/null | grep -E "^(Email|Account ID)" || true)
ok "Authenticated. $(echo "$ACCOUNT" | head -1)"

# ─── 3. D1 database create ───────────────────────────────────────────────────
DB_NAME="reflex-licenses"
EXISTING_DB=$(wrangler d1 list 2>/dev/null | grep -E "^│ +${DB_NAME}" || true)
if [ -n "$EXISTING_DB" ]; then
    DB_ID=$(echo "$EXISTING_DB" | awk -F'│' '{print $3}' | tr -d ' ')
    ok "D1 database ${DB_NAME} already exists (id=${DB_ID})"
else
    info "Creating D1 database ${DB_NAME}..."
    CREATE_OUT=$(wrangler d1 create "$DB_NAME" 2>&1)
    DB_ID=$(echo "$CREATE_OUT" | grep -oE 'database_id = "[^"]+"' | sed 's/database_id = "//;s/"$//' | head -1)
    if [ -z "$DB_ID" ]; then
        err "Could not extract database_id from wrangler output:"
        echo "$CREATE_OUT"
        exit 1
    fi
    ok "Created D1 database (id=${DB_ID})"
fi

# ─── 4. Patch wrangler.toml ──────────────────────────────────────────────────
if grep -q "REPLACE_WITH_DATABASE_ID_FROM_WRANGLER_D1_CREATE" wrangler.toml; then
    info "Writing database_id into wrangler.toml..."
    # Cross-platform sed (BSD on macOS, GNU on Linux): use a backup file
    sed -i.bak "s/REPLACE_WITH_DATABASE_ID_FROM_WRANGLER_D1_CREATE/${DB_ID}/" wrangler.toml
    rm -f wrangler.toml.bak
    ok "wrangler.toml updated with database_id"
else
    if grep -q "database_id = \"${DB_ID}\"" wrangler.toml; then
        ok "wrangler.toml already has correct database_id"
    else
        warn "wrangler.toml has a different database_id than the one we just got. Manual review needed."
    fi
fi

# ─── 5. Apply schema ─────────────────────────────────────────────────────────
info "Applying schema to ${DB_NAME}..."
wrangler d1 execute "$DB_NAME" --remote --file=schema.sql || {
    warn "Schema apply returned non-zero (may be re-run on existing tables, that's OK)"
}
ok "Schema applied"

# ─── 6. ADMIN_TOKEN ──────────────────────────────────────────────────────────
if wrangler secret list 2>/dev/null | grep -q '"ADMIN_TOKEN"'; then
    ok "ADMIN_TOKEN already set as Worker Secret"
    warn "If you've lost it, delete with: wrangler secret delete ADMIN_TOKEN — then re-run this script"
else
    info "Generating ADMIN_TOKEN..."
    ADMIN_TOKEN=$(openssl rand -base64 32 | tr -d '\n=' | tr '+/' '-_')
    printf "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    printf "${YELLOW}Your new ADMIN_TOKEN (SAVE THIS — printed only once):${NC}\n"
    printf "  ${CYAN}${ADMIN_TOKEN}${NC}\n"
    printf "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"
    printf "Press ENTER after copying the token to a password manager..."
    read -r _
    info "Setting ADMIN_TOKEN as Worker Secret..."
    echo -n "$ADMIN_TOKEN" | wrangler secret put ADMIN_TOKEN
    ok "ADMIN_TOKEN set"
    printf "\n${YELLOW}Set this in your shell for the admin CLI:${NC}\n"
    printf "  export REFLEX_ADMIN_TOKEN='${ADMIN_TOKEN}'\n\n"
fi

# ─── 7. SLACK_WEBHOOK_URL (optional) ─────────────────────────────────────────
if wrangler secret list 2>/dev/null | grep -q '"SLACK_WEBHOOK_URL"'; then
    ok "SLACK_WEBHOOK_URL already set"
else
    printf "\n"
    read -r -p "Set SLACK_WEBHOOK_URL for new-license + revoke + sharing alerts? (paste URL or leave blank to skip): " SLACK_URL
    if [ -n "$SLACK_URL" ]; then
        echo -n "$SLACK_URL" | wrangler secret put SLACK_WEBHOOK_URL
        ok "SLACK_WEBHOOK_URL set"
    else
        info "Skipped SLACK_WEBHOOK_URL"
    fi
fi

# ─── 8. Deploy the worker ────────────────────────────────────────────────────
info "Deploying worker..."
DEPLOY_OUT=$(wrangler deploy 2>&1)
echo "$DEPLOY_OUT" | tail -10
WORKER_URL=$(echo "$DEPLOY_OUT" | grep -oE 'https://[a-zA-Z0-9._-]+\.workers\.dev' | head -1)
if [ -z "$WORKER_URL" ]; then
    err "Could not extract worker URL from deploy output. Check above and use the URL manually."
    exit 1
fi
ok "Worker deployed at: ${WORKER_URL}"

# ─── 9. /admin/init ──────────────────────────────────────────────────────────
# Skip init if we don't have the ADMIN_TOKEN locally (it was already set as a
# Secret on a prior run and we don't have the value anymore).
if [ -z "${ADMIN_TOKEN:-}" ]; then
    if [ -n "${REFLEX_ADMIN_TOKEN:-}" ]; then
        ADMIN_TOKEN="$REFLEX_ADMIN_TOKEN"
        info "Using REFLEX_ADMIN_TOKEN from env for /admin/init"
    else
        warn "ADMIN_TOKEN not in this shell. Run /admin/init manually:"
        echo
        echo "  curl -X POST ${WORKER_URL}/admin/init -H \"Authorization: Bearer \$REFLEX_ADMIN_TOKEN\""
        echo
        echo "Then paste the public_key_b64 into src/reflex/pro/_public_key.py"
        exit 0
    fi
fi

info "Calling /admin/init to generate Ed25519 keypair..."
INIT_OUT=$(curl -sS -X POST "${WORKER_URL}/admin/init" \
    -H "Authorization: Bearer ${ADMIN_TOKEN}" \
    -H "Content-Type: application/json")

if echo "$INIT_OUT" | grep -q "key_already_exists"; then
    ok "Keypair already exists (skipping init). To rotate, see Phase 2 docs."
    PUBKEY_OUT=$(curl -sS "${WORKER_URL}/v1/pubkey")
    PUBKEY_B64=$(echo "$PUBKEY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['public_key_b64'])")
    KEY_ID=$(echo "$PUBKEY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['key_id'])")
elif echo "$INIT_OUT" | grep -q "keypair_generated"; then
    PUBKEY_B64=$(echo "$INIT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['public_key_b64'])")
    PRIVKEY_B64=$(echo "$INIT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['private_key_b64'])")
    KEY_ID=$(echo "$INIT_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['key_id'])")
    ok "Keypair generated (id=${KEY_ID})"

    # ─── 10. Set PRIVATE_KEY ──────────────────────────────────────────────────
    info "Setting PRIVATE_KEY as Worker Secret (this is the only place it goes — never persisted elsewhere)..."
    echo -n "$PRIVKEY_B64" | wrangler secret put PRIVATE_KEY
    ok "PRIVATE_KEY set as Worker Secret"
    unset PRIVKEY_B64
else
    err "Unexpected response from /admin/init:"
    echo "$INIT_OUT"
    exit 1
fi

# ─── 11. Print next-steps with values to paste ───────────────────────────────
PUBKEY_FILE="../../src/reflex/pro/_public_key.py"
ACTIVATE_FILE="../../src/reflex/pro/activate.py"

printf "\n\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
printf "${GREEN}DEPLOY COMPLETE${NC}\n"
printf "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n\n"

printf "${YELLOW}Worker URL:${NC} ${WORKER_URL}\n"
printf "${YELLOW}Key ID:${NC}     ${KEY_ID}\n"
printf "${YELLOW}Public key:${NC} ${PUBKEY_B64}\n\n"

printf "${CYAN}Next steps (3 manual edits, ~2 min):${NC}\n\n"

printf "1. Update ${PUBKEY_FILE}:\n"
printf "   BUNDLED_PUBLIC_KEY_B64 = \"${PUBKEY_B64}\"\n"
printf "   BUNDLED_KEY_ID         = \"${KEY_ID}\"\n\n"

printf "2. Update ${ACTIVATE_FILE}:\n"
printf "   DEFAULT_LICENSE_ENDPOINT = \"${WORKER_URL}\"\n\n"

printf "3. Commit + push:\n"
printf "   git add ${PUBKEY_FILE} ${ACTIVATE_FILE}\n"
printf "   git commit -m \"pro: wire bundled pubkey + license endpoint after first deploy\"\n"
printf "   git push\n\n"

printf "${CYAN}Then issue your first license:${NC}\n\n"
printf "   export REFLEX_LICENSE_ENDPOINT='${WORKER_URL}'\n"
printf "   export REFLEX_ADMIN_TOKEN='<your_admin_token>'\n\n"
printf "   python -m tether.admin.issue_license \\\\\n"
printf "       --customer-id alice@bigco.com \\\\\n"
printf "       --tier pro \\\\\n"
printf "       --expires-in 30\n\n"

printf "${CYAN}Customer redeems with:${NC}\n\n"
printf "   pip install --upgrade tether\n"
printf "   tether pro activate REFLEX-XXXX-XXXX-XXXX\n"
printf "   tether pro status\n\n"

ok "Done."
