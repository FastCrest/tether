#!/usr/bin/env bash
# Sanity check for per-embodiment configs (B.1).
#
# Verifies:
#   - 3 preset JSON files exist at configs/embodiments/{franka,so100,ur5}.json
#   - each parses as valid JSON
#   - the embodiments package imports + load_preset works for all 3
#
# Run from repo root:
#   bash scripts/verify_embodiment_structure.sh
#
# Used as a CI pre-flight before pytest. Exit 0 = pass, 1 = fail.

set -euo pipefail

cd "$(dirname "$0")/.."

PRESETS=(franka so100 ur5)

echo "[verify-embodiment-structure] checking JSON files..."
for robot in "${PRESETS[@]}"; do
  path="configs/embodiments/${robot}.json"
  if [[ ! -f "$path" ]]; then
    echo "  ✗ MISSING: $path"
    echo "    Run: python scripts/emit_embodiment_presets.py"
    exit 1
  fi
  if ! python3 -m json.tool "$path" > /dev/null 2>&1; then
    echo "  ✗ INVALID JSON: $path"
    exit 1
  fi
  echo "  ✓ $path"
done

echo ""
echo "[verify-embodiment-structure] checking Python imports..."
PYTHON="${PYTHON:-python3}"
PYTHONPATH="${PYTHONPATH:-src}" "$PYTHON" - <<'PY'
import sys
from tether.embodiments import EmbodimentConfig, list_presets
from tether.embodiments.validate import validate_embodiment_config

presets = list_presets()
expected = ["franka", "so100", "ur5"]
if presets != expected:
    print(f"  ✗ list_presets() returned {presets}, expected {expected}")
    sys.exit(1)

for name in presets:
    cfg = EmbodimentConfig.load_preset(name)
    ok, errors = validate_embodiment_config(cfg)
    blocking = [e for e in errors if e["severity"] == "error"]
    if blocking:
        print(f"  ✗ {name}: validation errors:")
        for e in blocking:
            print(f"      [{e['slug']}] {e['field']}: {e['message']}")
        sys.exit(1)
    print(f"  ✓ {name} (action_dim={cfg.action_dim}, state_dim={cfg.state_dim})")

print("")
print("[verify-embodiment-structure] all checks passed.")
PY
