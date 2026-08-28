import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";

function select(results, env = {}) {
  return spawnSync("bash", ["deploy.sh", "--select-active-key"], {
    cwd: new URL("..", import.meta.url),
    env: { ...process.env, TETHER_SIGNING_KEY_ID: "", ...env },
    input: JSON.stringify([{ results, success: true }]),
    encoding: "utf8",
  });
}

test("deploy helper distinguishes fresh install from one existing active key", () => {
  const fresh = select([]);
  assert.equal(fresh.status, 3);
  assert.equal(fresh.stdout, "");

  const existing = select([{ key_id: "key_existing", public_key_b64: "cHVi" }]);
  assert.equal(existing.status, 0, existing.stderr);
  assert.equal(existing.stdout.trim(), "key_existing\tcHVi");
});

test("deploy helper refuses ambiguous rotation rows unless id is explicit", () => {
  const rows = [
    { key_id: "key_old", public_key_b64: "b2xk" },
    { key_id: "key_new", public_key_b64: "bmV3" },
  ];
  const ambiguous = select(rows);
  assert.equal(ambiguous.status, 4);
  assert.match(ambiguous.stderr, /multiple active D1 keys/);

  const selected = select(rows, { TETHER_SIGNING_KEY_ID: "key_new" });
  assert.equal(selected.status, 0, selected.stderr);
  assert.equal(selected.stdout.trim(), "key_new\tbmV3");
});

test("existing signer pre-bind precedes deploy and fresh signer setup is verified", () => {
  const script = readFileSync(new URL("../deploy.sh", import.meta.url), "utf8");
  const deploy = script.indexOf('info "Deploying worker..."');
  const prebind = script.indexOf('echo -n "$KEY_ID" | wrangler secret put SIGNING_KEY_ID');
  const freshBind = script.lastIndexOf('echo -n "$KEY_ID" | wrangler secret put SIGNING_KEY_ID');
  const verify = script.indexOf('${WORKER_URL}/admin/signer');
  assert.ok(prebind !== -1 && prebind < deploy);
  assert.ok(freshBind > deploy);
  assert.ok(verify > freshBind);
});
