import assert from "node:assert/strict";
import test from "node:test";

import worker from "../worker.js";

for (const path of ["/v1/episodes/upload", "/v1/episodes/upload-url"]) {
  test(`${path} is permanently retired without touching storage`, async () => {
    let touched = false;
    const env = {
      BUCKET: { put: async () => { touched = true; } },
      DB: { prepare: () => { touched = true; throw new Error("unexpected DB access"); } },
    };
    const response = await worker.fetch(new Request(`https://data.test${path}`, {
      method: "POST",
      headers: {
        "X-Anonymized": "true",
        "X-Contributor-Hash": "0123456789abcdef",
      },
      body: "untrusted payload",
    }), env, {});

    assert.equal(response.status, 410);
    const payload = await response.json();
    assert.equal(payload.error, "upload_endpoint_retired");
    assert.equal(payload.service, "contribution-worker");
    assert.equal(payload.sign_endpoint, "/v1/uploads/sign");
    assert.equal(touched, false);
  });
}

test("legacy contributor stats defer to the sole authenticated authority", async () => {
  const response = await worker.fetch(new Request(
    "https://data.test/v1/contributor/0123456789abcdef/stats",
  ), { DB: { prepare: () => { throw new Error("must remain read-free"); } } }, {});
  assert.equal(response.status, 410);
  assert.equal((await response.json()).service, "contribution-worker");
});

test("legacy global stats defer to the sole authority", async () => {
  const response = await worker.fetch(
    new Request("https://data.test/v1/stats"),
    { DB: { prepare: () => { throw new Error("must remain read-free"); } } },
    {},
  );
  assert.equal(response.status, 410);
  assert.equal((await response.json()).service, "contribution-worker");
});
