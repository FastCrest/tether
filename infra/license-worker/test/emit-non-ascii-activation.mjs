import worker from "../worker.js";

function b64(buffer) {
  return Buffer.from(buffer).toString("base64");
}

const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
const privatePkcs8 = await crypto.subtle.exportKey("pkcs8", pair.privateKey);
const publicRaw = await crypto.subtle.exportKey("raw", pair.publicKey);
const expiresAt = new Date(Date.now() + 86400000).toISOString();
const DB = {
  prepare(sql) {
    return {
      bind() { return this; },
      async first() {
        if (sql.includes("FROM activation_codes")) {
          return { license_id: "lic_unicode_live", expires_at: expiresAt, used: 0 };
        }
        if (sql.includes("SELECT license_json")) {
          return { license_json: JSON.stringify({
            license_version: 2,
            license_id: "lic_unicode_live",
            customer_id: "客户 É",
            tier: "pro",
            issued_at: new Date().toISOString(),
            expires_at: expiresAt,
            max_seats: 1,
            hardware_binding: null,
          }) };
        }
        if (sql.includes("FROM master_keys")) {
          return { public_key_b64: b64(publicRaw) };
        }
        return null;
      },
      async run() { return { success: true, meta: { changes: 1 } }; },
    };
  },
};
const env = {
  DB,
  PRIVATE_KEY: b64(privatePkcs8),
  SIGNING_KEY_ID: "key_unicode_live",
};
const response = await worker.fetch(new Request(
  "https://worker.test/v1/activation/REFLEX-AAAA-BBBB-CCCC",
  {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      hardware_binding: {
        gpu_uuid: "GPU-一",
        gpu_name: "Éclair GPU",
        cpu_count: 8,
      },
    }),
  },
), env);
if (response.status !== 200) throw new Error(await response.text());
process.stdout.write(JSON.stringify({
  public_key_b64: b64(publicRaw),
  response: await response.json(),
}));
