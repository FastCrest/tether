"""Bundled Ed25519 public key for offline license verification.

After running POST /admin/init on the deployed license worker, paste the
``public_key_b64`` field from the response into ``BUNDLED_PUBLIC_KEY_B64``
below, then commit + release a new package version. Customers verify
license signatures against this key on every load (offline; no network
call required for signature verification — only the heartbeat needs the
network, and only daily).

Key rotation is represented as an allowlisted mapping. Releases add the next
public key before the Worker switches signers, keep both keys during the
overlap, and remove the retired key only in a later release.

The PUBLIC key in this file is intentional and safe to publish — that's
what public keys are for. The PRIVATE key lives only in the Cloudflare
Worker's PRIVATE_KEY secret and never appears in this codebase.
"""
from __future__ import annotations

# Public key bundled at deploy time (license worker first ran /admin/init
# 2026-05-03). To rotate: regenerate at the worker and replace these constants.
BUNDLED_PUBLIC_KEY_B64 = "luURwH5bpH5qHc7eTa3xyCiTc4X6cqXzunzw0bCeSzw="

# Key ID of the bundled public key. Used to verify the signature was made
# with a key the client knows about (rejects licenses signed by a different
# deployment, e.g., a forked or compromised license server).
BUNDLED_KEY_ID = "key_moq2zo8m_279ec0def41c69b8"

# Trusted signing keys for licenses and heartbeat attestations. Never fetch
# this set from the network at verification time: trust is established by the
# signed package release, not by the server being checked.
TRUSTED_PUBLIC_KEYS_B64 = {
    BUNDLED_KEY_ID: BUNDLED_PUBLIC_KEY_B64,
}
