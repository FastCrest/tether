"""Bundled Ed25519 public key for offline license verification.

After running POST /admin/init on the deployed license worker, paste the
``public_key_b64`` field from the response into ``BUNDLED_PUBLIC_KEY_B64``
below, then commit + release a new package version. Customers verify
license signatures against this key on every load (offline; no network
call required for signature verification — only the heartbeat needs the
network, and only daily).

Phase 2 will support multiple trusted keys (current + previous N) for key
rotation. Today we ship a single bundled key.

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
