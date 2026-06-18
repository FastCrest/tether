# Security Policy

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **hello@fastcrest.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce or proof-of-concept (as detailed as possible)
- Any suggested fix, if you have one

You will receive an acknowledgement within 48 hours and a status update within 7 days. If the issue is confirmed, we will coordinate a fix and disclosure timeline with you before publishing anything publicly.

## Supported Versions

Security fixes are applied to the **latest minor release** only. We do not backport fixes to older minor versions.

| Version | Supported |
|---------|-----------|
| latest minor | yes |
| older minors | no |

## Scope

Tether serves robot-control endpoints over the network. Issues in the following areas are taken seriously and should be reported promptly:

- **Network-exposed API endpoints** (`tether serve` HTTP/ZMQ surfaces) — authentication bypass, SSRF, injection, denial of service
- **Model / checkpoint loading** — path traversal, arbitrary code execution via crafted model files
- **License and telemetry workers** — data leakage, auth bypass
- **Dependency vulnerabilities** that affect the runtime serve path

Issues limited to local-only attack surfaces (e.g. a user who already has shell access to the serve host) are lower priority but still welcome.
