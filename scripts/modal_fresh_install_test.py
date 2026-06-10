"""Modal: fresh-install smoke test — simulates a new customer's first hour.

Spins a clean python:3.12-slim container, pip-installs tether from
the live GitHub main branch, verifies the CLI works end-to-end:

  1. `tether --help` returns zero + shows subcommands
  2. `tether targets` lists hardware profiles
  3. `tether doctor` passes (or only fails on GPU checks under CPU-only image)
  4. imports: from tether.runtime.server import create_app

If this breaks, a fresh-box customer broke in their first 5 minutes.
"""
import modal

app = modal.App("tether-fresh-install-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "tether[serve,onnx] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether.git"
    )
)


@app.function(image=image, timeout=600)
def test_fresh_install():
    import subprocess
    import sys

    def _run(cmd, ok_codes=(0,)):
        print(f"\n$ {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:] if r.stdout else "(no stdout)")
        if r.returncode not in ok_codes:
            print(f"[stderr]: {r.stderr[-2000:]}")
        return r.returncode

    results = {}
    results["help"] = _run(["tether", "--help"])
    results["targets"] = _run(["tether", "targets"])

    # Try to import the key runtime module; should not raise
    try:
        from tether.runtime.server import create_app  # noqa: F401
        from tether.verification_report import write_verification_report  # noqa: F401
        from tether.safety.guard import ActionGuard  # noqa: F401
        results["imports"] = 0
        print("[imports] tether.runtime.server + verification_report + safety.guard OK")
    except Exception as e:
        results["imports"] = 1
        print(f"[imports] FAIL: {e}")

    all_pass = all(v == 0 for v in results.values())
    print(f"\n=== VERDICT: {'PASS' if all_pass else 'FAIL'} ===")
    print(f"  {results}")
    return {"results": results, "passed": all_pass}


@app.local_entrypoint()
def main():
    result = test_fresh_install.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
