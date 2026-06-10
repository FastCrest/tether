"""Customer first-run dogfood test.

Pretends to be a first-time user who has only read the README.md of
tether. Uses NVIDIA's TRT container (as the README recommends for
GPU) and follows the quickstart verbatim:

  1. Install (pip install 'tether[serve,gpu] @ git+...')
  2. Explore CLI (tether --help, tether models, tether targets, tether doctor)
  3. Export (tether export lerobot/smolvla_base --target desktop --output ./smol)
  4. Serve (tether serve ./smol --port 8000 &)
  5. POST /act with the README's exact curl body

Records every stdout/stderr. Does NOT troubleshoot on the fly — the point
is to find the friction a real customer hits.

Usage:
    modal run scripts/modal_customer_dogfood.py
"""
import os
import subprocess
import modal

app = modal.App("tether-customer-dogfood")


def _repo_head_sha() -> str:
    """Return the current git HEAD SHA so the image rebuilds on every commit.
    Without this, Modal caches the image by `run_commands` string and a
    `pip install @ git+main` reuses the stale build. Discovered v5 2026-04-19."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ).decode().strip()[:12]
    except Exception:
        return "main"


_HEAD = _repo_head_sha()


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("huggingface")


# Per README: "Easiest path is NVIDIA's container:
#   docker run --gpus all -it nvcr.io/nvidia/tensorrt:24.10-py3"
image = (
    modal.Image.from_registry("nvcr.io/nvidia/tensorrt:24.10-py3", add_python="3.12")
    # clang: lerobot → evdev builds a C extension on install; TRT container
    # has gcc but NOT clang (evdev's build explicitly invokes clang).
    # Discovered by dogfood v3 2026-04-19.
    .apt_install("git", "curl", "clang")
    # Note: a naive customer would do what the README says here — we install
    # from the public git URL. If this fails, that's a real customer failure.
    .run_commands(
        # Customers don't get a pre-cloned repo; they install from the public URL.
        # README (post-fix) says: pip install 'tether[serve,gpu,monolithic] @ git+...'
        # SHA-pinning the install URL is what forces Modal to rebuild the image
        # when we commit — without this, the cached image from the previous dogfood
        # run is reused and the new fixes aren't actually tested.
        f'pip install "tether[serve,gpu,monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/FastCrest/tether@{_HEAD}"',
    )
)


def _run(cmd: str, timeout: int = 300, shell: bool = True):
    """Run a command and capture stdout/stderr/returncode/timing."""
    import subprocess
    import time
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_s": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout": (e.stdout or b"").decode(errors="replace") if e.stdout else "",
            "stderr": (e.stderr or b"").decode(errors="replace") if e.stderr else "",
            "elapsed_s": timeout,
            "timeout": True,
        }
    except Exception as e:
        return {
            "cmd": cmd,
            "returncode": -2,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.time() - t0, 1),
        }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[_hf_secret()],
)
def customer_first_run():
    """The full first-user journey, start to finish."""
    import json
    import os
    import subprocess
    import time

    transcript = []

    def log_step(name: str, note: str = ""):
        print(f"\n=== STEP: {name} ===", flush=True)
        if note:
            print(f"    [note] {note}", flush=True)

    def log_result(result: dict):
        transcript.append(result)
        rc = result["returncode"]
        elapsed = result["elapsed_s"]
        print(f"    $ {result['cmd']}", flush=True)
        print(f"    (exit={rc}, {elapsed}s)", flush=True)
        if result["stdout"]:
            stdout_preview = result["stdout"][:2000]
            for line in stdout_preview.splitlines()[:40]:
                print(f"    | {line}", flush=True)
            if len(result["stdout"]) > 2000:
                print(f"    | [... {len(result['stdout'])-2000} more stdout chars]", flush=True)
        if result["stderr"] and rc != 0:
            stderr_preview = result["stderr"][:1500]
            for line in stderr_preview.splitlines()[:20]:
                print(f"    !  {line}", flush=True)

    # ─── Step 0: sanity — what Python + pip do we have? ───
    log_step("0. environment baseline")
    log_result(_run("python --version"))
    log_result(_run("pip --version"))
    log_result(_run("which tether"))
    log_result(_run("tether --version", timeout=30))

    # ─── Step 1: naive CLI exploration (what a new user types) ───
    log_step("1. first CLI exploration — customer tries --help",
             "customer will type `tether --help` first")
    log_result(_run("tether --help", timeout=60))

    # `tether doctor` is called out in the README. Customer tries it.
    log_step("1b. tether doctor — README says run first if issues",
             "README: 'Something not working? Run `tether doctor` first'")
    log_result(_run("tether doctor", timeout=60))

    # `tether models` — README says "lists current support at any time"
    log_step("1c. tether models")
    log_result(_run("tether models", timeout=30))

    # `tether targets` — README says "lists current profiles"
    log_step("1d. tether targets")
    log_result(_run("tether targets", timeout=30))

    # ─── Step 2: export (verbatim from README quickstart) ───
    log_step("2. tether export lerobot/smolvla_base --target desktop --output ./smol",
             "exact command from README quickstart, step 2")
    # smolvla is smallest (1.6GB) → fastest customer experience
    export_dir = "/tmp/smol"
    log_result(_run(
        f"tether export lerobot/smolvla_base --target desktop --output {export_dir}",
        timeout=1800,
    ))

    # What's in the export directory? A customer might `ls` to see.
    log_step("2b. ls export directory — what got written?")
    log_result(_run(f"ls -la {export_dir}"))

    # Check VERIFICATION.md was written (per README's promise)
    log_step("2c. did VERIFICATION.md get written? (README promises it)")
    log_result(_run(f"test -f {export_dir}/VERIFICATION.md && echo YES || echo NO"))
    log_result(_run(f"head -30 {export_dir}/VERIFICATION.md 2>/dev/null || echo 'no file'"))

    # ─── Step 3: serve — start in background, give it time to warm up ───
    log_step("3. tether serve (from README: tether serve ./smol --port 8000)",
             "customer starts the HTTP server")
    # Start in background; we'll read from log file afterwards for the transcript
    server_log_path = "/tmp/serve.log"
    server_env = os.environ.copy()
    server_env["PYTHONUNBUFFERED"] = "1"
    log_fh = open(server_log_path, "wb")
    server_proc = subprocess.Popen(
        ["tether", "serve", export_dir, "--port", "8000", "--host", "127.0.0.1"],
        stdout=log_fh, stderr=subprocess.STDOUT,
        env=server_env,
    )

    # README claims "first tether serve takes ~30-90s to warm up" — give it 120s
    print("    [note] waiting up to 120s for server to be ready (per README warmup estimate)", flush=True)
    ready = False
    ready_after = None
    for i in range(120):
        time.sleep(1)
        if server_proc.poll() is not None:
            break  # process died
        # Try to hit /health — README doesn't mention /health but it exists
        try:
            probe = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "http://127.0.0.1:8000/health"],
                capture_output=True, text=True, timeout=3,
            )
            if probe.stdout.strip() == "200":
                ready = True
                ready_after = i + 1
                break
        except Exception:
            continue

    if ready:
        print(f"    [note] server ready after {ready_after}s", flush=True)
        transcript.append({
            "cmd": "(server boot)", "returncode": 0,
            "stdout": f"server ready after {ready_after}s",
            "stderr": "", "elapsed_s": ready_after,
        })
    else:
        # Tail the log to see what happened
        log_fh.close()
        tail = ""
        try:
            with open(server_log_path, "rb") as f:
                f.seek(0, 2); size = f.tell()
                f.seek(max(0, size - 3000))
                tail = f.read().decode(errors="replace")
        except Exception as e:
            tail = f"(couldn't read log: {e})"
        print(f"    [ERROR] server did not become ready in 120s", flush=True)
        print("    --- server log tail ---", flush=True)
        for line in tail.splitlines()[-30:]:
            print(f"    SRV| {line}", flush=True)
        transcript.append({
            "cmd": "(server boot)", "returncode": -1,
            "stdout": "", "stderr": tail[-1500:], "elapsed_s": 120,
        })
        # Bail — can't test /act if server isn't up
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            pass
        return {"transcript": transcript, "conclusion": "FAIL: server never ready"}

    # ─── Step 4: POST /act — verbatim from README's curl example ───
    log_step("4. POST /act (README curl body)",
             "exact curl from README: instruction='pick up the red cup', state=[0.1..0.6]")

    # Simulating the exact README curl
    readme_curl = (
        "curl -s -X POST http://127.0.0.1:8000/act "
        "-H 'content-type: application/json' "
        "-d '{\"instruction\":\"pick up the red cup\",\"state\":[0.1,0.2,0.3,0.4,0.5,0.6]}'"
    )
    act_result = _run(readme_curl, timeout=60)
    log_result(act_result)

    # Did the response have the fields the README promises?
    # README shows: actions, num_actions, latency_ms, denoising_steps, inference_mode
    readme_expected_fields = ["actions", "num_actions", "latency_ms", "denoising_steps", "inference_mode"]
    try:
        body = json.loads(act_result["stdout"]) if act_result["stdout"] else {}
        missing = [f for f in readme_expected_fields if f not in body]
        present = [f for f in readme_expected_fields if f in body]
        print(f"    [fields-check] present: {present}", flush=True)
        if missing:
            print(f"    [fields-check] MISSING (README promises these): {missing}", flush=True)
        if "actions" in body:
            actions = body["actions"]
            shape = (len(actions), len(actions[0]) if actions else 0)
            print(f"    [shape-check] action chunk shape: {shape}", flush=True)
        transcript.append({
            "cmd": "(/act response field check)",
            "returncode": 0 if not missing else 1,
            "stdout": json.dumps({
                "present": present, "missing": missing,
                "keys_actually_in_response": list(body.keys()),
            }, indent=2),
            "stderr": "", "elapsed_s": 0,
        })
    except Exception as e:
        print(f"    [fields-check] couldn't parse response as JSON: {e}", flush=True)

    # ─── Step 5: try the canonical "is it on the right provider?" question ───
    # A customer who paid attention to the README might check that inference_mode
    # reflects actual GPU use (not a silent CPU fallback).
    log_step("5. verify inference_mode tells us what provider is in use",
             "customer sanity check: are we on GPU?")
    try:
        body = json.loads(act_result["stdout"]) if act_result.get("stdout") else {}
        mode = body.get("inference_mode", "?")
        print(f"    inference_mode reported: {mode}", flush=True)
        transcript.append({
            "cmd": "(inference_mode check)", "returncode": 0,
            "stdout": f"inference_mode={mode}",
            "stderr": "", "elapsed_s": 0,
        })
    except Exception as e:
        print(f"    couldn't introspect: {e}", flush=True)

    # ─── Shutdown ───
    try:
        server_proc.terminate()
        server_proc.wait(timeout=10)
    except Exception:
        server_proc.kill()
    log_fh.close()

    # Final tail of server log (always useful — shows startup timing etc.)
    try:
        with open(server_log_path, "rb") as f:
            server_log = f.read().decode(errors="replace")
    except Exception:
        server_log = "(could not read server log)"

    return {
        "transcript": transcript,
        "server_log_tail": server_log[-4000:],
    }


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("REFLEX CUSTOMER DOGFOOD — first-user journey")
    print("=" * 60)
    result = customer_first_run.remote()
    print("\n" + "=" * 60)
    print("TRANSCRIPT (structured)")
    print("=" * 60)
    for step in result["transcript"]:
        print(f"\n$ {step['cmd']}")
        print(f"  exit={step.get('returncode', '?')}  elapsed={step.get('elapsed_s', '?')}s")

    print("\n\n=== SERVER LOG (last 4000 chars) ===")
    print(result.get("server_log_tail", "(none)"))
