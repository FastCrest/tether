# GPU loader and Doctor follow-up

Customer: a native WSL2 Ubuntu 24.04 operator exporting in one Python environment and serving an ONNX policy in another. The remaining job is a coherent serving dependency install and truthful diagnostics when GPU libraries are installed through pip or system paths. No device qualification is implied by these source checks.

## Causes and working hypothesis

- `doctor(model=...)` assigned its temporary ONNX ModelProto to `model`, replacing the optional export directory. The same branch could trigger deployment diagnostics even without `--model`. This assignment came from the earlier empirical TensorRT probe, introduced in 1f942497. The new regression reproduced the wrong value before the fix, including a JSON serialization failure from the substituted ModelProto test double.
- The old loader searched only sys.prefix and /usr/local Python directories, targeted CUDA 12 SONAMEs, and stopped after the first existing file even if dlopen failed. The old Doctor equated an absent pip-directory file with an uninstalled library and suppressed its empirical session probe. The source already included the standard tensorrt_libs directory; that directory was not simply omitted.
- Unbounded onnxruntime-gpu and generic tensorrt requirements could select CUDA 13 while explicit NVIDIA libraries remained CUDA 12. The base package also requires torch, so the serving extras need a compatible torch bound. Installing the prior serve and gpu extras together requested both CPU and GPU ORT distributions, which share Python module files.

## Change and evidence

The probe uses `probe_model`, preserving the user's path in both human and JSON modes. Library discovery covers active sys.path, site/sysconfig roots, explicit LD_LIBRARY_PATH, pip legacy/consolidated NVIDIA layouts, TensorRT layouts, and normal system paths. Failed candidates retain their concrete dlopen errors and allow later candidates. OS-cache results must match the requested SONAME major. Automatic preloading and path mutation skip mixed or unidentified CUDA families; Doctor still attempts the independent ORT session probe. Standard system library directories are searched but not prepended to child loader paths.

Doctor reports loadability evidence rather than claiming a missing pip file proves absence. Its tiny TensorRT session no longer predicts a production speedup. The unsupported cuDNN-minor-to-driver heuristic and Linux apt driver recommendation for WSL were replaced by observed driver facts and host-driver guidance.

The Linux x86_64 CUDA 12 serving cohort is ORT 1.25.1, torch 2.8.0, cudart 12.8.90, cuBLAS 12.8.4.1, cuDNN 9.10.2.21, cuda-toolkit 12.8.1 and TensorRT cu12 10.14.1.48.post1. The torch bound applies only when selecting gpu/gpu-min on Linux x86_64. The separate export extra remains unconstrained by this serving bound. Serve is now HTTP/transports only; select serve,onnx for CPU or serve,gpu for CUDA.

Sources used for dependency decisions:

- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html documents the PyPI CUDA 13 default starting at ORT 1.27 and CUDA/cuDNN major requirements.
- https://github.com/microsoft/onnxruntime/blob/v1.25.0/tools/ci_build/github/linux/docker/Dockerfile.manylinux2_28_cuda uses CUDA 12.8.1 with TensorRT 10.14.1.48.
- https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-pip.html documents explicit CUDA-major TensorRT packages.
- Exact PyPI JSON metadata for torch 2.8.0, cuda-toolkit 12.8.1, and TensorRT cu12 10.14.1.48.post1 supplied the dependency constraints. The postrelease includes Linux bindings; the base 10.14.1.48 binding release does not. Version-specific static metadata plus an explicit NVIDIA index allows resolution without executing the TensorRT library download shim. NVIDIA library/binding packages are direct marked requirements so their index mapping applies.

## Validation

Focused source suite: 103 passed, 1 platform skip in 1.35 seconds. Files: test_gpu_loader.py, test_doctor_model_path_followup.py, test_doctor_trt_ep.py, test_doctor_guards.py, test_pyproject_metadata.py, test_install_smoke.py, test_doctor_diagnostics.py. New tests cover path preservation with and without --model, human/JSON, a failed first library followed by a usable candidate, system-cache libraries without pip files, mixed-family refusal, CUDA 13 layout/SONAME handling, opt-out, idempotence, and preserving existing child loader paths. This used the existing Python 3.12 venv without installing packages.

Ruff passed all changed loader/test files and the owned Doctor/load-chain functions. Whole CLI still has 34 unrelated existing findings. git diff --check passed after removing a trailing blank line.

Metadata-only `uv pip compile` passed for both serve,gpu-min and serve,gpu on Linux x86_64 Python 3.12. No GPU or paid calls, large binary downloads, project/user environment package installs, or device inference were performed. An earlier resolver attempt selected the unlisted TensorRT postrelease and attempted its small metapackage metadata build in an isolated cache; it failed the macOS platform check before downloading runtime binaries. The final exact metadata avoids that build. Raw resolver receipts and source metadata hashes are under the task workspace work/gpu-followup directory.

## Explicit limits

The checked-in public uv.lock remains unchanged at the root owner's direction. It was already an obsolete reflex-vla 0.1.0 lock. Universal regeneration is blocked by a pre-existing aarch64 base NumPy<2 versus LeRobot 0.6.1 NumPy>=2 extras conflict. The README states that development must not use --locked/--frozen until that historical lock is reconciled. The successful targeted serving resolver does not establish a successful universal resolution or a full binary installation.

Family discovery uses distribution metadata and recognizable library files. It cannot prove that arbitrary manually copied binaries have the declared ABI. A tiny ORT session-init probe does not establish policy execution, TensorRT graph placement, latency, memory use, WSL2/Orin compatibility, or physical robot execution. The tester archive did not include raw kernel OOM evidence; no kernel OOM is claimed here.


## Final combined integration check

The affected runtime, Doctor, loader, CLI/chat, provider dispatch, proof,
certificate, release-consumer, Jetson installer and ScatterND suites pass
**360 tests**, with one Linux-only install-smoke skip on macOS (7.57 seconds).
[Final log](public-final-tests-aligned.log.gz) and
[source hashes](verification.json) bind the observed scope. This is not a full
repository or GPU acceptance run. The first combined run imposed global offline
flags on an existing test that expects an online missing-tokenizer warning; the
same mismatch reproduces unchanged at dcbcbb5. The final run leaves that test's
environment intact, with offline behavior tested by its own cases. Both initial
and baseline logs remain preserved here.

The additional CUDA 13 ABI check reads wheel ZIP directories, not whole native
wheels. cuSPARSE retains SONAME 12, cuRAND 10 and cuFFT 12 under CUDA 13;
nvJitLink is preloaded before its cuFFT/cuSPARSE dependents. The bounded-read
metadata receipts beside this document record sources and byte counts.

Keep the tester's Orin paused until corrected bits and a fresh identified export
are available. Old exports need the existing ScatterND repair or regeneration;
this change does not silently rewrite benchmark artifacts. Next target run must
record effective provider options, actual node placement, engine completion,
startup memory observations and inference samples. Neither the workspace budget
nor FP16 request proves that the full policy fits the board. The old Orin/5090
FAIL certificates remain historical failures. No tester message, GPU/model run,
physical motion, new protected installer, merge or release occurred here.
