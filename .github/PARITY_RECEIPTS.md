# Parity receipt production controls

The `parity-receipts.yml` workflow runs paid Modal A100 jobs and handles the
credentials used to download the approved model. Its `produce` job is bound to
the `parity-receipts-production` GitHub environment for every trigger:
scheduled runs, `release/**` pushes, and manual dispatches.

Repository administrators must configure that environment before enabling the
workflow:

1. Add at least one independent required reviewer and enable **Prevent
   self-review**. The person who initiated or pushed the run must not be able to
   approve their own deployment. Deselect **Allow administrators to bypass
   configured protection rules**.
2. Limit deployment branches/tags to protected `main` and protected
   `release/**` branches. Those protections must require pull requests and
   code-owner review for `.github/workflows/parity-receipts.yml`, this document,
   and receipt verifier/producer code.
3. Store `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, and `HF_TOKEN` as environment
   secrets, not repository secrets.
4. Store these environment variables:
   - `PARITY_RECEIPTS_ENVIRONMENT_PROTECTED=true`
   - `PARITY_MODEL_DIGEST=<sha256>` where the digest is SHA-256 of
     `huggingface:lerobot/pi05_libero_finetuned_v044@<immutable-hf-commit>`.
5. Audit changes to reviewers, deployment branches, secrets, variables, and
   administrator-bypass settings like code changes.

The manual `authorize_paid_run` input records intent only. It cannot supply or
override the approved model digest, credentials, environment, workflow, ref,
or reviewer gate. Scheduled and release runs still wait for the same protected
environment approval before any step can access secrets or start paid work.

Each authorized run derives a canonical receipt namespace from the lowercase
repository identity, checked-out commit SHA, GitHub run ID, and run attempt.
The workflow passes that complete binding to every Modal producer. In receipt
mode, Hugging Face caches, ONNX exports, benchmark inputs, and local result
JSON files live only below `receipt_runs/<namespace>`; partial receipt metadata
and non-canonical or path-like namespaces are rejected. Static legacy paths
remain available only when no receipt metadata is supplied, outside this
workflow. Retrying a run creates a new namespace and cannot reuse an earlier
attempt's measured export directory.

The same namespace is embedded in both GitHub artifact names. Consumers first
select trusted server-returned workflow metadata, recompute that run attempt's
namespace, and then accept exactly one matching manifest and payload artifact.
Artifacts from earlier attempts, similarly named artifacts, and duplicate
matches are rejected rather than selected heuristically.
