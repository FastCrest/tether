# Serving proof evidence followup, 10/22/2026

The customer is an engineer reviewing whether a tested policy/export can serve
its intended robot cell. A useful receipt must distinguish serving responses,
scheduled actions, reported consumer execution, and physical movement.

## Confirmed defect and fix

`deploy_proof.py` sends identical synthetic zero-state HTTP requests. It does not
run a robot consumer. The action certificate introduced in `ed346ca` nevertheless
used a full returned chunk when an execution horizon was absent, then treated its
tail-to-next-head delta as an executed boundary. RTC's historical
`executed_horizon` also described a configured/applied schedule before a consumer
executed it. The baseline reproduction produces a 0.8 boundary without any
execution telemetry; the corrected executed-boundary metric is null.

New proofs identify their scope as `serving_only`. Their chunks and scheduling
metadata remain intact. Execution checks require explicitly reported
`consumer_observed` telemetry with a positive integral `executed_horizon`.
Synthetic proofs cannot satisfy that contract, even if a response supplies the
marker. Missing observations remain missing and fail qualification. Returned
chunk-tail differences remain clearly labelled diagnostics. Numerical stale,
boundary and velocity limits are unchanged. The realtime serving budget still
applies to one control period, 50 ms at 20 Hz; a nominal 50-action chunk does not
multiply that budget. No certificate verifies physical robot movement.

Proofs and certificates now retain declared export metadata, model identity,
provider observations and configuration, export hashes, inference and warm
latency summaries. Declared precision is not measured runtime precision or node
placement. Startup readiness has a separate optional `startup_timeout_s`, with
the existing request timeout as its default. Optional psutil/procfs observations
record host RSS and available RAM once per second, retaining at most 64 samples.
These observations do not measure GPU allocation or prove OOM. SIGKILL before
readiness remains an exit with an unconfirmed cause.

## Checks and limits

| Check | Observation |
|---|---|
| Proof, certificate, scope and CLI tests | 63 passed in 4.32 s |
| Release assurance and certificate publisher consumers | 15 passed in 0.74 s |
| Owned source/test Ruff and whitespace checks | Pass |
| Actual offline CPU CLI, tiny constant ONNX | Three real HTTP calls; serving proof passes; requesting execution certification fails as expected |
| Archived failed Orin startup, zero actions | Re-evaluation remains FAIL; executed horizon and boundary remain null; SIGKILL does not become an OOM diagnosis |
| Three original private certificates | Original FAIL decisions preserved; file hashes recorded without copying private logs |

The actual CLI flow used an isolated HOME/TETHER_HOME and an existing local
runtime. It exercised `prove --timeout-s 3 --startup-timeout-s 15 --json`, packet
writing and `bench realtime --execution-cert --json`. The local constant model
is a functional fixture, not a full policy or target-hardware performance test.
No GPU run, download, paid request, physical execution, commit or push occurred.
The 5090 source proof JSON is absent from the supplied archive, so its raw samples
were not reconstructed from certificate aggregates. Old failure receipts remain
unchanged. Source and artifact identities are in [receipt.json](receipt.json).

Full local outputs and the fail-first baseline trace remain in the task scratch
directory `work/serving-proof-followup`. This receipt applies to the recorded
uncommitted source state based on `7603af8`; root owns integration and pin updates.
