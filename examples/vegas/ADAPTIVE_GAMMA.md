# Adaptive MTP+Vegas gamma

`--vegas-adaptive-gamma` is available with `--vegas-mode mtp-vegas`. The
value passed to `--vegas-gamma` is the safe dense-MTP horizon and the initial
sparse horizon. Sparse candidates remain gamma 1 through 10.

The request-local controller starts with two bounded measurements:

1. dense MTP at the configured gamma, with Vegas drafting and target mask
   collection paused;
2. MTP+Vegas at the same configured gamma.

It then compares the measured dense-MTP efficiency with the paper-style
expected efficiency of sparse gamma candidates. A dense decision keeps the
configured MTP horizon but pauses sparse attention; it is reported as
`adaptive_planned_sparse=false`. This is the conceptual gamma-zero action: it
disables Vegas, not MTP drafting.

Draft entropy and top probability are computed from the full MTP output
distribution by an attached backend sampler. CUDA performs the softmax and
reduction, and only two scalars are copied to the host. Telemetry reports
`adaptive_device_entropy_samples` and `adaptive_fallback_entropy_samples`.
The existing CPU full-logit calculation remains a correctness fallback when a
backend cannot run the entropy graph.

The controller is request-local. Cost and entropy priors are not currently
persisted between requests.
