# Prox experiment archive

Date archived: 2026-08-05

## Status

Prox and the related FFN-oracle experiments are documented but no longer part
of the active Vegas implementation.

The experimental graph path, command-line options, extension API, benchmark
arguments, and `llama-ffn-sparse-eval` executable were removed after the
feasibility study reached a negative result for the current RTX 3090 and GGUF
models. Vegas now contains only the sparse-attention work.

## Documents

- `TARGET_FFN_SPARSITY.md` contains the target-model quality screens, CUDA
  profile, VRAM measurement, quantized-layout analysis, alternatives, and stop
  decision.
- `FFN_ORACLE_SCREENING.md` contains the earlier draft-model oracle screens for
  Qwen3.6 27B and Gemma 4 31B.
- `results/target-ffn-sparsity-2026-08-04/` contains the raw target evaluator
  outputs.
- `results/ffn-oracle-2026-08-04/` contains the raw draft oracle outputs.

The paper evaluated was:

```text
/home/david/Downloads/2607.27591v1.pdf
Prox: Training-Free FFN Activation Sparsity via Approximate
Intermediate-Channel Salience in LLMs
```

## Historical implementation

Commit `f1e07e9651c53ca0dd43307be62263a6d999a71b` is the final snapshot that
contains the removed experimental implementation and evaluator. It includes:

- exact post-activation FFN oracle masking;
- the Prox-shaped sparse-input selection simulation;
- the direct-proxy-value experiment;
- target and draft command-line controls;
- the dense-versus-sparse teacher-forced quality evaluator.

This commit is retained only to make the measurements auditable. The archived
code was a quality and feasibility instrument, not a sparse CUDA implementation
and not a recommended runtime configuration.

## Conditions for reconsideration

Reopen this direction only if at least one material constraint changes:

- a sparse-friendly or channel-oriented GGUF weight layout exists;
- a model is trained for structured FFN sparsity;
- enough VRAM is available for the paper's separate low-precision proxy and an
  efficient down-projection representation;
- a compact predictor demonstrates robust quality and removes enough memory
  traffic to exceed the measured below-5% end-to-end ceiling.

Applying Prox only to the drafter remains a lower-priority possibility, most
plausibly for the Q8 Gemma assistant. Its total speed ceiling is modest, and any
acceptance loss can erase the saved draft time.
