# Private Vegas experiment

Private fork only. This is not an upstream submission artifact.

## Verdict

Vegas improves batch-one decoding on this RTX 3090, but only beyond a
model- and cache-dependent context crossover and only with a short draft chain.

- Qwen3.6 27B: substantial at 64K with q8/q4 or q4/q4 (about 27%) and at
  120K with q8/turbo4 (about 11%).
- Qwen3.6 35B-A3B: slower at 32K, but substantial at 64K with q4/q4
  (about 21%).
- Gemma 4 31B: slower at 32K. At 64K q8/q8 ranges from about 1% on code
  content to about 4% on paper content, which is not a substantial win.
- Paper-like gamma=5, ratio=7% is about 35% slower for Qwen3.6 27B at 64K.
  Gamma=1 is essential on this stack.

The best setting is not q8/turbo4 for every model. Standard q4 V kernels are
much faster for Qwen at 64K. The q8/turbo4 crossover occurs later.

## System and method

- GPU: NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6
- Driver observed during the final run: 610.43.03
- Build: Release, CUDA architecture 86, GCC/G++ 15, CUDA enabled
- Batch: one sequence, batch 1024, ubatch 512, all layers on CUDA
- Decode: 256 greedy tokens, fixed seed 1234, EOS ignored
- Benchmarking: fresh process per mode, alternating baseline/Vegas order,
  per-run GPU telemetry, paired ratios, JSONL append-only records
- Primary prompt seed: text extracted from `2602.07223v2.pdf`
- Independent content check: `tests/test-backend-ops.cpp`

Prompt prefill and model loading are reported separately and are excluded from
decode throughput. The one initial token selection is excluded consistently
from both timers while the fixed 256-token numerator is retained, so absolute
tok/s is slightly optimistic; paired Vegas/baseline ratios are unaffected.

The 95% intervals below are deterministic paired bootstrap intervals over
three or five repetitions. With such small samples they describe run-to-run
stability, not population-level confidence.

## Repeated paired results

| Model | Prompt | Context | K/V | gamma/ratio | n | Baseline | Vegas | Paired ratio (95%) | Accept |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B | paper | 64K | q8/turbo4 | 1/3% | 5 | 23.393 | 23.919 | 1.0225 [1.0218, 1.0231] | 99.2% |
| Qwen3.6 27B | paper | 120K | q8/turbo4 | 1/3% | 3 | 17.638 | 19.634 | 1.1132 [1.1124, 1.1143] | 99.2% |
| Qwen3.6 27B | paper | 64K | q8/q4 | 1/3% | 3 | 20.027 | 25.441 | 1.2703 [1.2661, 1.2733] | 100% |
| Qwen3.6 27B | paper | 64K | q4/q4 | 1/3% | 3 | 20.096 | 25.484 | 1.2681 [1.2656, 1.2704] | 100% |
| Qwen3.6 27B | paper | 64K | q8/q8 | 1/3% | 3 | 23.005 | 25.257 | 1.0979 [1.0907, 1.1028] | 100% |
| Qwen3.6 27B | paper | 64K | q8/turbo4 | 5/7% | 3 | 23.345 | 15.152 | 0.6491 [0.6461, 0.6510] | 48.1% |
| Qwen3.6 35B-A3B | paper | 32K | q8/turbo4 | 1/7% | 3 | 95.286 | 75.698 | 0.7945 [0.7862, 0.7999] | 91.7% |
| Qwen3.6 35B-A3B | paper | 64K | q4/q4 | 1/3% | 3 | 59.818 | 72.221 | 1.2073 [1.2029, 1.2119] | 100% |
| Gemma 4 31B | paper | 32K | q8/turbo4 | 1/7% | 3 | 26.391 | 22.905 | 0.8679 [0.8665, 0.8700] | 80.1% |
| Gemma 4 31B | paper | 64K | q8/q8 | 1/3% | 3 | 23.125 | 23.978 | 1.0369 [1.0359, 1.0379] | 100% |
| Gemma 4 31B | code | 64K | q8/q8 | 1/3% | 3 | 23.138 | 23.355 | 1.0094 [1.0085, 1.0105] | 94.7% |

Single-pair cross-content checks preserve the substantial Qwen gains:

- Qwen3.6 27B q8/q4 at 64K: 20.198 -> 25.373 tok/s, 1.2563x,
  99.2% acceptance.
- Qwen3.6 35B-A3B q4/q4 at 64K: 61.067 -> 72.491 tok/s, 1.1871x,
  98.4% acceptance.

Additional one-pair screens show the context crossover:

- Qwen3.6 27B q8/q4 at 8K: 0.903x.
- Qwen3.6 35B-A3B at 32K: q8/q4 0.896x, q4/q4 0.913x,
  q8/q8 0.804x.
- Gemma 4 31B at 32K: q8/q4 0.940x, q4/q4 0.944x,
  q8/q8 0.961x.

## Why it helps or fails

At long context, verification amortizes a full KV read across two output tokens
when gamma=1. Sparse drafting then reads only 3% of the stable prefix plus all
tokens created since the last verification.

Representative 256-token phase means:

| Model/configuration | Baseline total | Vegas draft | Vegas verify | Selection | Vegas total |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 27B, 64K q8/q4 | 12.783 s | 3.655 s | 6.237 s | 0.116 s | 10.063 s |
| Qwen3.6 27B, 120K q8/turbo4 | 14.514 s | 3.691 s | 9.080 s | 0.215 s | 13.039 s |
| Qwen3.6 35B-A3B, 64K q4/q4 | 4.280 s | 1.232 s | 2.191 s | 0.073 s | 3.545 s |
| Gemma 4 31B, 64K q8/q8 | 11.070 s | 3.953 s | 6.590 s | 0.074 s | 10.676 s |

At 8K or 32K, attention is not dominant enough to repay the extra sparse
draft, index collection, rollback, and sampling work. Gemma's 512-wide
attention head makes sparse drafting relatively expensive, so even at 64K the
fixed model and draft cost leaves little margin. Gamma=5 performs five sparse
draft forwards per cycle and accepts only 48.1% of them in the Qwen q8/turbo4
case, making it decisively slower.

## Implementation

The private `llama-vegas` executable implements:

- sequential self-drafting with the target model;
- dense batched verification and greedy or rejection-sampling acceptance;
- verification-guided per-layer scores from the first and bonus queries;
- GPU top-k selection followed by context-order restoration;
- direct indexed CUDA vector flash attention over quantized K/V rows;
- fixed-shape graph reuse with selected prefix rows plus recent dense rows;
- q8/turbo4, q8/q4, q4/q4, and q8/q8 cache support, including 512-wide
  Gemma attention;
- explicit rejection of CPU, partial offload, parallel sequences, flash-off,
  ALiBi, and unsupported cache layouts.

The cache matrix intentionally omits q4/q8 because it spends more V memory
while reducing the more accuracy-sensitive K precision. Turbo4 K layouts are
also omitted: this fork's production-oriented layout keeps K at q8 and uses
TurboQuant for V.

## Correctness and limitations

- Ratio=1 sparse-versus-dense self-speculation has identical hashes, cycles,
  accepted tokens, and rejected tokens for all four supported cache layouts
  on a 128-wide Qwen model.
- Ratio=1 checks also pass for q8/q4, q4/q4, and q8/q8 with Gemma's 512-wide
  attention. The q8/turbo4 512-wide path is exercised by the repeated Gemma
  runs.
- Greedy and stochastic rejection-sampling checks pass, including residual
  sampling after rejection.
- The implementation is lossless under the usual speculative-decoding model
  of identical target distributions. In practice, quantized single-token and
  batched CUDA kernels are not always bitwise identical. Some greedy baseline
  and Vegas hashes therefore differ even at high acceptance. Ratio=1 proves
  that sparse indexing matches the dense batched verifier; it cannot make the
  batched verifier bitwise equal to sequential baseline computation.
- Results measure one RTX 3090, these GGUFs, one batch, greedy generation, and
  two local prompt seeds. They do not establish a general model-quality or
  cross-hardware claim.

## Verification performed

- Full `build-vegas` build completed successfully.
- `test-sampling`: pass.
- `test-batch-alloc`: 198 assertions, zero failures.
- `test-recurrent-state-rollback` with the Qwen3.5 0.8B model: pass.
- Focused CUDA `FLASH_ATTN_EXT` q8/q8 and q4/q4 cases: pass.
- q8/q4 is absent from the generic backend test generator, so it is covered
  by direct ratio=1 and long-context model tests.
- Invalid ratio, parallel=2, CPU-only, flash-off, and f16/f16 Vegas invocations
  fail cleanly.
- `git diff --check` and Python bytecode compilation pass. No non-ASCII added
  lines were found. `clang-format` is not installed on this host.

All raw records are in `examples/vegas/results/`. No commit, push, issue, PR,
or upstream-facing text was created.
