# Fused Vegas collector results

## Scope

This experiment evaluates the CUDA-only fused verification collector on an RTX
3090. It keeps the existing Vegas mask algorithm, selected target layer,
boundary queries, and policy parameters unchanged. Runs use batch one, greedy
sampling, 128 generated tokens, full CUDA offload, and real Codex or OpenCode
conversation prefixes.

The final implementation:

- accumulates the existing first and last verification-query KQ logits while
  FlashAttention already has those logits in registers;
- writes one partial score vector per KV head, then reduces those vectors on
  the GPU;
- runs the existing anchor handling and top-k selection without reading the KV
  cache a second time;
- leaves the selected plan device-resident for sparse drafting;
- supports the CUDA MMA and VEC FlashAttention paths used in these tests. A
  score request on TILE or WMMA is rejected instead of silently collecting the
  wrong scores.

The per-KV-head scratch avoids the heavy global atomic contention observed in
the first fused prototype. The `*-atomic.jsonl` files preserve those
intermediate measurements. No new policy, selection heuristic, or mask-quality
change is included.

## Correctness

Two Qwen27 checks compared pre-fusion and fused collection:

- A ratio-1 mask reproduced hash `b9e73707b8752e42`, 41 accepted tokens, 63
  drafted tokens, 22 cycles, and 14 rejections.
- A real 3% sparse mask reproduced the same output hash, 39 accepted tokens,
  and 24 cycles.

The final per-KV-head implementation repeated both trajectories exactly. Raw
records are in `ratio1-head-partial.jsonl` and
`sparse-head-partial.jsonl`. The earlier all-head-atomic checks are retained as
`ratio1-correctness.jsonl` and `sparse-correctness.jsonl`.

The final build also passed `test-batch-alloc` with 198 assertions and zero
failures. `git diff --check` passed.

## Focused performance results

Each row is one run per shown mode. Small differences are not confirmation.
Gemma dense is the preserved same-configuration result from
`gemma-128k-q8-turbo4-atomic.jsonl`; at the user's request it was not rerun
after the final collector change.

| Model and cache | Prompt | Dense MTP | Fused MTP+Vegas | Change | Dense accept | Vegas accept |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 27B q8/turbo4 | 112,025 | 29.450 | 30.682 | +4.18% | 45.0% | 45.0% |
| Gemma 4 31B q8/turbo4 | 128,000 | 14.534 | 23.362 | +60.75% | 22.3% | 80.0% |
| Qwen3.6 35B q4/q4 | 112,025 | 128.544 | 170.943 | +32.98% | 82.9% | 82.9% |
| Qwen3.6 35B q8/turbo4 | 62,101 | 89.222 | 85.117 | -4.60% | 29.8% | 26.2% |

Qwen35 q8/turbo4 was tested at the known fitting 62K context. Its 112K
configuration is a previously confirmed OOM on this 24 GB card and was not
retried.

All paired rows produced matching output hashes. Gemma's fused sparse run also
matched the earlier sparse run's hash and acceptance trajectory.

## Timing interpretation

| Model and cache | Dense draft ms | Vegas draft ms | Dense verify ms | Vegas verify ms | Cycles dense/Vegas |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 27B q8/turbo4 | 781.956 | 466.077 | 3521.536 | 3661.590 | 46 / 46 |
| Qwen3.6 35B q4/q4 | 423.997 | 138.361 | 538.254 | 579.797 | 25 / 25 |
| Qwen3.6 35B q8/turbo4 | 434.758 | 291.464 | 962.736 | 1172.403 | 51 / 55 |

For Qwen27, sparse drafting saves about 316 ms, but verification rises about
140 ms. The net result remains a small roughly 4% win. Removing the separate
KV reread does not reveal a large hidden gain.

For Qwen35 q4/q4, acceptance is unchanged and draft time falls by about 67%,
so the established large Vegas win is preserved. The observed +33% pair is
strong evidence of benefit, but a single repetition should not replace the
earlier 16-22% multi-prompt expectation.

For Qwen35 q8/turbo4, sparse draft time is lower, but acceptance falls and four
extra target cycles raise verification time by about 210 ms. Fused collection
cannot fix that mask-economics problem, so this remains a negative
configuration.

A focused gamma-4 follow-up at 5% and 6% is documented in
[`../qwen35-64k-mask-budget-2026-08-04/README.md`](../qwen35-64k-mask-budget-2026-08-04/README.md).
It narrows the loss to about 1.8%, but neither larger mask beats dense MTP.

Gemma retains the large long-context win and the same 80% sparse acceptance.
Its fused throughput differs by only -0.1% from the intermediate fused result,
which is measurement noise at one repetition.

## Raw files

- `qwen27-112k-q8-turbo4.jsonl`
- `gemma-128k-q8-turbo4.jsonl`
- `qwen35-112k-q4-q4.jsonl`
- `qwen35-62k-q8-turbo4.jsonl`
- `*-atomic.jsonl`: intermediate single-score-vector fused collector
- `ratio1-*.jsonl` and `sparse-*.jsonl`: correctness checks
