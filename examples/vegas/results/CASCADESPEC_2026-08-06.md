# CascadeSpec normal-quantization optimization report

> This report records the broad optimization study before the final fixed
> four-round cadence and periodic recurrent checkpoint/replay implementation.
> Its component measurements remain useful, but its end-to-end recommendation
> is superseded by the final 32K and 128K runs indexed in
> [`../README.md`](../README.md).

## Outcome

CascadeSpec is now a correctness-checked implementation of the three-level
pipeline:

```text
MTP drafts -> sparse target corrects or extends -> dense target verifies
```

It supports direct batched indexed attention for normal Q4/Q8 KV caches,
causal sparse verification batches through 16 rows on the MMA path (with a
correct vector fallback above that), dense verification horizons through 50
tokens, and a low-memory replay mode for recurrent models.

The implementation works, but the three-level pipeline is not faster than
plain MTP on Qwen3.6-27B, Gemma4-31B, or Qwen3.6-35B-A3B. Sparse-target passes
are individually cheaper than dense-target passes, but not cheap enough to run
two or three times before each dense verification. The useful production result
from this work is therefore MTP+Vegas, not the additional sparse-target level.

All tests used an RTX 3090 and normal `q8_0/q4_0` KV caches. The benchmark
prompt is made from two natural technical OpenCode conversations and is applied
with each model's native chat template. Temperature is zero and each performance
run emits 128 tokens.

## Implemented milestones

- `a82c926fd`: direct batched indexed Q4/Q8 CUDA attention.
- `47e529462`: tensor-core path extended through 16 query rows.
- `38b807a39`: low-memory dense verification and committed-prefix replay, which
  lets recurrent Qwen models test horizons through 50 without retaining 50
  recurrent rollback states.
- `641044382`, `f54b98492`: resumable quality, horizon, and direct-versus-gather
  benchmark matrices.
- `802b601ca`: `auto` now uses gather-to-dense-MMA for normal Q4/Q8 caches,
  because it won every end-to-end direct/gather comparison. Direct remains
  explicitly selectable and remains useful as the no-materialization baseline.

## Correctness gates

- 516/516 focused CUDA `FLASH_ATTN_EXT` sparse cases pass. These cover direct,
  gather, and auto; Q4/Q4, Q8/Q4, and Q8/Q8 caches; 100% retention; reversed and
  suffix index layouts; and causal batches from 2 through 21 input rows.
- The direct-kernel focused subset passed 96/96 cases at 11 and 16 rows.
- Compute Sanitizer reported zero errors for the D=512, Q8/Q4, 11-row direct
  indexed case.
- The hierarchical policy, adaptive-gamma, and same-prefix unit tests pass.
- All benchmarked hierarchical runs report zero snapshot, rollback, and
  position-state failures.
- Repeated final dense output hashes are stable within each selected
  configuration. The speculative sparse branch can differ near probability
  ties, but the dense target remains authoritative.

At 100% retention, sparse and dense attention have different floating-point
reduction orders, so logits are not bit-identical. The structural gate is what
matters: top-1 agreement is 98--100%, mean total-variation distance is 0.6--2.3%,
and the gather baseline shows the same numerical floor.

## Safe attention retention

The table uses 63 teacher-forced comparisons on the same natural prompt. `TV`
is total-variation distance: half the sum of absolute probability differences.
Lower is closer. `Sparse/dense` compares target-model execution time on the
identical token prefix.

| Model | Context | Recommended retention | Top-1 agreement | Mean TV | Sparse/dense |
|---|---:|---:|---:|---:|---:|
| Qwen27 | 32K | 10% | 96.8% | 3.30% | 78.0% |
| Qwen27 | 64K | 10% | 96.8% | 2.77% | 68.8% |
| Gemma4 | 32K | 20% | 98.4% | 6.11% | 94.3% |
| Gemma4 | 64K | 20% | 96.8% | 4.89% | 92.7% |
| Qwen35 | 32K | 20% | 96.8% | 2.13% | 79.3% |
| Qwen35 | 64K | 10% | 98.4% | 2.34% | 61.5% |

For Gemma, 10% is too aggressive: 64K top-1 agreement falls to 93.7% and its
distribution shift is materially larger. For Qwen35 at 32K, 20% is the safer
choice because 10% top-1 agreement is only 93.7%. A distribution-fidelity
application may prefer 35% on Gemma, but at 32K that setting is no longer faster
than dense attention.

## Sparse-target cost

With horizon 8, gamma 3, 20% retention, and the gathered kernel at 64K:

| Model | Sparse pass | Dense pass | Sparse/dense per pass | Sparse passes per dense cycle |
|---|---:|---:|---:|---:|
| Qwen27 | 52.1 ms | 65.1 ms | 80.0% | 2.33 |
| Gemma4 | 43.0 ms | 74.9 ms | 57.5% | 3.06 |
| Qwen35 | 16.0 ms | 27.4 ms | 58.3% | 2.53 |

This satisfies the narrow goal of making sparse-target verification
substantially cheaper on Gemma4 and Qwen35, and measurably cheaper on Qwen27.
It also explains the overall loss. Even a 42% cheaper pass loses when it is run
roughly 2.5--3 times before every dense pass. Attention is only part of a target
forward pass; FFNs or experts, projections, normalization, logits, graph launch,
and KV work remain dense.

GPU entropy diagnostics are not the problem. On a representative Qwen27 64K
run they account for about 20 ms out of 3.31 seconds, below 1%. The measured
controller and state bookkeeping costs are similarly negligible.

## Dense verification horizon

All models reached the requested 50-token provisional horizon. There is no
functional dense-verification limit at 50. The practical limit arrives much
earlier because acceptance falls and recurrent replay costs another target
pass.

32K, 20% retention, gamma 3, gather kernel:

| Model | Plain MTP | H=8 | H=12 | H=20 | H=32 | H=50 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen27 | 49.94 | 43.42 | 34.00 | 35.08 | 27.40 | 25.74 |
| Gemma4 | 49.36 | 35.23 | 32.33 | 21.00 | 19.94 | 13.67 |
| Qwen35 | 171.88 | 105.11 | 80.24 | 92.07 | 66.80 | 68.79 |

Horizon 8 is the best hierarchical setting. At horizon 50, committed tokens per
dense cycle are about 25.4 for Qwen27, 11.6 for Gemma4, and 21.2 for Qwen35,
but dense acceptance falls to 58%, 30%, and 53%, respectively. Reducing dense
cycle count therefore does not compensate for constructing low-quality long
provisional blocks.

## MTP proposal length

For hierarchical horizon 8 at 32K and 20% retention, fixed gamma 3 is the best
common setting:

| Model | Gamma 3 | Gamma 5 | Gamma 10 |
|---|---:|---:|---:|
| Qwen27 | 43.42 | 39.54 | 43.63 |
| Gemma4 | 35.23 | 33.00 | 24.08 |
| Qwen35 | 105.11 | 93.47 | 83.95 |

Qwen27's gamma-10 result is statistically tied with gamma 3, while its sparse
agreement is lower. On the other two models, extra MTP rows are usually computed
past the first mismatch and discarded. Gamma 3 is consequently the clean,
stable default.

## Final 64K repeats

These are medians of three complete runs using the selected retention and the
gather kernel. All three repetitions have the same final output hash within a
configuration.

| Model | Retention | Plain MTP | MTP+Vegas | CascadeSpec H=8 | MTP+Vegas change | CascadeSpec change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen27 | 10% | 44.71 | 46.13 | 38.64 | +3.2% | -13.6% |
| Gemma4 | 20% | 38.12 | 38.99 | 30.86 | +2.3% | -19.0% |
| Qwen35 | 10% | 142.32 | 142.45 | 94.98 | +0.1% | -33.3% |

Qwen35's hierarchical repetitions range from 93.5 to 109.6 tokens/s because
near-tie sparse corrections change the number of inner rounds. The final dense
output hash remains stable. This variation is another reason not to build an
online horizon controller on top of the current hierarchy.

## Adaptive-controller decision

The existing MTP+Vegas adaptive-gamma controller already has the required
low-overhead machinery: GPU entropy/top-probability statistics, exponential
moving averages, measured costs by gamma, and a conceptual gamma-zero action
that selects dense MTP.

A second controller for both inner gamma and outer dense horizon was not added.
Every measured hierarchical candidate is slower than plain MTP. An adaptive
selector cannot outperform its fastest candidate; on this data its optimal
action would simply be to disable the hierarchy. Adding exploration, two noisy
parameters, and request-local cold-start measurements would make behavior less
stable without creating a viable arm.

The next credible architectural experiment is not finer scalar tuning. It is a
tree or multi-branch MTP proposal that lets one sparse-target pass choose among
several continuations, or a genuinely smaller intermediate model. Either can
reduce the number of full sparse-target passes per dense cycle. Until then,
`MTP+Vegas` should be the production path and `mtp-hierarchical` should remain
an explicitly experimental diagnostic mode.

## Reproduction artifacts

- `cascadespec-quality-normal-2026-08-06/`: 36 same-prefix runs covering all
  three models, 32K/64K, and 100/75/50/35/20/10% retention.
- `cascadespec-horizons-32k-r20-2026-08-06/`: vanilla MTP, MTP+Vegas, and both
  hierarchical kernel strategies through horizon 50.
- `cascadespec-gamma5-8-32k-r20-2026-08-06/` and
  `cascadespec-gamma10-32k-r20-2026-08-06/`: proposal-length screens.
- `cascadespec-selected-64k-r20-2026-08-06/` and
  `cascadespec-qwen-64k-r10-2026-08-06/`: selected 64K screens.
- `cascadespec-repeat-*-64k-*-2026-08-06/`: final three-run comparisons.

Each directory contains a manifest with the prompt hash, model/context matrix,
benchmark settings, and source commit; full JSONL telemetry; and a compact JSON
summary. The runners are `sparse_quality_matrix.py` and
`hierarchical_matrix.py` in `examples/vegas/`.
