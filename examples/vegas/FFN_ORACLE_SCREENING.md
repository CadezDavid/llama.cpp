# Draft FFN oracle sparsity screening

Date: 2026-08-04

Archive status: the oracle graph path and its command-line controls were
removed from active code on 2026-08-05. The final implementation snapshot is
commit `f1e07e9651c53ca0dd43307be62263a6d999a71b`; raw results remain preserved
below and in `results/ffn-oracle-2026-08-04/`.

## Question

Can MTP+Vegas drafts discard a large fraction of FFN channels without losing
speculative acceptance?

This is an oracle diagnostic, not an optimized sparse FFN implementation. Each
dense FFN computes its exact post-activation vector, selects the largest
absolute values per token, zeros the remaining channels, and then runs the
ordinary down projection. Only the draft context is modified. The target
context remains dense and verifies every token exactly.

Because the up and gate projections and the dense down projection still run,
the throughput numbers below do not represent the potential performance of a
real channel-sparse kernel. The useful measurements are acceptance, cycle
count, and final target output hash.

## Configuration

Both screens used greedy sampling, 128 generated tokens, q8_0/turbo4 KV for
target and draft, and real conversation text.

- Qwen3.6 27B: 112,025-token OpenCode conversation, gamma 4, Vegas ratio 10%,
  target selection layer 15, refresh interval 2.
- Gemma 4 31B: 128,000-token Codex conversation, separate Q8_0 assistant,
  gamma 1, Vegas ratio 3%, refresh interval 1.

Sparsity is the fraction of post-activation FFN channels set to zero. The
retained fractions for 40%, 50%, 60%, and 70% sparsity are 60%, 50%, 40%, and
30%, respectively.

## Results

### Qwen3.6 27B, 112K, q8_0/turbo4

| FFN sparsity | Acceptance | Accepted / drafted | Cycles | Draft ms | Output hash |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0% | 45.0% | 81 / 180 | 46 | 469.1 | `77b9f777f84ba946` |
| 40% | 45.0% | 81 / 180 | 46 | 492.7 | `77b9f777f84ba946` |
| 50% | 46.6% | 82 / 176 | 45 | 479.9 | `77b9f777f84ba946` |
| 60% | 45.0% | 81 / 180 | 46 | 488.2 | `77b9f777f84ba946` |
| 70% | 45.0% | 81 / 180 | 46 | 485.1 | `77b9f777f84ba946` |

Qwen's native MTP draft block tolerated the entire range. The 50% result took
one fewer cycle, but one deterministic run is not enough to claim that pruning
improves acceptance. There is no evidence of an acceptance penalty through
70% oracle sparsity in this screen.

### Gemma 4 31B, 128K, q8_0/turbo4

| FFN sparsity | Acceptance | Accepted / drafted | Cycles | Draft ms | Output hash |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0% | 80.0% | 56 / 70 | 71 | 189.9 | `4ba27d80f7a93771` |
| 40% | 80.0% | 56 / 70 | 71 | 192.7 | `4ba27d80f7a93771` |
| 50% | 82.6% | 57 / 69 | 70 | 191.9 | `4ba27d80f7a93771` |
| 60% | 76.4% | 55 / 72 | 72 | 197.1 | `4ba27d80f7a93771` |
| 70% | 78.9% | 56 / 71 | 71 | 194.5 | `4ba27d80f7a93771` |

Gemma also tolerated all tested sparsity levels despite applying the oracle to
the full separate assistant stack. The first visible degradation occurred at
60%, but it was small and did not continue monotonically at 70%. The practical
oracle target from this one prompt is therefore 40-50% sparsity, with 60-70%
still plausible but requiring broader acceptance tests.

## Interpretation

The experiment gives a positive answer to the narrow first-stage question:
post-SwiGLU channel magnitude is concentrated enough that removing at least
40-50% of draft FFN channels can preserve MTP+Vegas acceptance on both model
architectures. It also shows substantial oracle headroom beyond 50%.

It does not establish a speedup. A useful implementation still needs a cheap
predictor that identifies channels before the up and gate projections, plus
CUDA kernels that compute only selected projection rows and down-projection
columns. Any predictor must be tested against this oracle ceiling and must save
more compute than its selection overhead.

The unchanged output hashes across every point confirm that dense target
verification preserved exact greedy output for these runs.

## Raw data

Raw one-run JSONL records are in
`examples/vegas/results/ffn-oracle-2026-08-04/`:

- `qwen27-112k-q8-turbo4-s00.jsonl` through `s07.jsonl`
- `gemma-128k-q8-turbo4-s00.jsonl` through `s07.jsonl`
- `smoke-dense.jsonl`, `smoke-s40.jsonl`, and `smoke-s90.jsonl`

The long-context files use `s00`, `s04`, `s05`, `s06`, and `s07` for 0%, 40%,
50%, 60%, and 70% sparsity.
