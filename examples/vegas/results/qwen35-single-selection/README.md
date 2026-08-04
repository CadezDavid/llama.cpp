# Single target-layer Vegas selection

Date: 2026-08-03

## Change

MTP drafting consumed only the last nonempty target-layer selection, but target
verification previously generated, downloaded, and sorted a selection for
every eligible target layer. MTP+Vegas now configures the target context to
generate only the final main target-layer selection. Self-speculative Vegas
keeps its existing per-layer selections.

The selection layer is part of graph-reuse compatibility. Invalid recurrent or
sliding-window selection layers are rejected. No CUDA kernel was changed.

## Primary Qwen 35B confirmation

The primary workload is the real OpenCode conversation used by the preceding
gamma/sparsity screen: 113,612 prompt tokens ending on a user message, 128
generated tokens, Qwen3.6 35B-A3B Q4_K_XL, q4_0/q4_0 KV, greedy sampling,
context capacity 114,688, batch 1024, microbatch 64, and MTP microbatch 64.

Both real prompts were reconstructed from OpenCode session
`ses_2452593b5ffe6yJELmXEV6x9oM` in
`/home/david/.local/share/opencode/opencode.db`. The primary prompt retains
user text, assistant text, and assistant reasoning through source message 412;
it is 365,119 bytes with SHA-256
`14280de66702a9d45f8950082c42229d57ea73ab7675d1e9edb3fd6d20f17bea`.
The second prompt retains user and assistant text through user message 238; it
is 165,425 bytes with SHA-256
`a111c468c5a4617a7ad22d166f692af8443cbd2ba402f1fe128aca0fd3b0fbea`.

| Mode | Gamma | Ratio | tok/s | Delta vs same-gamma MTP | Acceptance | Cycles | Draft ms | Verify ms | Collect ms | Hash match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| MTP | 3 | - | 125.53 | baseline | 85.05% | 36 | 304.65 | 679.31 | 0.00 | yes |
| MTP+Vegas | 3 | 3% | 133.95 | +6.70% | 81.82% | 37 | 136.56 | 775.16 | 4.25 | yes |
| MTP | 4 | - | 118.70 | baseline | 71.54% | 34 | 368.45 | 672.63 | 0.00 | yes |
| MTP+Vegas | 4 | 3% | 134.01 | +12.90% | 71.54% | 34 | 156.58 | 757.56 | 3.89 | yes |
| MTP | 5 | - | 123.14 | baseline | 68.06% | 29 | 402.83 | 603.78 | 0.00 | yes |
| MTP+Vegas | 5 | 3% | 149.82 | +21.66% | 71.22% | 28 | 162.05 | 656.45 | 3.18 | yes |
| MTP+Vegas | 5 | 5% | 143.64 | +16.64% | 68.06% | 29 | 173.37 | 679.19 | 5.35 | yes |
| MTP+Vegas | 5 | 7% | 141.40 | +14.82% | 68.06% | 29 | 181.90 | 682.15 | 7.55 | yes |

Gamma-5 3% values are medians from two fresh order-rotated paired repetitions.
Its dense and hybrid standard deviations were 0.21 and 0.20 tok/s. A third
independent optimized run measured 150.29 tok/s, giving an optimized three-run
range of 149.67-150.29 tok/s. Other rows are one-pass screens.

## Before and after

These rows compare the same prompt, output hash, gamma, ratio, and generated
token count before and after the one-layer change.

| Gamma | Ratio | Old tok/s | New tok/s | Change | Old collect ms | New collect ms | Old verify ms | New verify ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3% | 117.35 | 133.95 | +14.14% | 36.86 | 4.25 | 878.62 | 775.16 |
| 5 | 3% | 132.97 | 149.98 | +12.79% | 28.18 | 3.17 | 739.27 | 655.61 |
| 5 | 5% | 125.13 | 143.64 | +14.79% | 49.66 | 5.35 | 766.30 | 679.19 |
| 5 | 7% | 121.62 | 141.40 | +16.26% | 71.17 | 7.55 | 765.07 | 682.15 |

The gamma-5 3% `new` value in this table is the mean of all three optimized
runs. Selection collection fell by about 89%, consistent with replacing ten
Qwen target-layer masks with one. The remaining verification overhead includes
the one retained score/top-k operation and workload-dependent cycle effects.

## Additional workloads

| Model/workload | Prompt | Gamma | Ratio | MTP tok/s | MTP+Vegas tok/s | Delta | Hash match |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Qwen 35B, second real OpenCode prefix | 62,405 | 5 | 3% | 110.43 | 121.68 | +10.19% | yes |
| Qwen 35B q8/turbo4, same prefix | 62,405 | 5 | 3% | 98.33 | 97.82 | -0.53% | yes |
| Qwen 27B, same real OpenCode prefix | 62,405 | 3 | 3% | 49.29 | 47.02 | -4.59% | yes |
| Gemma 4 31B + Q8 assistant, q8/turbo4 | 65,375 | 3 | 3% | 32.68 existing | 32.08 | -1.84% | yes |

Qwen 35B q8/turbo4 is effectively neutral in its one-pass screen. Vegas saved
about 135 ms of draft time, but acceptance fell from 35.84% to 34.63% and the
hybrid required one additional verification cycle. The optimized Gemma hybrid
improved from the previous 31.16 to 32.08 tok/s;
collection fell from 34.02 to 4.15 ms. It remains slightly behind dense MTP at
this 64K workload. Qwen 27B remains negative because sparse drafting lowers
acceptance enough to require three additional verification cycles; eliminating
discarded masks cannot compensate for that loss.

## Correctness and limits

- The 8,192-token ratio=1 test produced identical dense-MTP and MTP+Vegas
  hashes, acceptance, cycles, and proposal counts.
- A matched-gamma standalone self-speculative ratio=1 smoke test also produced
  identical dense-spec and Vegas hashes, acceptance, cycles, and proposal
  counts. This confirms that the default per-layer selection path remains
  active outside MTP mode.
- Every same-gamma pair in this campaign produced matching output hashes.
- Gamma 7 still OOMed in the FlashAttention workspace at 113,612 prompt tokens;
  removing selection masks did not change that limit.
- Gamma 3, 4, and 5 continue to produce different hashes. This predates the
  optimization and still blocks dynamic gamma from being considered
  output-neutral.
- The strongest result is confirmed on one prompt. Other prompts, ratios, and
  models are screening measurements unless stated otherwise.

## Conclusion

Computing one target-layer selection is a successful dead-work elimination.
It preserves the mask and output behavior already used by MTP, reduces Qwen
selection collection by roughly 89%, and raises the best confirmed Qwen 35B
q4_0/q4_0 result from a screening +8.7% over dense MTP to a stable +21.7%. It
also turns gamma 3 positive on that workload and produces a +10.2% q4_0/q4_0
result on a second real conversation. The benefit is not universal: Qwen 35B
q8_0/turbo4 is neutral, while Qwen 27B and 64K Gemma remain slightly negative
because acceptance and extra verification cycles dominate.

Raw JSONL files in this directory contain exact commands and measurements.
`excluded-mismatched-gamma-self-spec-smoke.jsonl` is retained for audit but is
excluded from all results because its dense-spec and Vegas gamma values were
accidentally different. The corrected matched-gamma smoke is
`self-spec-ratio1-g3-smoke.jsonl`.
