# Hierarchical Vegas at 20% sparse attention

This experiment repeats the three 64K hierarchical runs with
`--vegas-ratio 0.20`. All other model, cache, prompt, sampling, and hierarchy
settings match the baseline in `../hierarchical-64k-2026-08-05/`.

Each run generated 128 tokens from the same 65,536-token OpenCode technical
conversation prompt. Results are single runs, so small differences should not
be treated as stable without repetitions.

| Configuration | Baseline ratio | tok/s baseline | tok/s at 20% | Change | Dense cycles baseline -> 20% | Committed/dense baseline -> 20% | Sparse agreement baseline -> 20% | Dense acceptance baseline -> 20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen27 q8/turbo4 | 7.2768% | 11.619 | 10.775 | -7.27% | 46 -> 43 | 2.761 -> 2.953 | 46.99% -> 53.45% | 47.37% -> 45.16% |
| Gemma4 q8/turbo4 | 3% | 8.757 | 7.086 | -19.08% | 84 -> 81 | 1.512 -> 1.568 | 17.09% -> 22.44% | 21.50% -> 22.12% |
| Qwen35 q8/q4 | 3% | 46.340 | 50.851 | +9.73% | 22 -> 17 | 5.773 -> 7.471 | 77.34% -> 82.69% | 70.00% -> 88.00% |

The larger sparse set improved sparse agreement for every model, but only
Qwen35 converted that improvement into higher end-to-end throughput. Qwen27
and Gemma saved three dense cycles each, yet the extra sparse-target work cost
more than those cycles saved. For Qwen35, the stronger middle stage reduced
dense cycles by five and raised throughput by 9.73%.

All runs reported zero position mismatches, snapshot failures, rollback
failures, and CPU entropy fallbacks. However, all three output hashes differ
from their lower-ratio baselines. Dense verification is intended to be
authoritative, so the ratio-dependent output needs a paired dense-reference
correctness investigation before treating the 64K path as proven exact. It may
be numerical sensitivity from a changed graph shape, but this experiment does
not establish the cause.

Raw per-run data and traces are in `results.jsonl`; the compact metrics are in
`summary.json`.
