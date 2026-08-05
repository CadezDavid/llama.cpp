# Gemma 64K gamma-1 screen

## Setup

- Gemma 4 31B Q4_K_XL target and Q8_0 assistant
- q8_0/turbo4 target and assistant KV caches
- Real OpenCode conversation from `/tmp/opencode-spomin-user-238.txt`
- 65,071 prompt tokens, 128 greedy output tokens, seed 1234
- Context capacity 73,728, batch 1024, ubatch 128
- Gamma 1 and a 3% Vegas mask
- One run per configuration

The current prompt SHA-256 is
`7a8a48485c930295f92b133a19689d8f12b25dada44b25df9ae43b918322b244`.
The older gamma-3 report records 65,375 tokens and a different prompt hash, so
that result is useful context but is not a byte-identical comparison.

## Results

| Mode | tok/s | Versus dense | Acceptance | Accepted/drafted | Cycles | Draft ms | Verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense MTP | 28.425 | - | 54.88% | 45/82 | 82 | 302.668 | 4179.836 |
| Vegas, refresh 1 | 27.648 | -2.73% | 53.01% | 44/83 | 83 | 164.459 | 4443.864 |
| Vegas, refresh 2 | 27.696 | -2.57% | 53.01% | 44/83 | 83 | 159.409 | 4442.519 |

All configurations produced output hash `df6c42035ba0bb18`.

Gamma 1 does not move the 64K Gemma workload into the profitable Vegas
region in the original implementation. Sparse drafting saves about 138-143 ms,
but the small acceptance loss adds one target cycle and verification rises
about 263 ms.

A later graph-reuse fix removes most of the avoidable verification overhead for
refresh 1. With the same output and 44/83 sparse trajectory, it measured 28.802
tok/s, 4262.666 ms verification, and 333 reused graphs. The same-build
no-reuse control measured 27.709 tok/s, 4432.026 ms verification, and 254
reused graphs. See `../verification-graph-reuse-2026-08-04/README.md`.

Refresh 2 still alternates disabled and collector graph shapes and therefore
does not benefit from the single-entry graph reuse fix.

The older gamma-3 real-conversation screen was also negative. Its dense mode
was faster than the gamma-1 dense mode, but the prompt files are not identical,
so the exact gamma effect requires a new same-prompt gamma-3 pair if needed.

Raw records:

- `refresh1-pair.jsonl`
- `refresh2-vegas.jsonl`
