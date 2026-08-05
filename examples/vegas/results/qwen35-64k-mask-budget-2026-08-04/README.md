# Qwen35 64K q8/turbo4 mask-budget check

## Setup

- Model: Qwen3.6 35B-A3B Q4_K_XL with built-in MTP
- Prompt: 62,101-token OpenCode session
- Context capacity: 73,728
- Output: 128 greedy tokens, seed 1234
- Target and draft KV: q8_0/turbo4
- Gamma: 4
- Dense MTP was run once. Vegas was run once at 5% and once at 6%.

A subsequent controlled pair uses gamma 3 with a 7% Vegas mask.

## Results

| Mode | Selected history | tok/s | Versus dense | Acceptance | Accepted/drafted | Cycles | Draft ms | Verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense MTP | all | 92.964 | - | 33.95% | 73/215 | 54 | 370.360 | 967.979 |
| Vegas 5% | about 3,106 | 91.333 | -1.76% | 32.27% | 71/220 | 56 | 238.733 | 1123.353 |
| Vegas 6% | about 3,727 | 91.224 | -1.87% | 32.27% | 71/220 | 56 | 239.266 | 1124.577 |

| Gamma 3 mode | Selected history | tok/s | Versus dense | Acceptance | Accepted/drafted | Cycles | Draft ms | Verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense MTP | all | 107.598 | - | 47.44% | 74/156 | 53 | 276.021 | 877.082 |
| Vegas 7% | about 4,348 | 103.816 | -3.52% | 47.44% | 74/156 | 53 | 179.424 | 1009.163 |

The gamma-4 runs produced output hash `792389b5fa634027`. Both gamma-3 runs
produced `cea28a83c72ceb4a`. Each Vegas result therefore matches its controlled
dense configuration.

Gamma 4 and the larger mask substantially improve the earlier gamma-5/3%
Vegas result of 85.117 tok/s, but neither mask beats the paired gamma-4 dense
baseline. Five and six percent have the same speculative trajectory. The extra
621 selected positions at 6% do not recover the two-cycle acceptance deficit.

This indicates that an absolute token floor alone is not sufficient for this
prompt and cache format. The remaining roughly 1.8% deficit is small enough
that one repetition cannot establish its exact size, but there is no positive
screening result to justify confirmation yet.

Gamma 3 is substantially faster than gamma 4 for both practical references,
but the controlled gamma-3 pair remains negative for Vegas. Sparse drafting
saves about 97 ms while verification rises about 132 ms. Acceptance and cycle
count already match dense MTP, so increasing the mask further is not justified.

Raw records:

- `dense-gamma4.jsonl`
- `dense-gamma3.jsonl`
- `vegas-gamma4-ratio05.jsonl`
- `vegas-gamma4-ratio06.jsonl`
- `vegas-gamma3-ratio07.jsonl`
