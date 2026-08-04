# Qwen 35B MTP + Vegas gamma/sparsity screening

Date: 2026-08-03

This is a single-repetition screening run. It tests whether a small gamma and
sparsity sweep can make the current MTP + Vegas implementation useful on an
RTX 3090 without changing the implementation.

## Configuration

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Drafting: embedded MTP
- Prompt: real OpenCode session `ses_2452593b5ffe6yJELmXEV6x9oM`, titled
  `Rebuilding MCP memory server from scratch`
- Prompt boundary: user message at source message index 412
- llama.cpp prompt length: 113,612 tokens
- Generation: 128 tokens, greedy, seed 1234, EOS ignored
- GPU: NVIDIA RTX 3090
- KV cache: q4_0 K and q4_0 V for target and draft
- Batch / microbatch / MTP microbatch: 1024 / 64 / 64
- Flash attention: enabled
- Vegas minimum selected tokens: 256
- Repetitions: 1 per successful cell

Gamma 3 used a context capacity of 131,072. Gamma 5 initially OOMed at that
capacity, so gamma 5 and gamma 7 were attempted at 114,688. The actual prompt
and generated-token count were unchanged. Results must therefore be compared
against the dense-MTP baseline with the same gamma and capacity, not across
gamma rows.

## Results

| Gamma | Vegas ratio | Capacity | Mode | tok/s | Delta vs matching MTP | Acceptance | Cycles | Output hash |
|---:|---:|---:|:---|---:|---:|---:|---:|:---|
| 3 | - | 131,072 | MTP | 127.89 | baseline | 85.05% | 36 | `efc299ac858c04fa` |
| 3 | 3% | 131,072 | MTP + Vegas | 117.35 | -8.24% | 81.82% | 37 | `efc299ac858c04fa` |
| 3 | 5% | 131,072 | MTP + Vegas | 110.76 | -13.40% | 78.76% | 38 | `efc299ac858c04fa` |
| 3 | 7% | 131,072 | MTP + Vegas | 108.04 | -15.52% | 78.76% | 38 | `efc299ac858c04fa` |
| 5 | - | 114,688 | MTP | 122.29 | baseline | 68.06% | 29 | `4177b91ba5e9b174` |
| 5 | 3% | 114,688 | MTP + Vegas | 132.97 | +8.73% | 71.22% | 28 | `4177b91ba5e9b174` |
| 5 | 5% | 114,688 | MTP + Vegas | 125.13 | +2.32% | 68.06% | 29 | `4177b91ba5e9b174` |
| 5 | 7% | 114,688 | MTP + Vegas | 121.62 | -0.54% | 68.06% | 29 | `4177b91ba5e9b174` |
| 7 | - | 114,688 | MTP | OOM | unavailable | unavailable | unavailable | unavailable |
| 7 | 3% | 114,688 | MTP + Vegas | OOM | unavailable | unavailable | unavailable | unavailable |
| 7 | 5% | 114,688 | MTP + Vegas | stopped | prior gamma-7 OOM confirmed | unavailable | unavailable | unavailable |
| 7 | 7% | 114,688 | MTP + Vegas | not run | prior gamma-7 OOM confirmed | unavailable | unavailable | unavailable |

Gamma 7 failed during both the dense-MTP and 3%-Vegas FlashAttention graphs.
The shell loop proceeded to 5% after the first failure; that attempt was
interrupted as soon as the repeated failure condition was discovered. The 7%
cell was not run. No gamma-7 configuration was retried.

## Timing detail

| Gamma | Ratio | Draft ms | Verify ms | Select/collect ms | Total decode ms |
|---:|---:|---:|---:|---:|---:|
| 3 | MTP | 301.52 | 660.73 | 0.00 | 1000.86 |
| 3 | 3% | 136.09 | 878.62 | 36.86 | 1090.78 |
| 3 | 5% | 143.39 | 905.61 | 64.98 | 1155.69 |
| 3 | 7% | 148.96 | 901.83 | 93.22 | 1184.77 |
| 5 | MTP | 405.32 | 609.32 | 0.00 | 1046.71 |
| 5 | 3% | 161.69 | 739.27 | 28.18 | 962.63 |
| 5 | 5% | 172.80 | 766.30 | 49.66 | 1022.93 |
| 5 | 7% | 181.70 | 765.07 | 71.17 | 1052.43 |

## Interpretation

The best screened setting is gamma 5 with 3% sparsity. It improves throughput
by 8.73% in this one run and preserves the matching dense-MTP output hash. The
gain comes from reducing draft time by about 244 ms while adding about 130 ms
to verification and 28 ms to selection/collection. Gamma 5 at 5% is only
slightly positive, and 7% is flat. All gamma-3 hybrid settings are clearly
negative.

This is promising but not confirmatory evidence. The best cell is n=1, and its
advantage partly includes one fewer speculative cycle after a small acceptance
increase. It should be repeated before becoming a default.

The output hash changes between gamma 3 and gamma 5 even though each hybrid
hash matches its same-gamma dense-MTP baseline. Consequently, the sweep proves
same-gamma throughput behavior, but it does not prove that changing gamma is
output-neutral. That difference requires a separate correctness investigation
if gamma 5 is taken forward.

Raw JSONL files in this directory contain the exact commands and measurements.
