# Gemma 4 assistant Q4 screening

This screening compares the existing Gemma 4 31B Q8_0 assistant with a Q4_K_M assistant produced from the original BF16 safetensors. It is a small n=1 experiment, not a stable estimate of differences near zero.

## Assistant files

| Assistant | Bytes | Relative size |
|---|---:|---:|
| Q8_0 | 514,704,896 | 100.0% |
| Q4_K_M | 353,485,824 | 68.7% |

The Q4_K_M file was created by converting the original safetensors to F16 and then running `llama-quantize` with `Q4_K_M`. It was not requantized from Q8_0. Q4_K_M is a mixed quantization: notably, the token embedding/output tensor and selected FFN tensors remain Q6_K.

Installed Q4_K_M model:

`/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/gemma-4-31B-it-qat-q4_0-assistant-Q4_K_M.gguf`

SHA-256: `9d75ecaa6bb998a54e8ac0f414d93ed253e87aea7b1c0b1d04cf14571e795159`

## Workload

The prompt is a transcript extracted from this real Codex session:

`/home/david/.codex/sessions/2026/07/22/rollout-2026-07-22T18-03-37-019f8a91-7b8d-7e32-9883-555bacd67386.jsonl`

Only user and assistant text messages were retained. System/developer instructions, environment records, reasoning, and tool calls/results were excluded. The extracted transcript was 476,807 bytes with SHA-256 `a1eb612159451eeae6a79931f72537734021e9d8dd1889f1b5b75a58b7b9a9ef`.

Runs used Gemma 4 31B Q4_K_XL, batch one, greedy decoding, 128 generated tokens, gamma 3, q8_0/turbo4 target and draft KV, and one repetition. Vegas used a 3% mask and 256-token floor. Contexts were 65,536 and 128,000 prompt tokens. The 128K runs used ubatch and MTP ubatch 64; the 64K runs used 128.

## Results

| Context | Mode | Q8_0 tok/s | Q4_K_M tok/s | Q4/Q8 | Q8 accept | Q4 accept | Cycles Q8 -> Q4 | Draft ms/proposal Q8 -> Q4 | Hash match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 64K | MTP | 21.363 | 22.019 | +3.07% | 15.56% | 17.34% | 87 -> 84 | 3.861 -> 3.855 | yes |
| 64K | MTP+Vegas | 19.928 | 19.672 | -1.28% | 12.82% | 12.32% | 92 -> 93 | 1.939 -> 1.887 | yes |
| 128K | MTP | 10.509 | 10.064 | -4.24% | 7.07% | 5.23% | 105 -> 110 | 13.228 -> 13.170 | yes |
| 128K | MTP+Vegas | 22.071 | 22.467 | +1.80% | 36.46% | 37.64% | 61 -> 60 | 2.541 -> 2.486 | yes |

Raw JSONL is under `examples/vegas/results/assistant-q4-screening/`. A completed 64K C++-prompt Q8 run was retained with a `preliminary-cpp-` prefix but excluded from this table. The interrupted C++-prompt Q4 run produced no result row.

## Interpretation

Q4_K_M reduces the assistant file by 31.3%, but does not materially accelerate dense drafting. Draft time per proposal changes by less than 0.5% in dense MTP at both context lengths. At 128K, attention over the KV cache dominates assistant weight traffic, and Q4's lower acceptance makes total throughput 4.2% worse.

With Vegas sparse drafting, Q4 reduces draft time per proposal by about 2-3%. End-to-end differences remain small and are governed by one or two verification cycles: -1.3% at 64K and +1.8% at 128K. These differences are too small for an n=1 screening run to establish a reliable speed advantage.

The useful outcome is therefore memory reduction, not demonstrated throughput improvement. Q4_K_M preserves the generated output in every tested pair, but Q8_0 remains the safer default because the dense 128K run shows measurable acceptance degradation. A Q5_K_M assistant would be the next sensible quality/memory compromise if additional assistant quantization work is desired.
