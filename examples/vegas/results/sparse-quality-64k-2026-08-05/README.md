# Sparse-attention quality matrix at 64K

This matrix isolates sparse-attention quality after the execution and prompt-construction fixes.

- Source: two coherent technical conversations from the local OpenCode database.
- Formatting: each model's native chat template.
- Prefix packing: longest suffix of complete turns below a 65,536-token budget; no token repetition or truncation.
- Continuation: the first 128 tokens of the held-out final assistant answer.
- Diagnostic: sparse and dense logits are evaluated from the same prefix, then the held-out token advances both paths.
- Repetitions: one run per model and retention ratio.

The complete commands, per-token traces, GPU readings, prompt/reference hashes, and timing counters are in `results.jsonl`. `manifest.json` fixes the model paths, cache formats, source-fixture hash, gate, and implementation commit.

## Correctness gate

All three 100%-retention runs passed exactly:

| Model/cache | Prompt tokens | Top-1 | Mean TV | Mean JS | State failures |
|---|---:|---:|---:|---:|---:|
| Qwen27 q8/Turbo4 gather | 57,483 | 1.0000 | 0 | ~0 | 0 |
| Gemma4 q8/Turbo4 gather | 59,281 | 1.0000 | 0 | ~0 | 0 |
| Qwen35 q8/q4 direct | 57,483 | 1.0000 | 0 | ~0 | 0 |

This is the key result: lower-retention differences are no longer confounded by the D=256 mixed-cache decode bug, D=512 sparse execution, KV traversal order, recurrent rollback, repeated prompt text, or hand-written `User:`/`Assistant:` formatting.

The actual prompts are below the 65,536-token budget because the packer drops whole turns. Qwen and Gemma tokenize the same 157 selected messages differently. Prompt and reference hashes are constant across ratios for each model.

## Quality and sparse-attention time

`Time saved` compares only the accumulated sparse probe time with the accumulated dense probe time inside the same run. It is not an end-to-end MTP throughput prediction.

| Model | Retained | Top-1 | Mean TV | P95 TV | Mean JS | Reference NLL delta | Time saved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen27 | 50% | 1.0000 | 0.0070 | 0.0212 | 0.000087 | -0.0026 | 16.8% |
| Qwen27 | 20% | 1.0000 | 0.0152 | 0.0512 | 0.000478 | -0.0055 | 29.0% |
| Qwen27 | 10% | 0.9843 | 0.0258 | 0.0880 | 0.001421 | -0.0158 | 33.1% |
| Gemma4 | 50% | 0.9921 | 0.0215 | 0.0656 | 0.001230 | +0.0043 | 1.7% |
| Gemma4 | 20% | 0.9764 | 0.0438 | 0.1344 | 0.003842 | -0.0174 | 24.0% |
| Gemma4 | 10% | 0.9606 | 0.0737 | 0.2157 | 0.010473 | -0.0538 | 31.6% |
| Qwen35 | 50% | 0.9921 | 0.0123 | 0.0375 | 0.000270 | -0.0008 | 24.4% |
| Qwen35 | 20% | 0.9843 | 0.0141 | 0.0429 | 0.000399 | -0.0027 | 39.5% |
| Qwen35 | 10% | 0.9764 | 0.0200 | 0.0674 | 0.000718 | +0.0013 | 44.8% |

## Interpretation

- Qwen27 at 20% is the strongest point in this sample: all 127 probed top-1 tokens match dense, mean TV is 0.0152, and sparse-attention time falls by 29.0%.
- Qwen35 is also robust. A strict 99% top-1 requirement selects 50%; 20% changes two of 127 top-1 choices but saves substantially more sparse-attention time.
- Gemma4 is the model that now warrants mask-selection work. At 20%, three top-1 choices differ and 14 steps have TV above 0.1. At 10%, five choices differ, 41 steps exceed TV 0.1, and 12 exceed TV 0.2.
- Gemma's largest deviations recur at the same continuation offsets as retention falls (notably offsets 24 and 42), which is consistent with missing important historical tokens rather than random state corruption.
- A negative reference-NLL delta does not mean the sparse distribution is more faithful; it only means it assigned the held-out token more probability on average. TV, JS, ranks, and top-1 agreement remain the fidelity measures.

## Suggested next experiments

1. Use Qwen27 20% and Qwen35 20-50% as the first hierarchical-design baselines.
2. Sweep Gemma around 30%, 40%, and 50%; 50% is faithful here but its gather overhead erases nearly all attention-time benefit.
3. For Gemma, inspect the selection plans at the repeated high-TV offsets and compare per-layer retained attention mass before changing the global retention ratio.
4. Measure the chosen ratios inside full MTP/hierarchical cycles. Same-prefix timing establishes the attention component but does not include MTP draft cost, dense verification batching, or committed tokens per cycle.
