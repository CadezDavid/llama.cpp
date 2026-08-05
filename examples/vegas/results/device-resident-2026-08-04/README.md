# Vegas device-resident sparse plan experiment

Date: 2026-08-04

Branch: `feat/vegas`

Hardware: NVIDIA RTX 3090, batch one, greedy decoding

This is a focused engineering check, not a new benchmark matrix. It evaluates the device-resident sparse-index path and one bounded mask generalization after implementation. No DSpark comparison was run.

## Implemented changes

- Verification top-k is copied into a persistent CUDA tensor owned by the target context.
- MTP drafting references that tensor directly. Sparse indices are no longer downloaded, sorted on the CPU, copied into host vectors, or uploaded for every draft graph.
- A CUDA event orders target selection and draft attention when the contexts use different CUDA streams.
- The changing dense-suffix length is passed as a FlashAttention launch parameter. The device tensor contains indices only.
- GGML now exposes a generic optional sparse-KV index input for FlashAttention instead of a Vegas-named low-level API.
- Quantized sparse attention continues to use direct indexed VEC kernels. The available sparse MMA design requires an F16 K/V scratch conversion of the full cache, which would defeat Vegas for q8, q4, and TurboQuant KV caches.
- An optional fixed-prefix anchor budget was added with `--vegas-anchor-tokens`. The recent suffix remains dense and the rest of the historical budget remains verification-selected.

The selector itself is unchanged: it still performs a separate QK operation and therefore rereads K after verification. Fusing score collection into verification FlashAttention remains future work.

## Correctness

The ratio-1 test compares dense MTP with MTP+Vegas while selecting the entire historical prefix. Both produced:

- output hash `b9e73707b8752e42`;
- 41 accepted of 63 drafted tokens;
- 14 rejected draft tokens.

The genuinely sparse anchor-enabled smoke test also completed and produced the same target output hash. See `ratio1-correctness.jsonl` and `sparse-anchor-smoke.jsonl`.

`test-batch-alloc` passed all 198 assertions. A broad CUDA `FLASH_ATTN_EXT` backend run was started and showed no failures in the executed cases, but was intentionally stopped because the filter expands to thousands of unrelated dense cases; it is not counted as a completed test.

## Focused performance results

All long-context runs generated 128 tokens from real Codex or OpenCode session text. Each new configuration was run once. Historical comparisons are used only when the prompt, settings, output hash, and acceptance behavior make the comparison meaningful.

| Model and configuration | Reference | New result | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Gemma 4 31B, 128K, q8/turbo4, Q8 assistant, gamma 1, ratio 3%, 16 anchors | 23.633 tok/s | 23.227 tok/s | -1.7% | No throughput gain in this run. Acceptance fell from 82.6% to 80.0%. |
| Qwen3.6 27B, 112K, q8/turbo4, gamma 4, ratio 10%, layer 15, refresh 2, no anchors | 30.271 tok/s | 30.546 tok/s | +0.9% | Small systems gain; acceptance and output hash are unchanged. |
| Qwen3.6 27B, same run with 16 anchors | 30.546 tok/s | 30.549 tok/s | +0.01% | Anchors are neutral. |
| Qwen3.6 35B-A3B, 112K, q4/q4, gamma 5, ratio 3%, 16 anchors | dense MTP 141.330 tok/s | Vegas 164.334 tok/s | +16.3% | Strong same-prompt win with identical acceptance and output hash. |

The Gemma historical reference is `../unified-strategy/gemma-128k-q8-turbo4-auto-reserve-fixed.jsonl`. The Qwen 27B reference is `../unified-strategy/qwen27-112k-q8-turbo4-auto-resolved-before-init.jsonl`. The original Qwen 35B prompt fixture was no longer present, so a new same-prompt dense control was run instead of making an invalid comparison to the old 149-150 tok/s results.

### Where the systems work helped

| Case | Old collection | New collection | Reduction |
| --- | ---: | ---: | ---: |
| Gemma 128K | 9.052 ms | 0.201 ms | 97.8% |
| Qwen 27B 112K | 8.244 ms | 0.052 ms | 99.4% |
| Earlier Qwen 35B 113.6K fixture | about 3.17 ms | 0.061 ms on the new 112K fixture | about 98%, prompts differ |

`collect_ms` now measures host bookkeeping and event recording, not index transfer. The selector QK, reduction, and top-k remain part of verification time.

For the paired Qwen 35B run:

- draft time fell from 343.102 ms to 142.332 ms (-58.5%);
- verification time rose from 531.979 ms to 602.388 ms (+13.2%);
- total generation time fell from 905.681 ms to 778.903 ms;
- acceptance was identical at 82.9%;
- output hash was identical: `770eb4c97a70a659`.

## Decision

The device-resident path is worth keeping. It removes almost all measured host collection overhead, preserves correctness, gives a small improvement in the marginal Qwen 27B case, and retains a strong Qwen 35B win.

The fixed-prefix anchor generalization is not supported as a default by these results. It was neutral on Qwen 27B and slightly reduced Gemma acceptance. The option remains available for future prompt-shift or sink-token experiments, but automatic policy leaves it disabled.

These changes do not make Vegas universal. Gemma remains dominated by target verification, Qwen 27B has limited economic room because only part of the model uses global attention, and Qwen 35B q4/q4 remains the clearest systems win. The next substantial performance opportunity is fused verification-time score collection, not more gamma, ratio, or anchor tuning.

## Raw files

- `gemma-128k-q8-turbo4-anchor16.jsonl`
- `qwen27-112k-q8-turbo4-anchor0.jsonl`
- `qwen27-112k-q8-turbo4-anchor16.jsonl`
- `qwen35-112k-q4-q4-dense-mtp.jsonl`
- `qwen35-112k-q4-q4-anchor16.jsonl`
- `ratio1-correctness.jsonl`
- `sparse-anchor-smoke.jsonl`
