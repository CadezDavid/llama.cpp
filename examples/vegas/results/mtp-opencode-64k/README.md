# Gemma OpenCode 64K MTP screening

This n=1 screening compares dense MTP with Vegas-sparse assistant drafting and dense target verification.

The prompt was extracted from OpenCode session `ses_2452593b5ffe6yJELmXEV6x9oM`, titled `Rebuilding MCP memory server from scratch`, in `/home/david/.local/share/opencode/opencode.db`. User and assistant text parts were retained through message index 238. The selected prefix ends after a user message; no message was partially included and the benchmark did not resize or repeat the transcript.

Extracted transcript SHA-256: `a111c468c5a4617a7ad22d166f692af8443cbd2ba402f1fe128aca0fd3b0fbea`

The test used Gemma 4 31B Q4_K_XL, the Q8_0 Gemma assistant, q8_0/turbo4 target and draft KV, 65,375 prompt tokens, 128 generated tokens, gamma 3, greedy decoding, batch one, ubatch 128, a 3% Vegas mask, and a 256-token floor.

| Mode | tok/s | Acceptance | Cycles | Draft ms/proposal | Verify ms/cycle | Output hash |
|---|---:|---:|---:|---:|---:|---|
| MTP | 32.683 | 41.42% | 57 | 3.877 | 56.841 | `51f163b77cf73651` |
| MTP+Vegas | 31.156 | 39.08% | 59 | 1.918 | 63.034 | `51f163b77cf73651` |

MTP+Vegas was 4.67% slower. Sparse drafting reduced assistant time per proposal by 50.5%, but acceptance fell slightly, two additional dense target verification cycles were needed, and target selection increased verification cost per cycle by 10.9%. The output hashes matched.

This is consistent with a workload-dependent crossover near 64K: assistant sparsity is already effective, but its savings are not large enough to tolerate lower acceptance and Vegas selection overhead.

A later gamma-1 screen on a newly reconstructed version of the same real
conversation is documented in
[`../gemma-64k-gamma1-2026-08-04/README.md`](../gemma-64k-gamma1-2026-08-04/README.md).
It remains negative at refresh intervals 1 and 2.
