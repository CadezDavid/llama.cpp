# Vegas verification graph reuse

## Problem

The verification graph identity included the current Vegas prefix length. The
prefix advances during generation, so refresh-1 rebuilt the target graph once
per speculative cycle. On the Gemma 64K gamma-1 case this reduced
`graphs_reused` from 335 for dense MTP to 254 for Vegas.

## Implementation

The collector now uses score storage sized to the stable KV tensor capacity.
The active prefix is supplied as a one-element I32 graph input, and an F32
input mask excludes score positions beyond that prefix before top-k. The CUDA
FlashAttention collector reads the prefix through the device pointer. This is
required for CUDA graph correctness: changing an integer operation parameter
on the host does not update a kernel argument already captured in a CUDA graph.

The mask algorithm, boundary queries, selected layer, ratio, and speculative
policy are unchanged. Top-k remains part of graph identity, so a graph rebuild
still occurs when the rounded top-k size changes.

## Correctness checks

The 2K Qwen3.5 0.8B checks used 64 generated tokens.

- A 100% selection run matched dense MTP exactly: output hash
  `b9e73707b8752e42`, 41/63 accepted, and 22 cycles.
- A 3% sparse run retained output hash `b9e73707b8752e42`, 39/69 accepted,
  and 24 cycles.
- `test-batch-alloc` passed all 198 assertions.

The decisive Gemma run matched the no-reuse control exactly: output hash
`df6c42035ba0bb18`, 44/83 accepted, and 83 cycles.

## Gemma 64K result

Setup: Gemma 4 31B Q4_K_XL target, Q8_0 assistant, q8_0/turbo4 target and
assistant KV, 65,071 real OpenCode conversation tokens, 128 generated tokens,
gamma 1, 3% selection, seed 1234.

| Configuration | tok/s | Accepted/drafted | Cycles | Verify ms | Graphs reused |
|---|---:|---:|---:|---:|---:|
| Dense MTP, preserved earlier run | 28.425 | 45/82 | 82 | 4179.836 | 335 |
| Vegas refresh 1, old implementation | 27.648 | 44/83 | 83 | 4443.864 | 254 |
| Vegas refresh 1, same-build no-reuse control | 27.709 | 44/83 | 83 | 4432.026 | 254 |
| Vegas refresh 1, device-prefix reuse | 28.802 | 44/83 | 83 | 4262.666 | 333 |
| Vegas refresh 2, device-prefix implementation | 27.652 | 44/83 | 83 | 4447.912 | 254 |

Against the same-build no-reuse control, device-prefix reuse reduced
verification by 169.360 ms (3.8%), reduced total generation time from
4619.406 to 4444.064 ms (3.8%), and improved throughput by 3.9%. It made the
refresh-1 Vegas run 1.3% faster than the preserved dense result, although this
single-run difference is near measurement noise.

Refresh 2 still alternates disabled and collector graph shapes. The context
retains only the immediately previous graph, so this mode remains at 254 graph
reuses. Fixing that without changing refresh semantics requires either a
two-entry graph cache or separate collector and active plan storage. It was not
folded into this patch.

## Rejected approaches

Two failed implementations are preserved because they exposed important
correctness constraints.

1. Freezing the historical prefix at the prompt boundary restored reuse but
   changed the selector. Gemma acceptance collapsed to 1/125 and throughput to
   19.155 tok/s. This is `gemma-64k-g1-r03-refresh1.jsonl`.
2. Updating the prefix only as a host-side operation parameter produced 335
   reuses but CUDA graph capture retained a stale kernel argument. Gemma changed
   to 41/85 accepted and 86 cycles. This is
   `gemma-64k-runtime-prefix-r1.jsonl`.

The valid same-build control is
`gemma-64k-runtime-prefix-no-reuse-control.jsonl`; the final result is
`gemma-64k-device-prefix-r1.jsonl`.
