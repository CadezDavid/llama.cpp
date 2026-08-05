# Hierarchical MTP + sparse-target experiment

This experiment implements and exercises a three-level speculative path:

```text
MTP + Vegas proposes
        -> sparse target verifies, corrects, and extends
        -> dense target verifies the accumulated block exactly
```

The middle stage runs at most three rounds, targets eight provisional tokens,
has a hard cap of ten tokens, and stops after two sparse-target corrections.
All runs are greedy, generate 128 tokens from the 65,536-token OpenCode
technical-conversation fixture, and use one repetition. These are
hierarchy-only measurements, not paired speed comparisons against MTP or
MTP+Vegas controls.

## Results

| Configuration | tok/s | Dense cycles | Committed / dense cycle | Sparse agreement | Dense acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B q8/turbo4 | 11.619 | 46 | 2.761 | 47.0% | 47.4% |
| Gemma 4 31B q8/turbo4 | 8.757 | 84 | 1.512 | 17.1% | 21.5% |
| Qwen3.6 35B-A3B q8/q4 | 46.340 | 22 | 5.773 | 77.3% | 70.0% |

Every completed run reported zero rollback failures, zero recurrent/MTP
snapshot failures, and zero target/draft position mismatches. All entropy and
top-probability samples were computed on the backend; CPU fallback samples
were zero.

The saved output hashes are:

- Qwen27: `2a02668cbda00dbe`
- Gemma: `8ac8088e2f86eb88`
- Qwen35 q8/q4: `e8cb7a20e21062a5`

The focused 0.8B correctness check used during implementation matched dense
baseline for both 8 and 32 generated tokens. The 32-token hash was
`2f223472d9300160` in both modes.

## Diagnostic interpretation

Qwen35 is the only promising hierarchy in this slice. It reached the target
length in 8 of 22 cycles, stopped on the two-correction guardrail in 12 cycles,
and committed almost six tokens per dense verification. Its generation time
was split roughly as follows: 564 ms MTP drafting, 229 ms MTP state processing,
1,326 ms sparse-target work, and 578 ms dense verification.

Qwen27 stopped on the correction cap in 44 of 46 cycles. The sparse middle
stage cost 5,271 ms, versus 3,267 ms for dense verification, while producing
only 2.76 committed tokens per dense cycle. This policy should normally bypass
the hierarchy for this regime or stop after its first correction.

Gemma stopped on the correction cap in 82 of 84 cycles and committed only 1.51
tokens per dense verification. Its sparse-target agreement was 17.1%, so the
middle level mostly added work without constructing a useful longer block.
This configuration should disable the hierarchy quickly.

The next controller should therefore gate entry into the sparse-target middle
level and adapt the correction budget. A practical first policy is to require
recent sparse agreement and dense acceptance above a threshold, use one
correction by default, and permit the current two-correction/eight-token path
only for Qwen35-like confident stretches.

## Artifacts

- `results.jsonl`: complete aggregate records and per-cycle token/entropy/timing traces
- `summary.json`: compact health and performance summary
- `examples/vegas/hierarchical_matrix.py`: resumable command matrix

The first Qwen35 attempt exposed a one-token recent-window sizing bug at a
23-token active span. Commit `36170494a` reserves the MTP boundary token; the
successful result above is from the corrected rerun.
