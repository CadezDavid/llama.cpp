# Gemma4 20% same-prefix diagnostic at 64K

This run isolates sparse-target fidelity from MTP proposals and hierarchical
multi-round behavior. For every generated position it:

1. checkpoints the target state;
2. decodes one token with the current 20% Vegas plan;
3. copies the sparse logits and rolls the decode back;
4. restores recurrent state and verifies the exact KV position;
5. decodes the same input densely from the identical prefix;
6. compares the complete raw-softmax distributions; and
7. advances only along the dense greedy path.

The model, 65,536-token OpenCode prompt, layer 59 selector, q8/turbo4 KV
cache, and sampling settings match the previous Gemma4 hierarchical run.

## Result

| Metric | Value |
| --- | ---: |
| Same-prefix probes | 127 |
| Top-1 matches | 119 |
| Aggregate top-1 agreement | 93.70% |
| Mean dense-to-sparse KL | 0.4641 |
| Mean Jensen-Shannon divergence | 0.0977 |
| Mean total-variation distance | 0.2442 |
| Mean top-10 overlap | 44.41% |
| Dense entropy | 1.3360 |
| Sparse entropy | 2.0494 |
| Snapshot / rollback / position failures | 0 / 0 / 0 |

The aggregate top-1 result is misleadingly favorable. Token ID 715 accounts
for 122 of the 127 dense continuation tokens. Sparse and dense therefore agree
on a degenerate repeated continuation for most of the trace. In the first ten
comparisons, before that repetition dominates, only 2 of 10 top tokens match.

The eight mismatches are not merely close argmax swaps. On those positions the
dense top token has mean probability 56.28%, while sparse assigns it only
8.39%. Their mean total-variation distance is 0.7625 and mean top-10 overlap is
33.75%. Even across the full trace, top-10 overlap is only 44.41% and sparse is
substantially more entropic. The 20% target therefore preserves the greedy
token on the repetitive suffix but does not preserve a dense-like
distribution.

The diagnostic output hash (`8bf80a66542aebae`) exactly matches a separate
dense baseline using the same prompt and generation settings. This confirms
that the sparse probe rollback does not perturb the authoritative dense path
for this run. It also leaves the ratio-dependent and hierarchy-dependent
hashes from the earlier hierarchical experiments as a separate correctness
issue.

`results.jsonl` contains the aggregate and all 127 per-position comparisons.
`baseline.jsonl` contains the paired dense reference. `summary.json` extracts
the non-degenerate prefix, distribution, mismatch, integrity, and hash data.
