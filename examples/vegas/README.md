# Vegas and CascadeSpec experiment closeout

Status: archived experiment on `feat/vegas`, 2026-08-06.

This directory is the entry point for the private Vegas work. The branch is a
useful, reproducible research snapshot, but it should not be merged wholesale
or treated as a production llama.cpp feature. The implementation and raw
measurements are preserved so individual ideas can be resumed or transplanted
later.

## Final conclusions

| Area | Status | Conclusion |
|---|---|---|
| Vegas sparse self-speculation | Complete experiment | It can improve long-context batch-one decoding, but the crossover depends strongly on the model, context, KV cache, and draft length. See [`PRIVATE_RESULTS.md`](PRIVATE_RESULTS.md). |
| MTP+Vegas | Complete experiment | This is the most credible result on the branch. It can reduce MTP draft cost and has measured wins on selected long-context Qwen configurations, but it is not a universal default. See [`MTP_VEGAS_RESULTS.md`](MTP_VEGAS_RESULTS.md). |
| Adaptive MTP gamma | Implemented and unit-tested | The current controller supports sparse gamma 1 through 10 and a dense-MTP fallback, uses measured costs, and computes entropy on the backend. The recorded eight-run matrices predate the final controller, so they are historical screening data rather than a benchmark of the final policy. See [`ADAPTIVE_GAMMA.md`](ADAPTIVE_GAMMA.md). |
| CascadeSpec hierarchy | Correctness-validated, performance-negative | The sparse middle target does not save enough per pass to pay for the additional target passes. It is slower than plain MTP on Qwen27, Gemma4, and Qwen35 at the tested 32K, 64K, and 128K contexts. Keep `mtp-hierarchical` diagnostic-only. |
| Recurrent-state rollback | Exact checkpoint/replay implemented | Periodic full checkpoints every four input rows plus at most three exact forward replays avoid retaining one full state per speculative token. The attempted compact algebraic inverse is numerically unsafe and is not used. |
| TurboQuant hierarchy | Intentionally deferred | Direct indexed q8/Turbo4 work is preserved, but batched CascadeSpec currently rejects TurboQuant KV caches. The final hierarchy was developed and benchmarked only with normal q4/q8 caches. |
| Prox and target FFN sparsity | Discontinued | These negative experiments and their evidence are retained in [`PROX_ARCHIVE.md`](PROX_ARCHIVE.md) and [`TARGET_FFN_SPARSITY.md`](TARGET_FFN_SPARSITY.md). |

The practical stopping decision is therefore:

```text
keep the branch as an experimental reference
prefer plain MTP or a measured MTP+Vegas configuration
do not enable CascadeSpec or adaptive gamma by default
```

## Final CascadeSpec design

The final fixed cadence is:

1. MTP proposes up to four tokens.
2. A sparse target batch verifies or corrects the proposal.
3. Steps 1 and 2 run for three sparse rounds.
4. A fourth MTP proposal is appended without a redundant sparse verification.
5. The dense target verifies the complete provisional block.

The provisional cap is 19 tokens, so the final dense batch contains at most 20
input rows including the preceding committed token. Recurrent Qwen models use
full state checkpoints every four rows and exact replay from the nearest
checkpoint. Gemma4 has no recurrent target state and only needs KV-prefix
trimming.

This corrects the earlier schedule that performed a sparse verification
immediately before the dense verification. It still loses because a sparse
target pass executes the target model's dense projections, normalization,
FFNs or experts, logits, and graph work; it only saves part of attention.

## Results that should be used

The final cadence measurements supersede the earlier low-memory and variable
horizon experiments when judging CascadeSpec as implemented at closeout.

### Final 32K comparison

All runs use gamma 4, q8_0/q4_0 target and draft KV caches, native chat
formatting, and a coherent OpenCode conversation fixture.

| Model | Plain MTP | CascadeSpec | Change |
|---|---:|---:|---:|
| Qwen3.6 27B | 55.26 tok/s | 30.77 tok/s | -44.3% |
| Gemma4 31B | 51.71 tok/s | 36.33 tok/s | -29.7% |
| Qwen3.6 35B-A3B | 168.02 tok/s | 90.14 tok/s | -46.4% |

Full telemetry and fixture provenance are in
[`results/cascadespec-cadence-32k-2026-08-06/`](results/cascadespec-cadence-32k-2026-08-06/).

### Final Qwen27 128K comparison

The prompt contains 131,038 naturally formatted tokens. Plain MTP reached
37.27 tok/s and CascadeSpec reached 26.32 tok/s, a 29.4% regression. A sparse
pass was 19.7% cheaper than a plain gamma-4 dense pass, but CascadeSpec ran 41
sparse plus 14 dense target passes instead of 33 dense passes.

Full telemetry and prompt provenance are in
[`results/cascadespec-cadence-qwen27-128k-2026-08-06/`](results/cascadespec-cadence-qwen27-128k-2026-08-06/).

### Broader 32K/64K study

The earlier optimization report covers attention-retention quality, direct
versus gathered kernels, proposal lengths, dense horizons through 50 tokens,
and repeated 64K comparisons:
[`results/CASCADESPEC_2026-08-06.md`](results/CASCADESPEC_2026-08-06.md).

Its low-memory replay description and horizon-8 recommendation describe the
design before the final four-round cadence. Use it for the component-level
measurements, not as the final end-to-end result.

## Correctness evidence

The branch contains the following independent gates:

- focused adaptive-gamma, hierarchical-policy, and same-prefix unit tests;
- 516 focused CUDA sparse-attention cases covering direct, gather, and auto
  kernels, normal Q4/Q8 cache combinations, 100% retention, unusual index
  orderings, and causal batches through 21 rows;
- a model-level recurrent replay test that checks restored recurrent tensors,
  positions, continuation logits, selected tokens, and output hashes for
  prefix lengths 0 through 20;
- benchmark telemetry for snapshot, rollback, position, and recurrent-prefix
  restore failures; all are zero in the final cadence runs;
- natural conversation fixtures assembled at message boundaries and rendered
  with each model's native chat template.

Output hashes are useful within an identical execution shape, but they are not
a valid cross-gamma or cross-horizon correctness oracle. Different dense batch
lengths change floating-point reduction order and can change a greedy token
near a probability tie. State replay is instead validated against the same
20-row target execution at every restored prefix.

The rejected algebraic inverse experiment is documented separately in
[`results/recurrent-undo-2026-08-06/`](results/recurrent-undo-2026-08-06/).

### Closeout verification

The following checks were rerun successfully on 2026-08-06 after the final
implementation commit and before the documentation closeout commit:

- all six focused build targets listed below;
- all three Vegas policy tests, with 3/3 passing;
- `test-gated-delta-undo`, with exact checkpoint/replay at every tested prefix
  and the expected growing error in the unused algebraic inverse;
- `test-recurrent-replay-model` on Qwen3.6-27B, with exact recurrent tensors,
  positions, logits, selected tokens, and continuation hashes for every valid
  prefix from 0 through 20.

The complete generic `FLASH_ATTN_EXT` backend matrix is much broader than the
Vegas cases and was not repeated as part of closeout. The focused 516-case
sparse-attention result above is the preserved kernel validation from the
optimization milestone; subsequent commits changed recurrent-state handling,
the cadence controller, benchmark tooling, and documentation rather than the
sparse-attention kernels.

## Deliberately unfinished work

- CascadeSpec has no TurboQuant-compatible batched production path. The CLI
  rejects hierarchical caches other than q4_0/q4_0, q8_0/q4_0, and q8_0/q8_0.
- The selective-snapshot idea is not implemented. If recurrent inversion is
  revisited, snapshot only heads or tiles that fail an actual float32
  round-trip test, and first measure the unsafe rate and GPU-pool memory.
- The final adaptive-gamma controller has unit coverage but no fresh full
  eight-cell matrix after removal of forced gamma-10 calibration and addition
  of the dense-MTP action.
- No production integration, upstream proposal, public API stability promise,
  or multi-GPU/general-hardware validation was attempted.

These are recorded resume points, not branch-closeout blockers.

## Reproducing the closeout checks

Build the focused targets:

```bash
cmake --build build-vegas --target \
    llama-vegas \
    test-vegas-adaptive-gamma \
    test-vegas-hierarchical-policy \
    test-vegas-same-prefix \
    test-gated-delta-undo \
    test-recurrent-replay-model -j4
```

Run the policy tests:

```bash
ctest --test-dir build-vegas --output-on-failure \
    -R 'test-vegas-(adaptive-gamma|hierarchical-policy|same-prefix)'
```

Run the recurrent arithmetic and model-level replay checks:

```bash
build-vegas/bin/test-gated-delta-undo
build-vegas/bin/test-recurrent-replay-model \
    -m /home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    -ngl 66
```

The fixed 32K benchmark runner is resumable and writes a manifest, JSONL raw
records, and a compact summary:

```bash
python examples/vegas/cascadespec_benchmark.py \
    --output-dir /tmp/cascadespec-closeout
```

## Branch and integration notes

- This is a private CUDA experiment, measured on one RTX 3090.
- Result directories preserve exact commands, prompt hashes, source commits,
  raw JSONL telemetry, and summaries where applicable.
- The branch includes an upstream synchronization merge and an earlier WebUI
  attachment commit in its ancestry. It is not a clean single-feature patch
  series; transplant individual commits or code areas rather than merging the
  branch wholesale.
- No push is implied by a local closeout commit. Check `git status -sb` and
  compare `HEAD` with `origin/feat/vegas` before deleting a worktree or relying
  on the remote as the archive.
