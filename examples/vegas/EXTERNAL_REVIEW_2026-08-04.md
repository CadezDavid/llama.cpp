# External review of the Vegas experiment

Date received: 2026-08-04

This is an external model's assessment of the private Vegas experiment. It is
preserved for planning and comparison, not as a verified project conclusion.
Its external links, upstream statements, release information, and performance
claims have not been independently verified in this repository.

## My conclusion

**Continue Vegas for one more bounded engineering sprint, but stop tuning the current implementation.**

You have probably exhausted the **current design space**, not the underlying idea. The remaining opportunity is not another gamma, ratio, query, head, or refresh sweep. It is a different implementation architecture:

1. Keep selection and indices entirely on the GPU.
2. Make sparse KV indices a generic FlashAttention input rather than a Vegas-specific CUDA path.
3. Add a measured benefit gate instead of model-specific policy tables.
4. Test one stronger mask generalization, probably two layer groups plus fixed anchors.
5. Compare against DSpark as the baseline to beat.

There is also an important correction to your acceptance criterion. A sparse-draft-attention optimization will never help "most models most of the time" in a literal sense. It cannot materially help short contexts, models whose draft attention is already cheap, models dominated by recurrent/local-attention or feed-forward layers, or DSpark-style drafters that do not repeatedly scan the target's long KV cache.

The right criterion is:

> **Does automatic Vegas avoid regressions in almost all eligible long-context configurations and provide a meaningful gain across several model and drafter architectures?**

That is both achievable and upstream-worthy. If you require Vegas to be universally faster without gating, I would stop now.

## What your results actually prove

Your final report establishes three distinct regimes:

| Regime                                           |       Dense MTP | Best automatic result |                                                   Effect |
| ------------------------------------------------ | --------------: | --------------------: | -------------------------------------------------------: |
| Gemma 4 31B, separate assistant, 128K, q8/turbo4 |    14.619 tok/s |          23.633 tok/s |                                               **+61.7%** |
| Qwen3.6 35B-A3B, q4/q4, roughly 114K             | about 123 tok/s |       about 150 tok/s |                                     **+21.4% to +21.9%** |
| Qwen3.6 27B, q8/turbo4, 112K                     |    29.455 tok/s |          30.271 tok/s |                                                **+2.8%** |
| Qwen3.6 27B, q8/turbo4, 62K                      |    33.720 tok/s |          30.938 tok/s |                **-8.3%** before the threshold disabled it |
| Qwen3.6 35B-A3B, q8/turbo4, 62K                  |   127.565 tok/s |       about 134 tok/s | Gain came from **q4/q4 draft cache; Vegas was disabled** |

These are not random benchmark fluctuations. They show two genuine, substantial Vegas wins, one marginal win, and several configurations where sparse attention simply has insufficient economic room.

The Qwen35 single-selection work is especially convincing. Removing masks that were computed but never consumed cut collection work by roughly 89%, while a paired q4/q4 test still showed about **+21.7%**. A second prompt still gave about **+10.2%**. At the same time, q8/turbo4 Qwen35, Qwen27, and 64K Gemma did not benefit from the same configuration.

The Qwen27 recovery work leads to a different conclusion. You recovered a small positive result by changing the selected layer, refresh interval, gamma, and ratio, but the confirmed 62K gain was only about **0.8%**, with roughly 1-2% in the longer-context screens. That report itself correctly concludes that more manual sweeps are unlikely to move the needle; the next meaningful work is lower-overhead collection and a more robust policy.

So the evidence says:

* **Vegas is a real optimization.**
* **It is not currently a general optimization.**
* **The best cases are strong enough to justify one more systems-focused attempt.**
* **There is no evidence that another parameter sweep will make it general.**

## Is your implementation essentially optimal?

### Algorithmically, it is near a local optimum

Within the constraint of "one selected mask, applied broadly, selected using a small number of target queries," your implementation is already well explored.

You have established that:

* Boundary queries are sufficient; collecting every verification query is unnecessary.
* Query/head reduction did not recover enough performance.
* One selected target layer can be much cheaper than producing masks for every layer.
* Qwen27 prefers an earlier layer and slower refresh.
* Qwen35 prefers the final layer and immediate refresh.
* Dynamic gamma driven by recent acceptance oscillates and loses badly.
* Throughput exploration is too costly to amortize over ordinary requests.
* The useful sparsity ratio and gamma depend on the drafter and cache format.
* Resolving the auto policy before initializing the speculative helper matters substantially.

That is a serious amount of negative and positive evidence. I would not spend more time trying gamma 4 versus 5, 5% versus 7%, or another single-head heuristic.

### Systems-wise, it is not close to optimal

The paper's "free oracle" is not actually free in your port. The paper modifies FlashAttention so that verification collects the first-draft and bonus-token logits while it is already streaming the full KV cache. Your implementation instead constructs `q_score`, reads `k_prefix` again through a separate `ggml_mul_mat`, reduces the result, and runs top-k as another graph path. The paper also evaluates multi-request H100 serving, whereas your implementation is currently a single-sequence CUDA path.

That leaves several material gaps.

#### 1. The KV cache is read twice during selection

Verification FlashAttention reads K and V. Your selector then reads K again to recompute QK scores. That is precisely the bandwidth operation Vegas is supposed to avoid.

This is particularly harmful in borderline configurations such as Qwen27, where the sparse draft saves meaningful time but the selection and extra verification work consume almost all of it. It matters less in the Qwen35 q4/q4 case because the draft saving is so large.

Also, your reported `collect_ms` understates total selection cost. The QK matmul and top-k are inside the verification graph and therefore largely appear under verification time; `collect_ms` mainly measures synchronization, transfer, and host processing.

#### 2. The mask takes a GPU -> CPU -> GPU trip

`vegas_collect_indices()` synchronizes the context, downloads selected indices, sorts them on the CPU, and the draft graph later uploads them again. This is tolerable in a batch-one experiment but is hostile to:

* CUDA graph execution;
* overlapping target and draft work;
* multiple server slots;
* continuous batching;
* separate target and draft streams;
* future multi-GPU support.

The selected index tensor should remain device-resident from verification through drafting.

A practical first improvement does not require fused FlashAttention. Your top-k already exists on the GPU. Allocate a persistent backend tensor and copy top-k into it device-to-device. Then either consume the unsorted indices directly or order them on-device. A 128K-bit membership bitmap followed by an ordered compaction is only 16 KiB and may be simpler than a general GPU sort.

#### 3. Sparse attention forces the CUDA VEC path

When sparse indices are present, the current CUDA dispatch diverts directly to the vector FlashAttention path. The index list is encoded through optional input state and special metadata, including a sentinel-style sparse mode. That is compact for a prototype, but it bypasses the normal kernel-selection machinery and is difficult to extend to MMA, batching, other backends, or native sparse-attention models.

There is already an open llama.cpp draft PR adding a generic optional top-k index tensor to the MMA FlashAttention path for DeepSeek/GLM-style sparse attention. That abstraction is much closer to what upstream will accept than a Vegas-specific sparse mode. You should coordinate with or build on that work rather than maintaining a second sparse-attention interface.

Two implementations are worth comparing:

* **Direct indexed FlashAttention:** read selected KV rows in place.
* **Compact draft KV view:** gather selected rows once per speculative cycle and run the ordinary optimized attention kernel over contiguous K/V.

Direct indexing is simpler and avoids copying. Gathering may win when the same selection is reused for several draft tokens, because it restores locality and can reuse ordinary CUDA, Metal, HIP, or CPU attention kernels. There is no reliable way to choose between them without profiling on your 3090.

#### 4. The single global mask is probably the largest remaining algorithmic compromise

The original Vegas design naturally produces a selection for each attention layer. In MTP mode, your implementation ended up consuming one target-layer selection and broadcasting it broadly. That was an excellent simplification and removed a great deal of wasted collection work, but it also creates a brittle assumption:

> Tokens important at one target layer are sufficiently important in every draft-attention layer.

Your results show that the best representative layer differs sharply across architectures. Qwen27 prefers approximately quarter depth, while Qwen35 prefers the final layer. That strongly suggests that there is no universally optimal single layer.

I would not jump back to per-layer masks. That would recreate the overhead you successfully eliminated. The most promising middle ground is **two or four layer groups**:

* early or quarter-depth;
* late or final-depth;
* optionally middle groups if two are insufficient.

Produce one mask per group and map each draft layer to the nearest group. For a separate assistant with a different number of layers, map by normalized depth. Another simple variant is to aggregate scores from two representative layers and produce one consensus mask.

This is the one mask-quality generalization most likely to recover broader acceptance without multiplying overhead by the number of layers.

#### 5. The current implementation cannot represent the paper's main serving regime

The context currently requires one sequence, one fully offloaded CUDA device, supported cache formats, causal FlashAttention, and no ALiBi. That makes sense for your RTX 3090 experiments, but it excludes `llama-server -np N` and continuous batching - the environment in which the paper reports its strongest gains.

A generic representation should look roughly like this:

```cpp
struct sparse_kv_view {
    ggml_tensor * indices;      // concatenated selected KV positions
    ggml_tensor * offsets;      // [n_groups * n_seq + 1]
    ggml_tensor * recent_start; // one value per sequence
    ggml_tensor * recent_count; // one value per sequence
};
```

This supports ragged selections per request and optionally per layer group. Fixed maximum tensor capacities plus active lengths would preserve CUDA graph reuse.

Multi-sequence support is not necessary to decide whether the algorithm works on your 3090, but it is necessary before presenting Vegas as an upstream server feature.

#### 6. The automatic policy is empirical rather than portable

The current policy is sensible, but it encodes conclusions from three model families and one GPU:

* 48K or 96K thresholds;
* special treatment of q4/q4;
* different gamma and selected layers for dense, MoE, and separate-assistant configurations;
* a context-dependent ratio formula.

That is good experimental code. It is not a general llama.cpp policy.

A general policy should estimate this inequality:

```
(C_select + gamma C_sparse-draft + C_verify(gamma)) / E[A_sparse]
<
(gamma C_dense-draft + C_verify(gamma)) / E[A_dense]
```

where `A` is the number of output tokens obtained per speculative cycle.

The important point is that **acceptance alone is not enough**. Qwen35 q8/turbo4 has little room because dense drafting is already cheap. Qwen27 can save draft time and still lose because a few more target verification cycles cost more than the saving.

A portable gate can use:

* measured dense and sparse attention timings;
* target verification cost at the chosen gamma;
* fraction of layers using full attention;
* KV bytes per token and cache format;
* recent accepted tokens per cycle;
* selected-mask stability;
* backend and GPU bandwidth.

Use one warmup period and at most one dense/sparse switch with hysteresis. Keep gamma fixed for a request. Your failed dynamic-gamma experiments are good evidence not to build an online bandit into the first upstream version.

## The generalizations most worth testing

### First priority: a completely device-resident mask path

This is the decisive systems experiment.

The minimal version is:

1. Verification graph computes top-k on the GPU, as it already does.
2. Copy indices asynchronously into a persistent draft-plan tensor.
3. Perform ordering or compaction on the GPU only if measurements show it matters.
4. Make the draft graph depend on that tensor without synchronizing the CPU.
5. Keep the same tensor allocation and maximum shape across cycles.

Then profile the true selector kernels separately from target verification. This may be sufficient to turn some Qwen27 marginal cases into modest wins.

The higher-performance version fuses first/bonus-query score accumulation into verification FlashAttention. It should directly produce a reduced score per prefix token, not materialize all head-by-token logits. A two-stage GPU top-k - block candidates followed by exact token selection - would avoid writing a full score matrix.

I would implement the device-resident separate selector first. It is simpler, provides a clean measurement, and tells you whether full fusion is worth the additional CUDA complexity.

### Second priority: generic sparse KV indices

Rename the mechanism around what it does, not around Vegas:

* optional sparse KV indices;
* optional dense recent suffix;
* per-sequence offsets;
* optional layer-group selection.

Then Vegas becomes one producer of that sparse view. Other producers can include:

* model-native lightning or sparse indexers;
* fixed sliding windows;
* retrieval or heavy-hitter selectors;
* future cache-management policies.

That gives the low-level work value even if verification-guided selection never becomes the default speculative strategy. It also aligns directly with the existing sparse-MMA upstream PR.

### Third priority: a hybrid mask rather than pure top-k

The safest general mask is:

* a mandatory recent window;
* BOS and attention-sink positions;
* optionally application-pinned ranges such as system prompts or modality boundaries;
* verification-selected historical tokens with the remaining budget.

Your kernel already has the core idea of a selected historical prefix plus a dense recent suffix. Extending it to a few pinned ranges is cheap and could protect against topic switches, tool-call boundaries, and multimodal blocks.

For locality, test 16-token block selection alongside token selection. The paper found token pages as fast as 16-token pages on H100, but that does not establish the same result on Ampere with quantized llama.cpp KV layouts.

### Fourth priority: two representative layer groups

Test exactly three alternatives, not a huge sweep:

* current best single layer;
* two masks from quarter-depth and final-depth;
* one consensus mask obtained by aggregating those two score vectors before top-k.

If Qwen27 acceptance improves enough to make 64K and 112K reliably positive without damaging Qwen35 or Gemma, you have found a genuine generalization. If it does not, per-layer differences are probably too model-specific to solve cheaply.

### Fifth priority: adaptive token budget

A fixed ratio plus minimum and maximum token caps is already reasonable. A stronger version would choose the smallest `k` whose scores indicate sufficient concentration.

Raw logits are not ideal for this because their scales can differ by head and layer. Possible robust statistics include:

* per-head ranks before aggregation;
* a union of small per-KV-head selections;
* softmax-normalized mass using the verification layer's log-sum-exp;
* score margin around the chosen cutoff;
* stability or Jaccard overlap with the previous mask.

This is promising but not first-sprint work. It risks replacing a 500-line optimization with a much larger policy system.

## Can the implementation be made smaller?

The core implementation is not actually bloated. The initial patch added roughly **468 core lines** and removed about 33 across GGML, CUDA, graph, and context code; the 659-line `llama-vegas` executable and the large result directory are research infrastructure, not production implementation.

The problem is not raw line count. It is that the feature currently crosses many layers with Vegas-specific concepts:

* `llama_vegas_state`;
* enable/mode/pause/resume/collect/copy functions;
* per-layer host vectors;
* graph inputs;
* FlashAttention metadata conventions;
* speculative-loop policy.

For an upstreamable version, I would reduce it to:

1. One generic optional sparse-KV input in FlashAttention.
2. One persistent `sparse_kv_view` owned by the speculative engine or context.
3. One device selector operation that writes that view.
4. One MTP integration that chooses dense or sparse drafting.

Keep standalone self-speculation, dense-spec comparison, automatic-policy experiments, and all reporting code in `examples/vegas`. Do not initially upstream all six run modes.

I would also split out the **q4/q4 draft-cache policy**. Your Qwen35 q8/turbo4 result showed that this simple change can outperform adding Vegas. It is independently useful, much easier to explain, and likely much easier to merge.

## Does DSpark make Vegas obsolete?

### For some targets, probably

DSpark attacks speculative decoding at a more fundamental level. It combines a parallel draft backbone with a lightweight sequential Markov-style head and uses confidence-scheduled, hardware-aware verification. DeepSeek reports 60-85% higher per-user generation speed than its MTP-1 production baseline at matched throughput on DeepSeek-V4-Flash. That is a serving result, not a prediction for a 3090, but it sets a high bar. ([arXiv][1])

Basic DSpark support has already been merged into llama.cpp. The merged implementation reuses the DFlash machinery and adds the Markov head; the confidence head is loaded but was not yet used for scheduling in that first PR. Its Qwen3-8B RTX 4090 benchmark reported 1.88x versus no draft and 1.21x versus DFlash, although that is not a comparison against your MTP+Vegas setup or a long-context 27B target.

For a model that has:

* a high-quality matched DSpark checkpoint;
* sufficient memory for the additional drafter;
* good runtime support;
* training data matching the intended domain and thinking mode;

I expect DSpark to be the more attractive main speculative path. Its draft cost is not the same repeated long-context autoregressive attention cost that Vegas reduces. Therefore, Vegas and DSpark are mainly **competitors**, not automatically complementary optimizations.

### But DSpark does not make the general sparse-KV work obsolete

Official DeepSpec checkpoints currently cover Qwen3 4B, 8B, and 14B plus Gemma 4 12B. The official training workflow is target-specific, warns that thinking or domain-specific use may need retraining, assumes eight GPUs in its default scripts, and notes that the default Qwen3-4B target cache can be around 38 TB. ([GitHub][2])

Community Qwen3.6-27B DSpark checkpoints already exist, but their reported results vary substantially. One early checkpoint reports a measured 1.28-1.36x improvement and explicitly notes its limited 10,000-sample training and lack of long-context evaluation. Another reports much larger 1.6-2.7x gains in particular configurations. These are valuable experiments, not yet a stable universal baseline. ([Hugging Face][3])

Vegas retains important advantages:

* no drafter training;
* no target-specific sidecar requirement;
* little additional weight memory;
* immediate applicability to new checkpoints and private finetunes;
* potential usefulness when a DSpark sidecar would reduce the context that fits in VRAM;
* applicability to ordinary MTP and autoregressive assistants.

But the existence of DSpark changes the goal. Vegas no longer needs to become the best universal speculative decoder. It needs to be the best **training-free, low-memory, long-context draft-attention option**.

### The Qwen3.8 concern appears to conflate two releases

The Qwen3.8 announcement I could verify is the large **Qwen3.8-Max** flagship. I could not verify an official Qwen3.8-27B model card or an official paired DSpark drafter. The currently official 27B release is Qwen3.6-27B, which already includes native multi-step MTP. ([Reuters][4])

Qwen3.6-27B also illustrates a fundamental limit on Vegas: its 64-block language model repeats three Gated DeltaNet blocks followed by one gated-attention block. Only one quarter of those blocks uses the long-context attention mechanism Vegas can sparsify. Even a perfect sparse implementation can therefore accelerate only part of the draft model. ([Hugging Face][5])

A future Qwen3.8-27B with an official DSpark head would substantially weaken the case for Vegas on that one target. It would not invalidate a generic sparse-KV primitive, and it would not cover Gemma, private finetunes, or models without paired drafters.

I would not wait for that hypothetical release. Benchmark the existing Qwen3.6-27B community DSpark head now.

## The decisive continuation plan

### Stage 1: remove avoidable overhead

Implement only these changes:

* persistent device-side indices;
* no context synchronization for mask collection;
* no CPU sorting or index download;
* no per-draft-token re-upload;
* explicit GPU profiling of QK selection, top-k, sparse attention, and verification;
* generic optional top-k input rather than Vegas-specific sparse metadata where practical.

Repeat the exact current paired benchmarks before changing any mask algorithm. This tells you how much performance is being lost to plumbing.

### Stage 2: compare against the actual alternatives

For Qwen3.6-27B, compare at 32K, 64K, and roughly 112-128K:

* dense MTP;
* MTP with the best independent draft-cache format;
* MTP+Vegas;
* DSpark with the best memory-fitting quantization;
* ordinary decoding.

Record not only tokens per second, but also:

* total VRAM;
* maximum context that fits;
* draft time;
* target verification time;
* selection time;
* accepted tokens per cycle;
* output tokens per target forward pass.

For Qwen35 and Gemma, compare the same alternatives where compatible.

### Stage 3: test one mask generalization

Only after the device path is complete, test:

* current single layer;
* two layer groups;
* two-layer consensus;
* hybrid fixed anchors plus selected history.

Do not reopen broad gamma, ratio, head, or query sweeps.

### Evidence needed for an upstream proposal

I would set this bar:

* Auto mode is non-regressive within measurement error in at least **85-90% of eligible configurations**.
* The worst reliable regression after gating is below **2%**.
* Geometric-mean improvement over the **best existing compatible speculative baseline**, not ordinary decoding, is at least **5-8%**.
* At least three distinct model/drafter architectures show gains above 5%.
* Selection overhead is below about 5% of speculative-cycle time.
* Recommended stochastic sampling is tested in addition to greedy decoding.
* At least five prompts and 512 generated tokens per major configuration are used.
* One multi-sequence server experiment works without CPU synchronization.
* No model-name lookup table is required.

Your current deterministic paired tests are excellent engineering diagnostics, but the final performance results are mostly batch-one, greedy, short-output runs. The original Vegas work used model-recommended sampling and continuous batching, so broader evaluation is necessary before claiming generality.

### The stop condition

Stop Vegas after this sprint when either of these is true:

* Removing the host path and improving sparse attention still leaves meaningful wins only for Gemma 128K and Qwen35 q4/q4.
* Broader masks recover acceptance only by adding enough selection or KV traffic to erase the speedup.
* A memory-fitting DSpark checkpoint consistently beats MTP+Vegas across your Qwen workloads without materially reducing usable context.
* Upstream requires per-model tuning tables or several hundred lines of architecture-specific policy.

At that point, preserve `examples/vegas` as a strong research record, upstream the independent draft-cache improvement, and contribute to the generic sparse-attention infrastructure instead.

## Final evaluation

**The idea:** Strong and technically valid, but inherently conditional. Verification-guided sparse attention is a good long-context optimization, not a universal decoding strategy.

**The current implementation:** An excellent research prototype. It is compact, correctness-conscious, unusually well benchmarked, and has already found two large real wins. It is not systems-optimal because the oracle is rematerialized, masks cross the CPU, sparse attention bypasses the general kernel path, and serving is restricted to one sequence.

**The remaining algorithmic potential:** Moderate. Two-layer-group or consensus masks and hybrid anchors could improve generality. I do not expect a hidden selection trick to turn Qwen27 from +2.8% into +20%, because its architecture and verification cost impose a real ceiling.

**The remaining systems potential:** High enough to justify one more sprint. Device-resident selection and a generic sparse FlashAttention interface are the most important unfinished work.

**Upstream potential today:** Low as a complete "Vegas feature," because the policy is empirical and the implementation is CUDA/single-sequence-specific.

**Upstream potential after the pivot:** Good as a generic sparse-KV primitive, with Vegas as one producer and an automatic speculative consumer.

**DSpark obsolescence risk:** High for targets that receive a strong official sidecar and have sufficient memory; moderate for the broader local-GGUF ecosystem.

**Recommendation:** Continue, but only with the systems pivot and explicit go/no-go criteria. The very next change should eliminate the GPU-CPU-GPU mask path - not run another parameter sweep. Start a narrow upstream discussion around the generic sparse-index interface and coordination with the existing sparse-MMA PR, not around merging the current auto policy. Before submitting code, follow llama.cpp's `AGENTS.md` and contribution rules, disclose AI assistance, and personally review and own every submitted line.

[1]: https://arxiv.org/abs/2607.05147 "DSpark paper"
[2]: https://github.com/deepseek-ai/DeepSpec "DeepSpec repository"
[3]: https://huggingface.co/dbirks/Qwen3.6-27b-DSpark "Qwen3.6-27B DSpark checkpoint"
[4]: https://www.reuters.com/business/retail-consumer/alibaba-unveils-its-most-capable-ai-model-date-not-far-behind-moonshots-size-2026-08-03/ "Qwen3.8-Max report"
[5]: https://huggingface.co/Qwen/Qwen3.6-27B "Qwen3.6-27B model card"
