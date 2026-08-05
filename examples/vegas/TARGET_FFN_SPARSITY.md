# Target FFN sparsity experiment

Date: 2026-08-04

## Conclusion

Prox-style target FFN sparsity is not a viable performance optimization for the
current RTX 3090, long-context GGUF setup.

The negative result is specific to the models and storage formats tested here.
It is not evidence that activation sparsity is ineffective for FP16 models or
for models stored in a sparse-friendly weight layout.

There are two independent blockers:

1. The quality-preserving masks are fine-grained, but Q4_K, Q5_K, and Q6_K
   store the down-projection input dimension in packed 256-value blocks.
   Fine-grained masks therefore leave almost every packed block live and do not
   remove the HBM traffic that dominates batch-one GEMV.
2. The paper's separate INT4 up/gate proxy would require about 5.31 GiB for
   Qwen3.6 27B. At a 120K q8/turbo4 context, the process used 19,864 MiB and the
   GPU had only 3,479 MiB free, including about 598 MiB used by unrelated local
   servers. The faithful proxy does not fit, and it would not solve the packed
   exact down projection by itself.

A custom sparse CUDA kernel was deliberately not added after these gates
failed. Such a kernel could skip arithmetic, but it could not avoid enough
weight bytes to produce a meaningful speedup. Adding it would leave a large,
format-specific code path with a non-positive expected benefit.

## What was implemented

`llama-ffn-sparse-eval` runs dense and experimental target contexts over the
same real token stream. Prompt prefill stays dense. During the evaluation
window it teacher-forces identical tokens and reports:

- dense and sparse perplexity;
- token-wise KL divergence and probability L1 distance;
- dense/sparse top-1 agreement;
- teacher-token top-1 rates.

The target graph supports three diagnostic modes:

- exact post-activation oracle selection;
- a Prox-shaped sparse-input proxy whose values are used only to rank exact
  intermediate channels;
- a more aggressive direct-proxy experiment that uses selected proxy values,
  avoiding the second exact up/gate computation in the hypothetical kernel.

Selection uses a fixed per-token top-k budget. This is intentionally a quality
and feasibility screen, not a claim to reproduce the paper's calibrated
layer-wise thresholds. Exact top-k is a favorable oracle for the structured
mask tests: if it cannot preserve the current activation, a stale or
approximate selector at the same granularity cannot repair the layout problem.

## Performance ceiling

An Nsight Systems trace of Qwen3.6 27B batch-one generation on the RTX 3090
measured 38.28 tok/s at short context. In the captured CUDA graph:

| Kernel group | Share of GPU kernel time | Relevant shape |
| --- | ---: | --- |
| fused FFN up/gate MMVQ | 31.0% | 17,408 output rows |
| all 5,120-row MMVQ operations | 28.1% | includes FFN down and projections |
| full-attention kernel | 2.6% | short-context trace |

The FFN is therefore worth optimizing. The failed result is not caused by an
insignificant target.

Qwen's 64 target FFNs occupy approximately:

| Weights | Existing GGUF storage |
| --- | ---: |
| up | 3.040 GiB |
| gate | 3.040 GiB |
| down | 3.639 GiB |
| total | 9.720 GiB |

The model contains 152 Q4_K, 28 Q6_K, and 15 Q5_K FFN matrices. Up and gate
are predominantly Q4_K; down has many Q6_K tensors.

## Quality screens

All rows use a 1,024-token dense prefix and 128 teacher-forced evaluation
tokens. Lower KL and higher top-1 agreement are better. Perplexity alone is not
sufficient because compensating logit changes can reduce it on a short window.

### Qwen3.6 27B

| Variant | Corpus | PPL ratio | Mean KL | Top-1 agreement | Result |
| --- | --- | ---: | ---: | ---: | --- |
| exact oracle, 40%, individual | OpenCode | 1.003 | 0.066 | 91.4% | quality upper bound only |
| Prox selection, s1=50%, s2=50%, individual | OpenCode | 1.020 | 0.144 | 86.7% | ranking loss too high |
| direct proxy, s1=30%, s2=30%, individual | OpenCode | 1.003 | 0.018 | 95.3% | promising quality, unusable layout |
| direct proxy, s1=30%, s2=30%, individual | Prox paper text | 0.980 | 0.013 | 93.8% | confirms coherent-text result |
| direct proxy, s1=30%, s2=30%, pair blocks | OpenCode | 1.092 | 0.066 | 89.8% | fails quality gate |
| direct proxy, s1=30%, s2=30%, 32 blocks | OpenCode | 1.664 | 0.680 | 73.4% | fails decisively |
| exact oracle, 40%, 256 blocks | OpenCode | 1.304 | 0.287 | 83.6% | layout-native upper bound fails |

An untemplated Codex transcript was also retained as a stress test. Its dense
perplexity was 438, so it is not a useful language-model perplexity sample, but
the 75.8% top-1 agreement shows that the apparently safe individual-channel
point is not robust to distribution shift.

### Gemma 4 31B

Gemma was fragile even under the favorable exact oracle:

| Variant | PPL ratio | Mean KL | Top-1 agreement |
| --- | ---: | ---: | ---: |
| exact oracle, 20%, individual | 1.015 | 0.123 | 82.0% |
| exact oracle, 40%, individual | 1.370 | 0.218 | 78.9% |
| direct proxy, s1=20%, s2=20%, individual | 1.036 | 0.118 | 93.8% |

Twenty percent sparsity is already outside a conservative negligible-impact
criterion, and it is too low to overcome selection and proxy overhead in the
current packed layout.

## Why individual masks do not become fast Q4_K kernels

Q4_K packs 256 scalar weights per quantization block. Its values are also
nibble-packed, so a byte commonly contains two scalar values. For an
independent retained fraction `r`, the probability that a packed byte is needed
is approximately:

```text
P(byte used) = 1 - (1 - r)^2
```

At the best quality point, `r = 0.70`, so about 91% of value bytes remain live.
Every 256-value block is effectively guaranteed to be live, including its
scales and metadata. The access pattern also becomes less coalesced.

For exact Prox using the existing Q4 weights as the proxy, the idealized dense
matrix-equivalent cost at s1=s2=30% would already be:

```text
proxy up/gate:       2 * 0.70
exact up/gate:       2 * 0.70
exact down:          1 * 0.70
total:               3.50, versus 3.00 dense
```

The paper wins because its proxy is much cheaper than the exact weights. In
this GGUF, the existing up/gate tensors are already predominantly Q4_K, so
there is no such precision gap. Packed-byte survival makes the practical cost
worse than this favorable arithmetic model.

The direct-proxy variant removes exact up/gate recomputation and has an
arithmetic ceiling of 30% FFN work reduction at 30% sparsity. With 91% of packed
bytes still live, its bandwidth ceiling is only about 9% of FFN traffic, before
top-k, masking, irregular loads, and graph work. Since FFN is roughly half of
short-context kernel time and a smaller fraction at long context, the absolute
end-to-end ceiling is below 5% and will shrink further after overhead.

## Alternatives considered

- Quantization-aligned dynamic blocks: rejected by the oracle quality tests.
- Repack the down projection in a channel-major sparse layout: requires
  replacing or duplicating 3.64 GiB of weights and new prefill/decode kernels.
  Duplicating it does not fit at the target context; replacing it is a new
  model format and subsystem, not a minimal experiment.
- Add a lower-bit proxy: a 2-bit up/gate proxy is still about 2.66 GiB, has no
  demonstrated ranking quality, and leaves the exact down-layout problem.
- Reuse previous-token masks: it removes the proxy but cannot discover newly
  important channels without refreshes, and individual down masks still do not
  save packed weight traffic.
- Qwen3.6 35B-A3B: its target FFN is already expert-sparse. The dense-SwiGLU
  Prox path tested here does not directly apply to its MoE expert dispatch.

## Reproduction

Build:

```sh
cmake --build build-vegas --target llama-ffn-sparse-eval -j 12
```

Representative quality command:

```sh
build-vegas/bin/llama-ffn-sparse-eval \
  -m /home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -f /tmp/opencode-spomin-user-412.txt \
  --ctx-size 2048 --batch-size 1024 --ubatch-size 512 \
  -ngl 99 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --prefix-tokens 1024 --eval-tokens 128 \
  --ffn-sparsity 0.3 --ffn-block-size 1 \
  --ffn-proxy-input-sparsity 0.3 --ffn-proxy-block-size 1 \
  --ffn-proxy-use-values
```

Raw outputs are in
`examples/vegas/results/target-ffn-sparsity-2026-08-04/`.

## Stop decision

The requested stop condition is met by a setup-specific negative result:

- FFN time is large enough to matter;
- fine-grained activation sparsity can sometimes preserve Qwen quality;
- the minimum layout-compatible structure destroys that quality;
- the faithful INT4 proxy does not fit at the desired long context;
- reusing existing Q4 weights removes the paper's cheap-proxy advantage;
- a fine-grained packed-weight kernel has a below-5% end-to-end ceiling before
  overhead and therefore cannot provide the requested meaningful speedup.

Further work should resume only if one constraint changes: a model with native
FFN sparsity, a sparse-friendly/repacked GGUF layout, substantially more VRAM,
or a trained compact predictor. None is a minimal extension of the current
Vegas implementation.
