# Target FFN sparsity raw results

Archive status: the evaluator that produced these files has been removed from
active code. Its final source is preserved in commit
`f1e07e9651c53ca0dd43307be62263a6d999a71b`.

The files in this directory are one-line `FFN_SPARSE_EVAL` records produced by
`llama-ffn-sparse-eval`. They are screening runs, not repeated throughput
benchmarks.

The evaluator uses two contexts sharing one model. Both contexts prefill the
same prefix densely. Sparsity is enabled only for the experimental context, and
both contexts then consume the same teacher-forced tokens.

Filename conventions:

- `s40-b1`: 40% exact post-activation oracle sparsity, block size 1.
- `prox-s1-50-s2-50`: sparse-input proxy used for selection, followed by exact
  selected values.
- `proxy-values`: selected proxy values are used directly.
- `b2`, `b32`, `b256`: channel selection granularity.
- `paper`, `codex`: alternate evaluation corpora.

`qwen27-proxy-values-s1-30-s2-30-prompt2.txt` duplicated the first OpenCode
result because both source files share the same first 1,152 tokens. It is kept
to preserve the completed measurement.

The interpretation and profiler evidence are in
`examples/vegas/TARGET_FFN_SPARSITY.md`.
