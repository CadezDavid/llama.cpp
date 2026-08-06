# Qwen recurrent-state inverse experiment

Date: 2026-08-06

GPU: NVIDIA RTX 3090 24 GB

Model: Qwen3.6-27B Q4_K_XL

Context allocation: 1024 tokens

Prompt: native model chat template, 55 tokens, natural greenhouse-design conversation

Continuation policy: greedy

## Question

Can CascadeSpec avoid storing full recurrent rollback states by recording the
per-token `k`, `delta`, decay, and evicted convolution slice, then algebraically
reversing the Qwen gated-delta update?

## Method

The diagnostic saves the complete context before a 16-token generated block.
For rollback depths 1, 4, 8, and 16 it restores that state, processes the same
prefix in exact and inverse contexts, processes only the rejected tail in the
inverse context, and reverses that tail. It compares:

- the convolution and gated-delta recurrent states;
- the next 16 full logit distributions;
- greedy token choices;
- hashes of the effective complete token streams;
- CUDA kernel time and end-to-end inverse-call time.

The compact 16-token log used 56,770,560 bytes (54.14 MiB), or about 3.38 MiB
per token. The saved complete context at this short context length used
160,500,350 bytes.

## Results

| Reversed tokens | Minimum decay | Worst theoretical decimal amplification | S max error | Output tokens | Inverse wall time | CUDA kernel time |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 3.56e-11 | 10.448 orders | 1.61e2 | 16/16 matched | 1.316 ms | 0.838 ms |
| 4 | 3.85e-11 | 20.094 orders | 1.26e12 | 16/16 matched | 1.394 ms | 0.908 ms |
| 8 | 9.97e-16 | 48.065 orders | infinity | 0/16 matched | 1.571 ms | 1.083 ms |
| 16 | 3.02e-16 | 80.813 orders | infinity | 0/16 matched | 1.872 ms | 1.393 ms |

The convolution state reconstructed exactly at every tested depth. The failure
is entirely in the large gated-delta matrix state.

At depths 1 and 4, the selected tokens and complete output hashes still matched
despite large state error. At depths 8 and 16, the reconstructed matrix
overflowed to non-finite values and generation diverged immediately.

## Interpretation

The algebraic inverse is correct but numerically ill-conditioned in float32.
The forward update multiplies the old state by decays as small as roughly
1e-16 before adding the new rank-one update. At that scale, float32 no longer
retains enough information about the old state. Subtracting the rank-one update
and dividing by the tiny decay cannot recreate information that was rounded
away during the forward pass.

Therefore this compact inverse cannot safely replace recurrent snapshots for
CascadeSpec. Higher-precision arithmetic in only the inverse kernel cannot fix
the lost bits. A reliable undo record would need to retain a much larger
residual or state correction, undermining the intended memory saving.

## Commands

```text
cmake --build build-vegas -j6 --target test-recurrent-undo-model test-gated-delta-undo test-backend-ops
build-vegas/bin/test-gated-delta-undo
build-vegas/bin/test-backend-ops test -o GATED_DELTA_NET -b CUDA0
build-vegas/bin/test-recurrent-undo-model \
  -m /home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  -ngl 999 -c 1024
```

The synthetic inverse test passed at all four depths; all 36 CUDA
`GATED_DELTA_NET` backend cases also passed. The real-model diagnostic exits
non-zero by design when tokens/hashes differ or non-finite values appear.
