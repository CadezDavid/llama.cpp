# Qwen3.6 27B MTP+Vegas recovery experiment

Date: 2026-08-03

## Question

Determine why MTP+Vegas was about 4% slower than dense MTP on Qwen3.6
27B, and test minimal changes that could make the combination beneficial.

## Diagnosis

The original 62K q4/q4 result used gamma 3, a 3% mask, and the final
target layer. Sparse drafting saved 166 ms, but acceptance fell from 75.9%
to 68.5%. Three additional dense target-verification cycles and the target
mask-selection work outweighed the draft saving.

The implementation also derives the selection by recomputing a QK matrix
after FlashAttention. The Vegas paper collects logits inside FlashAttention.
The separate operation is proportionally expensive for the dense 27B target.

## Experimental changes

- Added an opt-in target selection layer to the private executable. The
  existing final-layer behavior remains the default.
- Added an opt-in selection refresh interval. Interval 1 preserves the
  previous behavior.
- Added pause/resume state so a retained sparse mask is used only for
  single-token drafting. Qwen's batched MTP state update remains dense.
- Added the same controls and result fields to `benchmark.py`.

No CUDA kernel was changed. No automatic model or context policy was added.

## Prompts and setup

- Model: Qwen3.6 27B Q4_K_XL with its embedded MTP block.
- GPU: NVIDIA RTX 3090.
- Batch one, full CUDA offload, FlashAttention, greedy sampling, seed 1234.
- Short real prompt: OpenCode session
  `ses_2452593b5ffe6yJELmXEV6x9oM` through database message 238,
  reconstructed as 62,101 model tokens.
- Long real prompt: the same session through user message 412, including
  user text, assistant text, and assistant reasoning, reconstructed as
  112,025 model tokens.
- Results exclude model loading and prompt prefill.

The reconstructed prompts have the same source boundaries and essentially
the same byte sizes as the earlier documented prompts, but different hashes.
These results are therefore a new prompt campaign and are not pooled with the
earlier measurements.

## Screening results at 62K, q4/q4, 128 generated tokens

| Policy | tok/s | Cycles | Accept | Delta vs matched dense |
|---|---:|---:|---:|---:|
| Dense MTP, gamma 3 | 43.58 | 45 | 61.7% | baseline |
| Vegas, gamma 3, 3%, final layer | 42.27 | 47 | 57.6% | -3.0% |
| Vegas, gamma 3, 5%, final layer | 41.99 | 47 | 57.6% | -3.7% |
| Vegas, gamma 3, 7%, final layer | 43.77 | 45 | 61.7% | +0.4% |
| Dense MTP, gamma 4 | 46.39 | 37 | 62.1% | baseline |
| Vegas, gamma 4, 7%, final layer | 45.57 | 39 | 57.5% | -1.8% |
| Vegas, gamma 4, 7%, layer 47 | 46.68 | 38 | 59.7% | +0.6% |
| Vegas, gamma 4, 7%, layer 31 | 45.58 | 39 | 57.5% | -1.7% |
| Vegas, gamma 4, 7%, layer 15 | 46.71 | 38 | 59.7% | +0.7% |
| Vegas, gamma 4, 7%, layer 15, refresh 2 | 46.91 | 38 | 59.7% | +1.1% |
| Vegas, gamma 4, 7%, layer 15, refresh 4 | 45.86 | 39 | 57.5% | -1.1% |
| Dense MTP, gamma 5 | 41.91 | 36 | 52.0% | baseline |
| Vegas, gamma 5, 7%, final layer | 42.59 | 37 | 50.0% | +1.6% |

Increasing the gamma-4 layer-15 or layer-47 mask from 7% to 10% did not
change acceptance or cycles and was slightly slower. Ratio tuning has broad
plateaus rather than a smooth response.

## Confirmed 62K result

Two fresh order-rotated 128-token pairs compared gamma-4 dense MTP with the
gamma-4, 7%, layer-15, refresh-2 hybrid.

| Mode | Mean tok/s | Range | Cycles | Output hash |
|---|---:|---:|---:|:---:|
| Dense MTP | 46.420 | 46.395-46.445 | 37 | match |
| MTP+Vegas | 46.785 | 46.766-46.803 | 38 | match |

The confirmed mean gain is 0.79%. It is repeatable on this prompt but small.

## Long-context screening and confirmation

The 112,025-token q4/q4 prompt showed that the 62K ratio does not transfer.

| KV | Output | Policy | tok/s | Cycles | Accept | Delta |
|---|---:|---|---:|---:|---:|---:|
| q4/q4 | 128 | Dense gamma 4 | 43.26 | 33 | 73.4% | baseline |
| q4/q4 | 128 | 7%, layer 15, refresh 2 | 40.90 | 38 | 61.4% | -5.5% |
| q4/q4 | 128 | 10%, layer 15, refresh 2 | 44.12 | 35 | 69.2% | +2.0% |
| q4/q4 | 256 | Dense gamma 4 | 46.00 | 62 | 78.5% | baseline |
| q4/q4 | 256 | 10%, layer 15, refresh 2 | 46.64 | 66 | 72.1% | +1.4% |
| q8/turbo3 | 128 | Dense gamma 4 | 38.37 | 35 | 66.7% | baseline |
| q8/turbo3 | 128 | 10%, layer 15, refresh 2 | 39.04 | 37 | 61.6% | +1.8% |

The 256-token q4/q4 pair is the long-context confirmation. The q8/turbo3
pair is a one-pass transfer screen and should not be treated as confirmed.

The long q4/q4 dense and hybrid hashes differ. Sparse proposals changed the
verification batch shapes, and the greedy target path diverged numerically.
The 62K pairs and the long q8/turbo3 pair matched hashes.

## Correctness and failed experiment

- A 2,048-token ratio-1 test with gamma 4, layer 15, and refresh 2 produced
  identical dense and hybrid hashes, 18 cycles, 71 proposals, 45 accepted
  proposals, and 12 rejections.
- The first refresh-reuse implementation incorrectly left sparse mode active
  during Qwen's batched MTP state update. It failed the existing single-query
  assertion in `build_attn_mha`. It produced no result file.
- The corrected implementation pauses Vegas for the batched update and
  resumes the retained mask only for sequential draft decoding.
- All successful commands, timings, hashes, acceptance counts, and telemetry
  are retained in the JSONL files in this directory.

## Conclusion

Qwen3.6 27B can benefit from MTP+Vegas, but the measured gain is modest and
policy-sensitive. The previous roughly -4% result was primarily an acceptance
and extra-cycle problem, compounded by an unfused target selection operation.

For these real prompts, the best observed policy is gamma 4, layer 15, refresh
interval 2, with a 7% mask around 62K and a 10% mask around 112K. It improves
q4/q4 by 0.8% at 62K and 1.4% at 112K in confirmation measurements. A
q8/turbo3 screen improved by 1.8% at 112K.

This is not strong enough to enable by default. The next high-value work is a
fused attention-logit collector and a robust mask-quality policy, not a larger
manual sweep of layers and percentages.
