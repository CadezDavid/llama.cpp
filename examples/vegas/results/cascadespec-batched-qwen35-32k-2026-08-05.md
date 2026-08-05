# Batched hierarchical verification: Qwen35 at 32K

Date: 2026-08-05

## Shared configuration

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Prompt: native-chat OpenCode conversation fixture, 32,716 tokens
- Generation: 128 tokens, temperature 0, seed 1234, EOS ignored
- Context allocation: 36,864 tokens
- Target and MTP KV: q8_0/q4_0
- GPU layers: all
- MTP gamma: 3
- Hierarchical limit: target 8, hard cap 10, 3 rounds, 2 corrections
- Sparse selection: layer 39, 16 anchors, 256-token minimum, refresh every cycle

All successful hierarchical runs produced output hash `18ed0fdccc6de83f` and reported zero rollback, position, and snapshot failures.

## Results

| Mode | Retained attention | tok/s | Total ms | Dense cycles | Dense acceptance | MTP ms | Sparse target ms | Dense verify ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla MTP | n/a | 174.755 | 732.453 | 38 | 80.2% MTP acceptance | 176.517 | n/a | 529.163 |
| MTP+Vegas | 20% | 161.004 | 795.013 | 39 | 77.2% MTP acceptance | 132.969 | n/a | 633.428 |
| Batched hierarchy | 20% | 106.723 | 1199.371 | 15 | 88.2% | 130.810 | 579.741 | 388.716 |
| Batched hierarchy | 50% | 108.626 | 1178.360 | 15 | 94.1% | 127.547 | 581.668 | 380.057 |
| Batched hierarchy | 100% | 109.838 | 1165.353 | 14 | 100.0% | 131.478 | 588.831 | 360.287 |

The old sequential sparse-target implementation reached 61.849 tok/s. The first batched implementation reached 88.161 tok/s. Separating the target rollback capacity from the per-round MTP capacity raised the final 20% result to 106.723 tok/s.

The hierarchy reduced dense cycles from 38-39 to 14-15, but its 580-589 ms sparse-target stage outweighed the saved dense-verification work. Retaining more attention improved acceptance, but barely changed sparse-target time, so this gather path is dominated by fixed model and gather work rather than the retained KV percentage.

## Target-20 capacity result

Target 20 with hard cap 20 and five inner rounds is implemented, but it did not fit on the 24 GB RTX 3090 at a 32K prompt. The final capacity-separated implementation still exhausted CUDA memory while initializing prompt/MTP state with q8_0/q4_0 KV at both 36,864 and 33,024 context allocations. q4_0/q4_0 KV and disabled CUDA graphs also did not fit.

The limiting allocation is the target model's 20 recurrent rollback states, not the attention KV cache or the MTP per-round capacity. Partial CPU offload is not a usable workaround because Vegas currently requires all model layers on one CUDA device. Target 8 with cap 10 is therefore the largest tested fully GPU-resident configuration for this model and GPU.
