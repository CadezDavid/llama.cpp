# MTP and Vegas performance on RTX 3090

## Scope and protocol

This is a private CUDA-only experiment for batch-one long-context decoding. The four modes are dense baseline decoding, MTP drafting, Vegas self-speculative decoding, and MTP drafting with Vegas-guided sparse target verification.

Focused follow-up tuning, the architecture-class auto policy, and its separate
validation results are documented in
[`results/unified-strategy/README.md`](results/unified-strategy/README.md). That
work supersedes the original campaign's untuned conclusion for Qwen27 and
Qwen35 q4_0/q4_0; the screening tables below remain unchanged evidence.

Completed 256-token measurements from the original campaign were preserved. Every remaining model/context/cache/mode cell was screened once with 128 generated tokens. Clearly negative screens were not confirmed. The retained 256-token MTP versus MTP+Vegas cells use sequential stopping: two order-rotated repetitions first, followed by a third pair only when signs disagree or the effect is within about 5%. Existing n=3 results were preserved and not repeated. OOM configurations were not retried. Screening and confirmatory evidence are reported separately.

Prompt lengths are 16,384, 32,768, 65,536, and 128,000 tokens. Context capacities are 24,576, 40,960, 73,728, and 131,072. Decode throughput excludes model loading and prompt prefill. Runs use batch one, greedy sampling, flash attention, full CUDA offload, and matched target/draft KV formats. Vegas uses gamma 1 and a 3% mask with a 256-token floor. MTP and MTP+Vegas use gamma 3.

## One-pass screening (128 generated tokens)

Recorded screening measurements: 158 successful and 10 unavailable, representing 168 mode cells.
Values are tok/s from one run and must not be interpreted as stable small differences.

| Model | Context | KV | baseline | mtp | vegas | mtp-vegas | n(baseline) | n(mtp) | n(vegas) | n(mtp-vegas) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-31b | 128k | q8-turbo3 | 15.76 | 30.25 | 18.20 | 41.44 | 1 | 1 | 1 | 1 |
| gemma4-31b | 128k | q8-turbo4 | 14.69 | 12.96 | 12.35 | 18.00 | 1 | 1 | 1 | 1 |
| qwen36-27b | 128k | q4-q4 | 13.61 | 46.18 | 21.09 | 45.54 | 1 | 1 | 1 | 1 |
| qwen36-27b | 128k | q8-q4 | 13.56 | 45.89 | 20.95 | 42.62 | 1 | 1 | 1 | 1 |
| qwen36-27b | 128k | q8-q8 | 16.51 | 47.22 | 20.56 | 42.38 | 1 | 1 | 1 | 1 |
| qwen36-27b | 128k | q8-turbo3 | 13.36 | 42.99 | 19.76 | 41.36 | 1 | 1 | 1 | 1 |
| qwen36-27b | 128k | q8-turbo4 | 16.97 | 40.03 | 19.11 | 37.27 | 1 | 1 | 1 | 1 |
| qwen36-27b | 16k | q4-q4 | 31.92 | 73.06 | 30.53 | 66.28 | 1 | 1 | 1 | 1 |
| qwen36-27b | 16k | q8-q4 | 31.93 | 72.70 | 30.39 | 66.32 | 1 | 1 | 1 | 1 |
| qwen36-27b | 16k | q8-q8 | 33.80 | 73.36 | 30.49 | 68.25 | 1 | 1 | 1 | 1 |
| qwen36-27b | 16k | q8-turbo3 | 31.37 | 69.80 | 30.19 | 62.09 | 1 | 1 | 1 | 1 |
| qwen36-27b | 16k | q8-turbo4 | 33.70 | 52.77 | 29.35 | 49.97 | 1 | 1 | 1 | 1 |
| qwen36-27b | 32k | q4-q4 | 26.63 | 66.13 | 28.47 | 60.05 | 1 | 1 | 1 | 1 |
| qwen36-27b | 32k | q8-q4 | 26.63 | 65.80 | 28.42 | 60.09 | 1 | 1 | 1 | 1 |
| qwen36-27b | 32k | q8-q8 | 29.13 | 66.51 | 28.37 | 58.39 | 1 | 1 | 1 | 1 |
| qwen36-27b | 32k | q8-turbo3 | 26.09 | 64.15 | 27.87 | 57.13 | 1 | 1 | 1 | 1 |
| qwen36-27b | 32k | q8-turbo4 | 29.34 | 54.08 | 27.07 | 45.56 | 1 | 1 | 1 | 1 |
| qwen36-27b | 64k | q4-q4 | 20.20 | 59.53 | 25.43 | 54.56 | 1 | 1 | 1 | 1 |
| qwen36-27b | 64k | q8-q4 | 20.11 | 57.66 | 25.30 | 54.27 | 1 | 1 | 1 | 1 |
| qwen36-27b | 64k | q8-q8 | 23.12 | 58.49 | 25.22 | 54.07 | 1 | 1 | 1 | 1 |
| qwen36-27b | 64k | q8-turbo3 | 19.59 | 55.09 | 24.37 | 52.17 | 1 | 1 | 1 | 1 |
| qwen36-27b | 64k | q8-turbo4 | 23.42 | 44.10 | 23.52 | 38.08 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 128k | q4-q4 | 38.13 | 135.08 | 59.71 | 128.22 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 128k | q8-q4 | 38.01 | - | 58.33 | - | 1 | - | 1 | - |
| qwen36-35b-a3b | 128k | q8-turbo3 | 36.56 | - | 54.92 | - | 1 | - | 1 | - |
| qwen36-35b-a3b | 128k | q8-turbo4 | 48.87 | - | 54.38 | - | 1 | - | 1 | - |
| qwen36-35b-a3b | 16k | q4-q4 | 106.88 | 225.73 | 86.33 | 179.29 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 16k | q8-q4 | 108.57 | 223.53 | 86.26 | 178.44 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 16k | q8-q8 | 114.82 | 234.16 | 86.23 | 178.20 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 16k | q8-turbo3 | 104.88 | 227.40 | 85.65 | 178.33 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 16k | q8-turbo4 | 114.27 | 200.57 | 82.24 | 158.90 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 32k | q4-q4 | 83.85 | 197.88 | 80.53 | 175.80 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 32k | q8-q4 | 83.94 | 198.73 | 80.89 | 175.99 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 32k | q8-q8 | 95.41 | 204.81 | 81.09 | 171.16 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 32k | q8-turbo3 | 81.88 | 197.44 | 79.16 | 167.24 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 32k | q8-turbo4 | 94.67 | 158.11 | 76.63 | 130.09 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 64k | q4-q4 | 59.99 | 178.14 | 72.19 | 150.59 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 64k | q8-q4 | 60.02 | 170.26 | 71.88 | 153.06 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 64k | q8-q8 | 71.62 | 177.78 | 71.70 | 144.08 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 64k | q8-turbo3 | 57.08 | 164.35 | 68.95 | 148.15 | 1 | 1 | 1 | 1 |
| qwen36-35b-a3b | 64k | q8-turbo4 | 71.55 | 134.87 | 65.65 | 113.73 | 1 | 1 | 1 | 1 |

## Focused MTP versus MTP+Vegas confirmation (256 generated tokens)

Recorded measurements in the focused confirmatory slice: 22. Gemma q8/turbo4 has n=3 from the preserved confirmation run. Gemma q8/turbo3 stopped at n=2 after two clear positive effects. Both Qwen 27B cells proceeded to n=3 because their effects remained within 5%.
The throughput columns are medians. `Hash match` compares the sets of output hashes observed in the two modes; a `no` requires inspection rather than automatically implying incorrect verification.

| Model | Context | KV | MTP | MTP+Vegas | Hybrid/MTP | MTP accept | Hybrid accept | Hash match | n/mode |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| gemma4-31b | 128k | q8-turbo3 | 30.66 | 38.63 | 1.260 | 0.839 | 0.843 | yes | 2 |
| gemma4-31b | 128k | q8-turbo4 | 13.29 | 20.37 | 1.533 | 0.182 | 0.311 | yes | 3 |
| qwen36-27b | 128k | q4-q4 | 45.42 | 43.50 | 0.958 | 0.945 | 0.940 | yes | 3 |
| qwen36-27b | 128k | q8-turbo3 | 43.58 | 41.66 | 0.956 | 0.974 | 0.974 | yes | 3 |

## Preserved 256-token measurements

Recorded preserved measurements: 227.
These include cells completed before the screening protocol replaced the exhaustive full matrix. Counts are shown explicitly so partial cells are not mistaken for confirmed results.

| Model | Context | KV | baseline | mtp | vegas | mtp-vegas | n(baseline) | n(mtp) | n(vegas) | n(mtp-vegas) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-31b | 128k | q4-q4 | 17.74 | 46.31 | 20.15 | 46.60 | 3 | 3 | 3 | 3 |
| gemma4-31b | 128k | q8-q4 | 17.46 | 34.78 | 19.61 | 40.52 | 3 | 3 | 3 | 3 |
| gemma4-31b | 128k | q8-turbo3 | - | 30.66 | - | 38.63 | - | 2 | - | 2 |
| gemma4-31b | 128k | q8-turbo4 | 14.61 | 13.29 | - | 20.37 | 1 | 3 | - | 3 |
| gemma4-31b | 16k | q4-q4 | 32.03 | 65.93 | 29.29 | 59.97 | 3 | 3 | 3 | 3 |
| gemma4-31b | 16k | q8-q4 | 31.89 | 68.43 | 29.31 | 62.72 | 3 | 3 | 3 | 3 |
| gemma4-31b | 16k | q8-q8 | 31.83 | 78.94 | 28.77 | 74.49 | 3 | 3 | 3 | 3 |
| gemma4-31b | 16k | q8-turbo3 | 31.36 | 78.43 | 28.49 | 73.84 | 3 | 3 | 3 | 3 |
| gemma4-31b | 16k | q8-turbo4 | 30.90 | 79.92 | 21.67 | 77.16 | 3 | 3 | 3 | 3 |
| gemma4-31b | 32k | q4-q4 | 28.63 | 67.89 | 27.24 | 63.06 | 3 | 3 | 3 | 3 |
| gemma4-31b | 32k | q8-q4 | 28.40 | 74.22 | 27.54 | 68.55 | 3 | 3 | 3 | 3 |
| gemma4-31b | 32k | q8-q8 | 28.28 | 74.86 | 27.17 | 70.32 | 3 | 3 | 3 | 3 |
| gemma4-31b | 32k | q8-turbo3 | 27.34 | 62.03 | 26.28 | 58.71 | 3 | 3 | 3 | 3 |
| gemma4-31b | 32k | q8-turbo4 | 26.53 | 56.34 | 23.62 | 51.20 | 3 | 3 | 3 | 3 |
| gemma4-31b | 64k | q4-q4 | 23.68 | 56.39 | 24.53 | 54.76 | 3 | 3 | 3 | 3 |
| gemma4-31b | 64k | q8-q4 | 23.36 | 53.77 | 24.40 | 50.74 | 3 | 3 | 3 | 3 |
| gemma4-31b | 64k | q8-q8 | 23.15 | 47.91 | 24.28 | 44.81 | 3 | 3 | 3 | 3 |
| gemma4-31b | 64k | q8-turbo3 | 21.83 | 46.10 | 22.89 | 44.61 | 3 | 3 | 3 | 3 |
| gemma4-31b | 64k | q8-turbo4 | 20.77 | 32.96 | 20.01 | 35.88 | 3 | 3 | 3 | 3 |
| qwen36-27b | 128k | q4-q4 | - | 45.42 | - | 43.50 | - | 3 | - | 3 |
| qwen36-27b | 128k | q8-turbo3 | - | 43.58 | - | 41.66 | - | 3 | - | 3 |

## Unavailable configurations

| Model | Context | KV | Mode | Reason |
|---|---:|---|---|---|
| gemma4-31b | 128k | q8-q8 | baseline | run failed: mode=baseline repetition=0 exit=-6 |
| gemma4-31b | 128k | q8-q8 | mtp | run failed: mode=mtp repetition=0 exit=1 |
| gemma4-31b | 128k | q8-q8 | mtp-vegas | run failed: mode=mtp-vegas repetition=0 exit=1 |
| gemma4-31b | 128k | q8-q8 | vegas | run failed: mode=vegas repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-q4 | mtp | run failed: mode=mtp repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-q4 | mtp-vegas | run failed: mode=mtp-vegas repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-q8 | baseline | run failed: mode=baseline repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-q8 | mtp | run failed: mode=mtp repetition=0 exit=1 |
| qwen36-35b-a3b | 128k | q8-q8 | mtp-vegas | run failed: mode=mtp-vegas repetition=0 exit=1 |
| qwen36-35b-a3b | 128k | q8-q8 | vegas | run failed: mode=vegas repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-turbo3 | mtp | run failed: mode=mtp repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-turbo3 | mtp-vegas | run failed: mode=mtp-vegas repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-turbo4 | mtp | run failed: mode=mtp repetition=0 exit=-6 |
| qwen36-35b-a3b | 128k | q8-turbo4 | mtp-vegas | run failed: mode=mtp-vegas repetition=0 exit=-6 |

## Correctness interpretation

MTP+Vegas verifies draft tokens with the target model under Vegas sparse attention; it does not accept tokens from the draft model alone. Sparse Vegas execution is not bitwise equivalent to dense baseline execution, and baseline-versus-Vegas hashes commonly differ in the screening data. MTP and hybrid hashes match in every focused confirmation. Output hashes, acceptance counts, and failures are retained with every measurement.

## Conclusion

MTP and Vegas work together correctly in the tested implementation: every runnable focused confirmation produced matching MTP and hybrid output hashes, and no hybrid-only crash occurred. Performance is not generally positive. At Gemma 4 31B 128K, MTP+Vegas improved over MTP by about 53% with q8/turbo4 and 26% with q8/turbo3. At Qwen 3.6 27B 128K, it was about 4% slower with both q8/turbo3 and q4/q4. The broader one-pass screen was predominantly negative below 128K.

The honest result on this RTX 3090 is therefore conditional: Vegas can materially improve long-context MTP verification for Gemma with the fork's turbo caches, but it is not a universal batch-one decoding improvement and should not be enabled by default for the tested Qwen models. Several Qwen 35B 128K MTP configurations are unavailable because they exceed 24 GB VRAM.
