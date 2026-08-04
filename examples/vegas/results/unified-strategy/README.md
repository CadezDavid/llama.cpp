# Vegas adaptive strategy experiment

Date: 2026-08-04

This directory records focused follow-up experiments for a simple MTP/Vegas
policy across Gemma 4 31B, Qwen3.6 27B, and Qwen3.6 35B. The primary cache of
interest is q8_0/turbo4. Results here are private-fork experiments, not upstream
performance claims.

## Result

The useful unifying rule is a break-even rule, not one universal sparse ratio:

```
Vegas wins when sparse draft savings exceed mask selection and any extra target
verification cycles caused by lower draft acceptance.
```

`--vegas-mode mtp-auto` applies an architecture-class policy without checking
model names:

| MTP arrangement | Condition | Effective mode | Gamma | Sparse settings |
| --- | --- | --- | ---: | --- |
| Separate assistant | prompt below 96K | dense MTP | 1 | none |
| Separate assistant | prompt at least 96K | MTP+Vegas | 1 | 3%, final layer, refresh 1 |
| Embedded MoE MTP | q4_0/q4_0 and prompt at least 48K | MTP+Vegas | 5 | 3%, final layer, refresh 1 |
| Embedded MoE MTP | otherwise | dense MTP | 2 | q8_0/turbo4 target uses q4_0/q4_0 draft cache |
| Embedded dense MTP | q4_0/q4_0 below 48K, or another cache below 96K | dense MTP | 4 | none |
| Embedded dense MTP | q4_0/q4_0 at least 48K, or another cache at least 96K | MTP+Vegas | 4 | ratio below, quarter-depth layer, refresh 2 |

For embedded dense MTP, the ratio is:

```
clamp(0.04 + prompt_tokens / 2,000,000, 0.05, 0.10)
```

This gives 7.1% at 62K and 9.6% at 112K. It preserves acceptance as the
context grows while keeping most of the attention saving.

## Primary measurements

All rows use real Codex or OpenCode conversation text, batch one, greedy
sampling, 128 generated tokens, and an RTX 3090. Throughput excludes prompt
prefill and includes draft, verification, sampling, and rollback.

| Model and cache | Prompt | Dense MTP | MTP+Vegas / auto | Difference | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma 4 31B q8_0/turbo4 | 128K Codex | 14.619 tok/s | 23.633 tok/s | +61.7% | auto gamma 1, 3% Vegas; same output hash |
| Qwen3.6 27B q8_0/turbo4 | 112K OpenCode | 29.455 tok/s | 30.271 tok/s | +2.8% | auto 9.6%, layer 15, refresh 2 |
| Qwen3.6 27B q8_0/turbo4 | 62K OpenCode | 33.720 tok/s | 30.938 tok/s | -8.3% | screen caused non-q4_0 threshold to move to 96K |
| Qwen3.6 35B q8_0/turbo4 | 62K OpenCode | 127.565 tok/s | auto 132.238 tok/s; configured 134.243, 134.010 | +3.7% auto; +5.2%, +5.1% configured | dense gamma 2, q4_0/q4_0 draft; Vegas disabled |
| Qwen3.6 35B q4_0/q4_0 | 113.6K OpenCode | 123.291, 122.992 tok/s | 149.674, 149.962 tok/s | +21.4%, +21.9% | preserved paired n=2 result |

The 112K Qwen27 auto run initially regressed because the MTP helper was created
with the command-line placeholder gamma 8 before auto mode selected gamma 4.
Two runs reproduced 844 ms sparse draft time. Resolving the policy before MTP
initialization reduced draft time to 473 ms and restored the positive result.
The corrected auto output hash and cycle count match the manually tuned run.

The Gemma gamma 1 pair is in
`gemma-128k-q8-turbo4-g1-r03.jsonl` and
`gemma-128k-q8-turbo4-g1-r03-vegas.jsonl`; the final auto validation is in
`gemma-128k-q8-turbo4-auto-reserve-fixed.jsonl`. The Qwen27 corrected auto result is
in `qwen27-112k-q8-turbo4-auto-resolved-before-init.jsonl`. The Qwen35 q4_0
paired source is `../qwen35-single-selection/g5-r03-paired-n2.jsonl`.
The Qwen35 mixed-cache pair and confirmation are
`qwen35-62k-target-q8-turbo4-draft-q4-q4-g2-r03-pair.jsonl` and
`qwen35-62k-target-q8-turbo4-draft-q4-q4-dense-g2-repeat.jsonl`; final auto
validation is `qwen35-62k-q8-turbo4-auto-q4-draft.jsonl`.

## Why the classes differ

Gemma uses a separate assistant. At 128K, dense gamma 1 accepted only 22.3% of
draft tokens, while sparse drafting accepted 82.6%. Vegas therefore improves
both assistant cost and agreement with the target. Gamma 1 avoids paying for a
wide verification batch when the dense assistant often diverges immediately.
On the same prompt, gamma 1 auto Vegas is 7.1% faster than the earlier gamma 3
Vegas run (23.633 versus 22.071 tok/s).

Qwen27 uses embedded dense MTP layers. The target-layer selection work used for
Vegas should be collected once, at a layer early enough that later discarded
masks are never computed. Layer 15 in the 64-layer model is the measured
quarter-depth point. At 112K, Vegas saves about 306 ms of draft work while
preserving the same 46 verification cycles. The gain is real but only about
3%, so ratio and initialization mistakes easily erase it.

At 62K q8_0/turbo4, the same policy shape saved 141 ms of draft work but added
five verification cycles, increasing verification by 475 ms and producing an
8.3% regression. Non-q4_0 caches therefore enable only from 96K. The 48K
threshold is retained for q4_0/q4_0, where the existing 62K paired screen was
small but positive (+0.8%).

Qwen35 is MoE. Its active-expert target and embedded MTP path make dense
q8_0/turbo4 drafting cheap enough that selection overhead and acceptance loss
dominate at 62K. Gamma 2 dense is substantially better than gamma 3 or 5 on the
tested q8_0/turbo4 prompt. With q4_0/q4_0 at 113.6K, dense attention is more
expensive: Vegas cuts draft time from about 403 ms to 162 ms and retains enough
acceptance to produce the confirmed 21.7% mean gain. The policy therefore uses
dense MTP for Qwen35 q8_0/turbo4 and enables Vegas only for the proven long
q4_0/q4_0 region. For the q8_0/turbo4 target, changing only the MTP draft cache
to q4_0/q4_0 raises acceptance from 72.8% to 81.3%, reduces cycles from 52 to
49, and improves 127.565 tok/s to 134.243 and 134.010 tok/s in two explicit
runs. Final auto mode reaches 132.238 tok/s, within the observed timing spread.
Adding Vegas to that gamma 2 mixed-cache setup drops to 115.999 tok/s, so the
optimized fallback remains dense.

## Rejected ideas

Dynamic gamma based only on accepted fractions oscillated and produced 90.1
tok/s on Qwen35 q8_0/turbo4, versus 127.8 tok/s for fixed gamma 2. A bounded
online throughput search found gamma 2 but still produced only 88.1 tok/s over
128 output tokens because exploration was not amortized and changed the
deterministic numerical trajectory. Both implementations were removed.

Selecting one KV head and one query reduced Qwen35 selection overhead, but the
best result remained 0.2% slower than dense gamma 3 and gamma 2 Vegas remained
11.8% slower than dense. The extra API and graph state were removed.

Refresh 2 helps Qwen27 by halving target mask collection without changing the
accepted trajectory. It hurt Qwen35 in the tested q8_0/turbo4 case, so the MoE
and separate-assistant policies retain refresh 1.

Sparse draft graph construction originally assumed every scheduler reserve
probe had a real single-token position beyond the selected prefix. Gemma auto
mode exposed multi-token and pre-prefix probe shapes. Sparse draft is now used
only when its shape and position invariants hold; reserve probes conservatively
fall through to dense attention. A targeted 8K reproduction and the final 128K
auto run both pass.

## Memory boundaries

Qwen35 MTP at 112K OOMed during prompt processing with q8_0/turbo4. A second
112K OpenCode prompt also OOMed with q4_0/q4_0 under the current narrow VRAM
margin, although the preserved 113.6K q4_0/q4_0 fixture completed previously.
Neither failed configuration was retried.

## Scope and confidence

The large Gemma and Qwen35 q4_0 results are confirmed by paired or repeated
measurements. The Qwen27 q8_0/turbo4 result is small but reproduced by the
manual and corrected auto configurations with identical output trajectories.
The thresholds deliberately fall back to dense MTP in unproven or clearly
negative regions. They are empirical RTX 3090 defaults, not portable universal
constants; other GPUs should recalibrate the break-even thresholds.

A final q4_0/q4_0 ratio=1 smoke produced identical output hash, 22 cycles, 63
drafted tokens, 41 accepted tokens, and 14 rejections for dense MTP and
MTP+Vegas. `test-batch-alloc` and the final build also pass. The raw smoke result
is `ratio1-reserve-correctness-smoke.jsonl`.
