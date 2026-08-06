# Qwen27 CascadeSpec cadence at 128K

One paired execution compared plain MTP gamma 4 with the four-round CascadeSpec schedule on an
identical, naturally formatted OpenCode conversation.

## Configuration

- Model: Qwen3.6-27B-MTP Q4_K_XL.
- Prompt: 131,038 tokens in a 131,584-token context.
- Generation: 128 tokens, temperature 0, seed 1234.
- Target and draft KV caches: q8_0/q4_0.
- CascadeSpec: MTP gamma 4, three 10%-attention sparse rounds, then one direct dense round.
- Provisional cap: 19 tokens; recurrent checkpoint stride: 4.

The native-chat prompt was assembled from four coherent OpenCode sessions. Its requested 131,072
token budget had a gap of only 34 tokens.

## Result

| Mode | tok/s | Total generation time | Target passes | Output hash |
|---|---:|---:|---:|---|
| Plain MTP | 37.27 | 3434.21 ms | 33 dense | `b26bd385cfed9913` |
| CascadeSpec | 26.32 | 4862.56 ms | 41 sparse + 14 dense | `3b22ee6063e9623d` |

CascadeSpec was 29.4% slower.

At 128K, a sparse pass averaged 61.46 ms versus 76.52 ms for a plain gamma-4 dense pass, a 19.7%
saving per pass. That saving was insufficient because CascadeSpec performed 55 target passes rather
than 33. Its final larger dense passes averaged 89.03 ms.

The accumulated branch was also not stable enough: sparse agreement was 74.5%, final dense
acceptance was 66.1%, and the system committed 9.07 tokens per dense cycle. Sparse plus dense target
time alone was 3766.40 ms, already 49.2% above plain MTP's complete dense-verification time of
2525.19 ms.

All state-integrity counters were zero. The run completed 55 recurrent prefix restores, replayed 53
intermediate updates, and exercised replay depths zero through three without a snapshot, rollback,
position, or recurrent-restore failure. Output hashes are retained as an advisory diagnostic because
different dense batch lengths can change floating-point rounding in greedy generation.

Files:

- `manifest.json`: configuration and OpenCode fixture provenance.
- `results.jsonl`: the two complete raw executions and commands.
- `summary.json`: compact comparison.
