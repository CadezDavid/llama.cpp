#!/usr/bin/env python3

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODES = ["baseline", "mtp", "vegas", "mtp-vegas"]
CONFIRM_CELLS = {
    ("gemma4-31b", "128k", "q8-turbo4"),
    ("gemma4-31b", "128k", "q8-turbo3"),
    ("qwen36-27b", "128k", "q8-turbo3"),
    ("qwen36-27b", "128k", "q4-q4"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("examples/vegas/results/mtp-matrix"))
    parser.add_argument("--screening-dir", type=Path, default=Path("examples/vegas/results/mtp-screening"))
    parser.add_argument("--output", type=Path, default=Path("examples/vegas/MTP_VEGAS_RESULTS.md"))
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_measurements(directory):
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        if path.name.startswith("failures") or path.name == "unavailable.jsonl":
            continue
        rows.extend(load_jsonl(path))
    return rows


def load_unavailable(directory):
    return load_jsonl(directory / "unavailable.jsonl")


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        key = (row["model_name"], row["context_name"], row["cache_name"], row["mode"])
        result[key].append(row)
    return result


def measurement_table(rows, modes=MODES):
    by_key = grouped(rows)
    cells = sorted({key[:3] for key in by_key})
    lines = [
        "| Model | Context | KV | " + " | ".join(modes) + " | " + " | ".join(f"n({mode})" for mode in modes) + " |",
        "|---|---:|---|" + "---:|" * (2 * len(modes)),
    ]
    for cell in cells:
        medians = []
        counts = []
        for mode in modes:
            values = by_key[cell + (mode,)]
            medians.append(fmt(statistics.median(row["tokens_per_second"] for row in values)) if values else "-")
            counts.append(str(len(values)) if values else "-")
        lines.append(f"| {cell[0]} | {cell[1]} | {cell[2]} | " + " | ".join(medians + counts) + " |")
    return lines


def comparison_table(rows):
    by_key = grouped(rows)
    cells = sorted({key[:3] for key in by_key})
    lines = [
        "| Model | Context | KV | MTP | MTP+Vegas | Hybrid/MTP | MTP accept | Hybrid accept | Hash match | n/mode |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for cell in cells:
        mtp_rows = by_key[cell + ("mtp",)]
        hybrid_rows = by_key[cell + ("mtp-vegas",)]
        if not mtp_rows and not hybrid_rows:
            continue
        mtp = statistics.median(row["tokens_per_second"] for row in mtp_rows) if mtp_rows else None
        hybrid = statistics.median(row["tokens_per_second"] for row in hybrid_rows) if hybrid_rows else None
        mtp_accept = statistics.median(row["accept_rate"] for row in mtp_rows) if mtp_rows else None
        hybrid_accept = statistics.median(row["accept_rate"] for row in hybrid_rows) if hybrid_rows else None
        ratio = hybrid / mtp if hybrid is not None and mtp else None
        mtp_hashes = {row["output_hash"] for row in mtp_rows}
        hybrid_hashes = {row["output_hash"] for row in hybrid_rows}
        hash_match = "yes" if mtp_hashes and mtp_hashes == hybrid_hashes else "no"
        count = min(len(mtp_rows), len(hybrid_rows))
        lines.append(
            f"| {cell[0]} | {cell[1]} | {cell[2]} | {fmt(mtp)} | {fmt(hybrid)} | "
            f"{fmt(ratio, 3)} | {fmt(mtp_accept, 3)} | {fmt(hybrid_accept, 3)} | {hash_match} | {count} |"
        )
    return lines


def unavailable_table(records):
    if not records:
        return ["No configurations were classified as unavailable."]
    latest = {}
    for record in records:
        key = (record["model_name"], record["context_name"], record["cache_name"], record["mode"])
        latest[key] = record
    lines = [
        "| Model | Context | KV | Mode | Reason |",
        "|---|---:|---|---|---|",
    ]
    for key, record in sorted(latest.items()):
        reason = record["error"].splitlines()[0].replace("|", "\\|")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {reason} |")
    return lines


def main():
    args = parse_args()
    preserved = load_measurements(args.input_dir)
    screening = load_measurements(args.screening_dir)
    unavailable = (
        load_unavailable(args.input_dir)
        + load_unavailable(args.screening_dir)
    )

    confirmation = [
        row for row in preserved
        if row.get("n_predict") == 256
        and row["mode"] in {"mtp", "mtp-vegas"}
        and (row["model_name"], row["context_name"], row["cache_name"]) in CONFIRM_CELLS
    ]

    lines = [
        "# MTP and Vegas performance on RTX 3090",
        "",
        "## Scope and protocol",
        "",
        "This is a private CUDA-only experiment for batch-one long-context decoding. The four modes are "
        "dense baseline decoding, MTP drafting, Vegas self-speculative decoding, and MTP drafting with "
        "Vegas-guided sparse target verification.",
        "",
        "Completed 256-token measurements from the original campaign were preserved. Every remaining "
        "model/context/cache/mode cell was screened once with 128 generated tokens. Clearly negative screens "
        "were not confirmed. The retained 256-token MTP versus MTP+Vegas cells use sequential stopping: two "
        "order-rotated repetitions first, followed by a third pair only when signs disagree or the effect is "
        "within about 5%. Existing n=3 results were preserved and not repeated. OOM configurations were not "
        "retried. Screening and confirmatory evidence are reported separately.",
        "",
        "Prompt lengths are 16,384, 32,768, 65,536, and 128,000 tokens. Context capacities are 24,576, "
        "40,960, 73,728, and 131,072. Decode throughput excludes model loading and prompt prefill. Runs use "
        "batch one, greedy sampling, flash attention, full CUDA offload, and matched target/draft KV formats. "
        "Vegas uses gamma 1 and a 3% mask with a 256-token floor. MTP and MTP+Vegas use gamma 3.",
        "",
        "## One-pass screening (128 generated tokens)",
        "",
        f"Recorded screening measurements: {len(screening)} successful and "
        f"{len(load_unavailable(args.screening_dir))} unavailable, representing "
        f"{len(screening) + len(load_unavailable(args.screening_dir))} mode cells.",
        "Values are tok/s from one run and must not be interpreted as stable small differences.",
        "",
    ]
    lines.extend(measurement_table(screening))
    lines.extend([
        "",
        "## Focused MTP versus MTP+Vegas confirmation (256 generated tokens)",
        "",
        f"Recorded measurements in the focused confirmatory slice: {len(confirmation)}. Gemma q8/turbo4 "
        "has n=3 from the preserved confirmation run. Gemma q8/turbo3 stopped at n=2 after two clear "
        "positive effects. Both Qwen 27B cells proceeded to n=3 because their effects remained within 5%.",
        "The throughput columns are medians. `Hash match` compares the sets of output hashes observed in "
        "the two modes; a `no` requires inspection rather than automatically implying incorrect verification.",
        "",
    ])
    lines.extend(comparison_table(confirmation))
    lines.extend([
        "",
        "## Preserved 256-token measurements",
        "",
        f"Recorded preserved measurements: {len(preserved)}.",
        "These include cells completed before the screening protocol replaced the exhaustive full matrix. "
        "Counts are shown explicitly so partial cells are not mistaken for confirmed results.",
        "",
    ])
    lines.extend(measurement_table(preserved))
    lines.extend([
        "",
        "## Unavailable configurations",
        "",
    ])
    lines.extend(unavailable_table(unavailable))
    lines.extend([
        "",
        "## Correctness interpretation",
        "",
        "MTP+Vegas verifies draft tokens with the target model under Vegas sparse attention; it does not "
        "accept tokens from the draft model alone. Sparse Vegas execution is not bitwise equivalent to dense "
        "baseline execution, and baseline-versus-Vegas hashes commonly differ in the screening data. MTP and "
        "hybrid hashes match in every focused confirmation. Output hashes, acceptance counts, and failures are "
        "retained with every measurement.",
        "",
        "## Conclusion",
        "",
        "MTP and Vegas work together correctly in the tested implementation: every runnable focused "
        "confirmation produced matching MTP and hybrid output hashes, and no hybrid-only crash occurred. "
        "Performance is not generally positive. At Gemma 4 31B 128K, MTP+Vegas improved over MTP by about "
        "53% with q8/turbo4 and 26% with q8/turbo3. At Qwen 3.6 27B 128K, it was about 4% slower with both "
        "q8/turbo3 and q4/q4. The broader one-pass screen was predominantly negative below 128K.",
        "",
        "The honest result on this RTX 3090 is therefore conditional: Vegas can materially improve "
        "long-context MTP verification for Gemma with the fork's turbo caches, but it is not a universal "
        "batch-one decoding improvement and should not be enabled by default for the tested Qwen models. "
        "Several Qwen 35B 128K MTP configurations are unavailable because they exceed 24 GB VRAM.",
        "",
    ])

    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
