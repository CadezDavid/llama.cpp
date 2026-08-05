#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmark import run_one
from opencode_prompt import DEFAULT_OUTPUT, ensure_prompt


CONFIGURATIONS = {
    "qwen27-q8-turbo4-64k": {
        "model": "/home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "context": 73728,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "kernel": "gather",
        "selection_layer": 15,
        "anchor_tokens": 0,
        "refresh_interval": 2,
        "ubatch": 128,
    },
    "gemma4-q8-turbo4-64k": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "context": 73728,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "kernel": "gather",
        "selection_layer": 59,
        "anchor_tokens": 0,
        "refresh_interval": 1,
        "ubatch": 128,
    },
    "qwen35-q8-q4-64k": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "context": 66048,
        "cache_k": "q8_0",
        "cache_v": "q4_0",
        "kernel": "direct",
        "selection_layer": 39,
        "anchor_tokens": 16,
        "refresh_interval": 1,
        "ubatch": 64,
    },
}

RATIOS = (1.0, 0.5, 0.2, 0.1)
GATE = {
    "min_top1_agreement": 0.99,
    "max_mean_total_variation": 0.03,
    "max_mean_jensen_shannon": 0.002,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=65536)
    parser.add_argument("--reference-tokens", type=int, default=128)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--ratios", nargs="+", type=float, choices=RATIOS, default=list(RATIOS))
    return parser.parse_args()


def make_args(args, config, ratio):
    return SimpleNamespace(
        binary=args.binary,
        model=config["model"],
        draft_model=None,
        prompt=None,
        conversation_file=str(DEFAULT_OUTPUT),
        prompt_tokens=args.prompt_tokens,
        reference_tokens=args.reference_tokens,
        context=config["context"],
        predict=args.reference_tokens,
        repetitions=1,
        modes=["same-prefix"],
        gamma=1,
        self_gamma=None,
        mtp_gamma=None,
        adaptive_gamma=False,
        adaptive_beta=0.9,
        hier_target=8,
        hier_max_tokens=10,
        hier_max_rounds=3,
        hier_max_corrections=2,
        hier_trace=False,
        same_prefix_trace=True,
        mtp_ubatch=config["ubatch"],
        ratio=ratio,
        selection_layer=config["selection_layer"],
        anchor_tokens=config["anchor_tokens"],
        refresh_interval=config["refresh_interval"],
        min_tokens=256,
        batch=1024,
        ubatch=config["ubatch"],
        seed=1234,
        temperature=0.0,
        cache_type_k=config["cache_k"],
        cache_type_v=config["cache_v"],
        sparse_kernel=config["kernel"],
        draft_cache_type_k=None,
        draft_cache_type_v=None,
        output=args.output_dir / "results.jsonl",
    )


def load_results(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def gate_failures(row):
    failures = []
    if row["same_prefix_top1_agreement"] < GATE["min_top1_agreement"]:
        failures.append("top1_agreement")
    if row["same_prefix_mean_total_variation"] > GATE["max_mean_total_variation"]:
        failures.append("mean_total_variation")
    if row["same_prefix_mean_jensen_shannon"] > GATE["max_mean_jensen_shannon"]:
        failures.append("mean_jensen_shannon")
    for metric in (
        "same_prefix_snapshot_failures",
        "same_prefix_rollback_failures",
        "same_prefix_position_mismatches",
    ):
        if row[metric] != 0:
            failures.append(metric)
    return failures


def summarize(results, gate_status):
    rows = []
    for result in results:
        rows.append({
            "configuration": result["configuration"],
            "ratio": result["matrix_ratio"],
            "prompt_tokens": result["n_prompt"],
            "prompt_hash": result["conversation_prompt_hash"],
            "reference_hash": result["conversation_reference_hash"],
            "top1_agreement": result["same_prefix_top1_agreement"],
            "mean_tv": result["same_prefix_mean_total_variation"],
            "p95_tv": result["same_prefix_p95_total_variation"],
            "mean_js": result["same_prefix_mean_jensen_shannon"],
            "reference_nll_delta": result["same_prefix_mean_reference_nll_delta"],
            "sparse_ms": result["same_prefix_sparse_ms"],
            "dense_ms": result["same_prefix_dense_ms"],
            "sparse_over_dense_time": (
                result["same_prefix_sparse_ms"] / result["same_prefix_dense_ms"]
                if result["same_prefix_dense_ms"] else 0.0
            ),
            "state_failures": (
                result["same_prefix_snapshot_failures"]
                + result["same_prefix_rollback_failures"]
                + result["same_prefix_position_mismatches"]
            ),
        })
    return {
        "schema_version": 1,
        "gate": GATE,
        "gate_status": gate_status,
        "runs": rows,
    }


def write_outputs(output_dir, results, gate_status):
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summarize(results, gate_status), stream, indent=2, sort_keys=True)
        stream.write("\n")


def ensure_manifest(args):
    prompt_bytes = DEFAULT_OUTPUT.read_bytes()
    design = {
        "schema_version": 1,
        "prompt_file": str(DEFAULT_OUTPUT),
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "prompt_tokens": args.prompt_tokens,
        "reference_tokens": args.reference_tokens,
        "configurations": args.configurations,
        "ratios": args.ratios,
        "gate": GATE,
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: existing[key] for key in design}
        if comparable != design:
            raise SystemExit("existing matrix manifest does not match the requested design")
        return

    design["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    design["models"] = {
        name: {
            "path": CONFIGURATIONS[name]["model"],
            "size": Path(CONFIGURATIONS[name]["model"]).stat().st_size,
            "cache_k": CONFIGURATIONS[name]["cache_k"],
            "cache_v": CONFIGURATIONS[name]["cache_v"],
            "kernel": CONFIGURATIONS[name]["kernel"],
        }
        for name in args.configurations
    }
    path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if any(ratio not in RATIOS for ratio in args.ratios):
        raise SystemExit(f"ratios must be selected from {RATIOS}")

    ensure_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(args)
    output_path = args.output_dir / "results.jsonl"
    results = load_results(output_path)
    completed = {
        (row["configuration"], row["matrix_ratio"])
        for row in results
    }

    selected = [(name, CONFIGURATIONS[name]) for name in args.configurations]
    gate_rows = {}
    for configuration, config in selected:
        key = (configuration, 1.0)
        existing = next((row for row in results if key == (row["configuration"], row["matrix_ratio"])), None)
        if existing is None:
            print(f"VEGAS_QUALITY_START configuration={configuration} ratio=1.0", flush=True)
            existing = run_one(make_args(args, config, 1.0), "same-prefix", 0, 0)
            existing["configuration"] = configuration
            existing["matrix_ratio"] = 1.0
            existing["gate_failures"] = gate_failures(existing)
            results.append(existing)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(existing, sort_keys=True) + "\n")
        gate_rows[configuration] = existing
        completed.add(key)
        write_outputs(args.output_dir, results, "pending")
        print("VEGAS_QUALITY_RESULT " + json.dumps({
            "configuration": configuration,
            "ratio": 1.0,
            "failures": gate_failures(existing),
            "top1": existing["same_prefix_top1_agreement"],
            "mean_tv": existing["same_prefix_mean_total_variation"],
            "mean_js": existing["same_prefix_mean_jensen_shannon"],
        }, sort_keys=True), flush=True)

    failed = {name: gate_failures(row) for name, row in gate_rows.items() if gate_failures(row)}
    if failed:
        write_outputs(args.output_dir, results, "failed")
        raise SystemExit("100%-retention gate failed; lower retention ratios were not run: " + json.dumps(failed))

    for ratio in args.ratios:
        if ratio == 1.0:
            continue
        for configuration, config in selected:
            key = (configuration, ratio)
            if key in completed:
                continue
            print(f"VEGAS_QUALITY_START configuration={configuration} ratio={ratio}", flush=True)
            result = run_one(make_args(args, config, ratio), "same-prefix", 0, 0)
            result["configuration"] = configuration
            result["matrix_ratio"] = ratio
            result["gate_failures"] = []
            expected = gate_rows[configuration]
            if result["conversation_prompt_hash"] != expected["conversation_prompt_hash"]:
                raise RuntimeError(f"prompt hash changed for {configuration} at ratio {ratio}")
            if result["conversation_reference_hash"] != expected["conversation_reference_hash"]:
                raise RuntimeError(f"reference hash changed for {configuration} at ratio {ratio}")
            results.append(result)
            completed.add(key)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            write_outputs(args.output_dir, results, "passed")
            print("VEGAS_QUALITY_RESULT " + json.dumps({
                "configuration": configuration,
                "ratio": ratio,
                "top1": result["same_prefix_top1_agreement"],
                "mean_tv": result["same_prefix_mean_total_variation"],
                "p95_tv": result["same_prefix_p95_total_variation"],
                "mean_js": result["same_prefix_mean_jensen_shannon"],
                "reference_nll_delta": result["same_prefix_mean_reference_nll_delta"],
            }, sort_keys=True), flush=True)

    write_outputs(args.output_dir, results, "passed")


if __name__ == "__main__":
    main()
