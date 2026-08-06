#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmark import run_one
from opencode_prompt import DEFAULT_OUTPUT, ensure_prompt


MODELS = {
    "qwen27": {
        "model": "/home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "selection_layer": 15,
        "anchor_tokens": 0,
        "recurrent": True,
    },
    "gemma4": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "draft_model": "/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/gemma-4-31B-it-qat-q4_0-assistant-Q8_0.gguf",
        "selection_layer": 59,
        "anchor_tokens": 0,
        "recurrent": False,
    },
    "qwen35": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "selection_layer": 39,
        "anchor_tokens": 16,
        "recurrent": True,
    },
}

CONTEXTS = {
    "32k": {"prompt_tokens": 32768, "context": 36864},
    "64k": {"prompt_tokens": 65536, "context": 66048},
}

MODES = ("mtp", "mtp-vegas", "mtp-hierarchical")
KERNELS = ("direct", "gather")
RATIOS = (1.0, 0.75, 0.5, 0.35, 0.2, 0.1)
HORIZONS = (8, 12, 19, 20, 32, 50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["mtp-hierarchical"])
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=["direct"])
    parser.add_argument("--ratios", nargs="+", type=float, choices=RATIOS, default=[0.5])
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--dense-interval", type=int, default=0)
    parser.add_argument("--rs-checkpoint-stride", type=int, default=1)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def make_args(args, model, context, ratio, horizon, mode, kernel):
    rounds = max(args.dense_interval, math.ceil(horizon / args.gamma) + 2)
    return SimpleNamespace(
        binary=args.binary,
        model=model["model"],
        draft_model=model["draft_model"],
        prompt=None,
        conversation_file=str(DEFAULT_OUTPUT),
        prompt_tokens=context["prompt_tokens"],
        reference_tokens=0,
        context=context["context"],
        predict=args.predict,
        repetitions=1,
        modes=[mode],
        gamma=args.gamma,
        self_gamma=None,
        mtp_gamma=args.gamma,
        adaptive_gamma=False,
        adaptive_beta=0.9,
        hier_target=horizon,
        hier_max_tokens=horizon,
        hier_max_rounds=rounds,
        hier_max_corrections=rounds,
        hier_dense_interval=args.dense_interval,
        hier_rs_checkpoint_stride=args.rs_checkpoint_stride,
        hier_trace=args.trace,
        hier_recompute_state=model["recurrent"] and horizon > 10 and args.dense_interval == 0,
        same_prefix_trace=False,
        mtp_ubatch=64,
        ratio=ratio,
        selection_layer=model["selection_layer"],
        anchor_tokens=model["anchor_tokens"],
        refresh_interval=1,
        min_tokens=256,
        batch=1024,
        ubatch=64,
        seed=1234,
        temperature=0.0,
        cache_type_k="q8_0",
        cache_type_v="q4_0",
        sparse_kernel=kernel,
        draft_cache_type_k="q8_0",
        draft_cache_type_v="q4_0",
        output=args.output_dir / "results.jsonl",
    )


def load_results(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def result_key(row):
    return (
        row["configuration"],
        row["requested_mode"],
        row["requested_sparse_kernel"],
        row["matrix_ratio"],
        row["matrix_horizon"],
        row["repetition"],
    )


def compact_row(row):
    compact = {
        "configuration": row["configuration"],
        "mode": row["requested_mode"],
        "kernel": row["requested_sparse_kernel"],
        "ratio": row["matrix_ratio"],
        "horizon": row["matrix_horizon"],
        "repetition": row["repetition"],
        "tokens_per_second": row["tokens_per_second"],
        "output_hash": row["output_hash"],
    }
    if row["requested_mode"] == "mtp-hierarchical":
        compact.update({
            "outer_cycles": row["hierarchical_outer_cycles"],
            "committed_per_dense_cycle": row["hierarchical_committed_per_dense_cycle"],
            "sparse_agreement": row["hierarchical_sparse_agreement"],
            "dense_acceptance": row["hierarchical_dense_acceptance"],
            "sparse_ms": row["hierarchical_sparse_ms"],
            "dense_ms": row["hierarchical_dense_ms"],
            "recompute_ms": row["hierarchical_recompute_ms"],
            "direct_dense_rounds": row.get("hierarchical_direct_dense_rounds", 0),
            "replayed_updates": row.get("hierarchical_recurrent_replayed_updates", 0),
            "provisional_histogram": row["hierarchical_provisional_histogram"],
            "state_failures": (
                row["hierarchical_snapshot_failures"]
                + row["hierarchical_rollback_failures"]
                + row["hierarchical_position_mismatches"]
            ),
        })
    return compact


def write_summary(output_dir, results):
    rows = sorted((compact_row(row) for row in results), key=lambda row: (
        row["configuration"], row["mode"], row["kernel"], row["ratio"], row["horizon"], row["repetition"]
    ))
    (output_dir / "summary.json").write_text(
        json.dumps({"schema_version": 2, "runs": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_manifest(args):
    design = {
        "schema_version": 2,
        "prompt_file": str(DEFAULT_OUTPUT),
        "prompt_sha256": hashlib.sha256(DEFAULT_OUTPUT.read_bytes()).hexdigest(),
        "models": args.models,
        "contexts": args.contexts,
        "modes": args.modes,
        "kernels": args.kernels,
        "ratios": args.ratios,
        "horizons": args.horizons,
        "gamma": args.gamma,
        "dense_interval": args.dense_interval,
        "rs_checkpoint_stride": args.rs_checkpoint_stride,
        "predict": args.predict,
        "repetitions": args.repetitions,
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {key: existing[key] for key in design} != design:
            raise SystemExit("existing matrix manifest does not match the requested design")
        return
    design["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    path.write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    ensure_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(args)
    output_path = args.output_dir / "results.jsonl"
    results = load_results(output_path)
    completed = {result_key(row) for row in results}

    for model_name in args.models:
        model = MODELS[model_name]
        for context_name in args.contexts:
            context = CONTEXTS[context_name]
            configuration = f"{model_name}-q8-q4-{context_name}"
            for mode in args.modes:
                # Vanilla MTP has no sparse ratio or dense horizon. Run it once per configuration.
                ratios = [1.0] if mode == "mtp" else args.ratios
                horizons = [args.gamma] if mode != "mtp-hierarchical" else args.horizons
                kernels = ["direct"] if mode == "mtp" else args.kernels
                for kernel in kernels:
                    for ratio in ratios:
                        for horizon in horizons:
                            for repetition in range(args.repetitions):
                                key = (configuration, mode, kernel, ratio, horizon, repetition)
                                if key in completed:
                                    continue
                                print(
                                    f"VEGAS_HIERARCHICAL_START configuration={configuration} mode={mode} "
                                    f"kernel={kernel} ratio={ratio} horizon={horizon} repetition={repetition}",
                                    flush=True,
                                )
                                row = run_one(
                                    make_args(args, model, context, ratio, horizon, mode, kernel),
                                    mode,
                                    repetition,
                                    0,
                                )
                                row.update({
                                    "configuration": configuration,
                                    "matrix_ratio": ratio,
                                    "matrix_horizon": horizon,
                                })
                                results.append(row)
                                completed.add(key)
                                with output_path.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                                write_summary(args.output_dir, results)
                                print("VEGAS_HIERARCHICAL_RESULT " + json.dumps(compact_row(row), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
