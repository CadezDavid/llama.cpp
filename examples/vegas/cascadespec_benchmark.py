#!/usr/bin/env python3

import argparse
import hashlib
import json
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
        "ratio": 0.10,
    },
    "gemma4": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "draft_model": "/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/"
                       "gemma-4-31B-it-qat-q4_0-assistant-Q8_0.gguf",
        "selection_layer": 59,
        "anchor_tokens": 0,
        "ratio": 0.20,
    },
    "qwen35": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "selection_layer": 39,
        "anchor_tokens": 16,
        "ratio": 0.20,
    },
}

MODES = ("mtp", "mtp-hierarchical")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed 32K plain-MTP versus CascadeSpec cadence check.")
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--predict", type=int, default=128)
    return parser.parse_args()


def make_args(args, model):
    return SimpleNamespace(
        binary=args.binary,
        model=model["model"],
        draft_model=model["draft_model"],
        prompt=None,
        conversation_file=str(DEFAULT_OUTPUT),
        prompt_tokens=32768,
        reference_tokens=0,
        context=36864,
        predict=args.predict,
        repetitions=1,
        modes=list(MODES),
        gamma=4,
        self_gamma=None,
        mtp_gamma=4,
        adaptive_gamma=False,
        adaptive_beta=0.9,
        hier_target=19,
        hier_max_tokens=19,
        hier_max_rounds=4,
        hier_max_corrections=4,
        hier_dense_interval=4,
        hier_rs_checkpoint_stride=4,
        hier_trace=False,
        hier_recompute_state=False,
        same_prefix_trace=False,
        mtp_ubatch=64,
        ratio=model["ratio"],
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
        sparse_kernel="direct",
        draft_cache_type_k="q8_0",
        draft_cache_type_v="q4_0",
        output=args.output_dir / "results.jsonl",
    )


def load_results(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(args):
    manifest = {
        "schema_version": 1,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "binary": args.binary,
        "models": args.models,
        "modes": list(MODES),
        "context": 36864,
        "prompt_tokens": 32768,
        "predict": args.predict,
        "gamma": 4,
        "dense_interval": 4,
        "provisional_token_cap": 19,
        "dense_input_row_cap": 20,
        "rs_checkpoint_stride": 4,
        "cache_type_k": "q8_0",
        "cache_type_v": "q4_0",
        "temperature": 0.0,
        "seed": 1234,
        "conversation_file": str(DEFAULT_OUTPUT),
        "conversation_sha256": hashlib.sha256(DEFAULT_OUTPUT.read_bytes()).hexdigest(),
    }
    path = args.output_dir / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: previous.get(key) for key in manifest if key != "git_commit"}
        expected = {key: value for key, value in manifest.items() if key != "git_commit"}
        if comparable != expected:
            raise SystemExit("existing CascadeSpec manifest does not match this benchmark")
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(results, models):
    runs = []
    all_hashes_equal = True
    all_state_checks_passed = True
    for model_name in models:
        pair = {row["requested_mode"]: row for row in results if row["configuration"] == model_name}
        if not all(mode in pair for mode in MODES):
            continue
        mtp = pair["mtp"]
        cascade = pair["mtp-hierarchical"]
        hash_equal = mtp["output_hash"] == cascade["output_hash"]
        all_hashes_equal &= hash_equal
        state_failures = (
            cascade["hierarchical_snapshot_failures"]
            + cascade["hierarchical_rollback_failures"]
            + cascade["hierarchical_position_mismatches"]
            + cascade["hierarchical_recurrent_restore_failures"]
        )
        all_state_checks_passed &= state_failures == 0
        runs.append({
            "configuration": model_name,
            "output_hash_equal": hash_equal,
            "plain_mtp": {
                "tokens_per_second": mtp["tokens_per_second"],
                "total_ms": mtp["total_ms"],
                "cycles": mtp["cycles"],
                "draft_ms": mtp["draft_ms"],
                "verify_ms": mtp["verify_ms"],
            },
            "cascadespec": {
                "tokens_per_second": cascade["tokens_per_second"],
                "total_ms": cascade["total_ms"],
                "outer_cycles": cascade["hierarchical_outer_cycles"],
                "committed_per_dense_cycle": cascade["hierarchical_committed_per_dense_cycle"],
                "direct_dense_rounds": cascade["hierarchical_direct_dense_rounds"],
                "skipped_sparse_decodes": cascade["hierarchical_skipped_sparse_decodes"],
                "mtp_ms": cascade["hierarchical_mtp_ms"],
                "mtp_process_ms": cascade["hierarchical_mtp_process_ms"],
                "sparse_ms": cascade["hierarchical_sparse_ms"],
                "dense_ms": cascade["hierarchical_dense_ms"],
                "rollback_ms": cascade["hierarchical_rollback_ms"],
                "recurrent_checkpoint_ms": cascade["hierarchical_recurrent_checkpoint_ms"],
                "recurrent_checkpoint_copy_ms": cascade["hierarchical_recurrent_checkpoint_copy_ms"],
                "recurrent_replay_gdn_ms": cascade["hierarchical_recurrent_replay_gdn_ms"],
                "recurrent_replay_conv_ms": cascade["hierarchical_recurrent_replay_conv_ms"],
                "replayed_updates": cascade["hierarchical_recurrent_replayed_updates"],
                "state_failures": state_failures,
            },
            "speedup_percent": 100.0 * (
                cascade["tokens_per_second"] / mtp["tokens_per_second"] - 1.0),
        })
    return {
        "schema_version": 2,
        "all_cascadespec_state_checks_passed": all_state_checks_passed,
        "all_output_hashes_equal": all_hashes_equal,
        "output_hash_note": (
            "Advisory only: changing the dense verification batch length can change floating-point "
            "rounding and therefore greedy output, including between two plain-MTP gamma values."),
        "runs": runs,
    }


def main():
    args = parse_args()
    ensure_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args)

    output_path = args.output_dir / "results.jsonl"
    results = load_results(output_path)
    completed = {(row["configuration"], row["requested_mode"]) for row in results}
    order = len(results)
    for model_name in args.models:
        run_args = make_args(args, MODELS[model_name])
        for mode in MODES:
            if (model_name, mode) in completed:
                continue
            print(f"CASCADESPEC_BENCHMARK_START model={model_name} mode={mode}", flush=True)
            row = run_one(run_args, mode, 0, order)
            row["configuration"] = model_name
            results.append(row)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            order += 1

            summary = summarize(results, args.models)
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"CASCADESPEC_BENCHMARK_RESULT model={model_name} mode={mode} "
                f"tps={row['tokens_per_second']:.3f} hash={row['output_hash']}",
                flush=True,
            )

    summary = summarize(results, args.models)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(summary["runs"]) != len(args.models):
        raise SystemExit("benchmark did not produce a complete mode pair for every requested model")
    if not summary["all_cascadespec_state_checks_passed"]:
        raise SystemExit("CascadeSpec reported a state-integrity failure")


if __name__ == "__main__":
    main()
