#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark import run_one
from opencode_prompt import ensure_prompt


PROMPT = "/tmp/opencode-vegas-technical-conversations.txt"

CONFIGURATIONS = {
    "qwen27-q8-turbo4-64k": {
        "model": "/home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "context": 73728,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "ratio": 0.072768,
        "selection_layer": 15,
        "anchor_tokens": 0,
        "refresh_interval": 2,
        "ubatch": 128,
    },
    "gemma4-q8-turbo4-64k": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "draft_model": "/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/gemma-4-31B-it-qat-q4_0-assistant-Q8_0.gguf",
        "context": 73728,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "ratio": 0.03,
        "selection_layer": 59,
        "anchor_tokens": 0,
        "refresh_interval": 1,
        "ubatch": 128,
    },
    "qwen35-q8-q4-64k": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "context": 66048,
        "cache_k": "q8_0",
        "cache_v": "q4_0",
        "ratio": 0.03,
        "selection_layer": 39,
        "anchor_tokens": 16,
        "refresh_interval": 1,
        "ubatch": 64,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    return parser.parse_args()


def make_args(args, config):
    return SimpleNamespace(
        binary=args.binary,
        model=config["model"],
        draft_model=config["draft_model"],
        prompt=PROMPT,
        prompt_tokens=65536,
        context=config["context"],
        predict=args.predict,
        repetitions=args.repetitions,
        modes=["mtp-hierarchical"],
        gamma=3,
        self_gamma=None,
        mtp_gamma=3,
        adaptive_gamma=False,
        adaptive_beta=0.9,
        hier_target=8,
        hier_max_tokens=10,
        hier_max_rounds=3,
        hier_max_corrections=2,
        hier_trace=True,
        mtp_ubatch=config["ubatch"],
        ratio=config["ratio"],
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
        draft_cache_type_k=config["cache_k"],
        draft_cache_type_v=config["cache_v"],
        output=args.output_dir / "results.jsonl",
    )


def load_results(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(results):
    return {
        row["configuration"]: {
            "tokens_per_second": row["tokens_per_second"],
            "output_hash": row["output_hash"],
            "outer_cycles": row["hierarchical_outer_cycles"],
            "committed_per_dense_cycle": row["hierarchical_committed_per_dense_cycle"],
            "sparse_agreement": row["hierarchical_sparse_agreement"],
            "dense_acceptance": row["hierarchical_dense_acceptance"],
            "device_entropy_samples": row["hierarchical_device_entropy_samples"],
            "fallback_entropy_samples": row["hierarchical_fallback_entropy_samples"],
            "rollback_failures": row["hierarchical_rollback_failures"],
            "position_mismatches": row["hierarchical_position_mismatches"],
            "snapshot_failures": row["hierarchical_snapshot_failures"],
        }
        for row in results
    }


def main():
    args = parse_args()
    ensure_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.jsonl"
    results = load_results(output_path)
    completed = {
        (row["configuration"], row["repetition"])
        for row in results
    }

    for configuration_name in args.configurations:
        config = CONFIGURATIONS[configuration_name]
        for repetition in range(args.repetitions):
            key = (configuration_name, repetition)
            if key in completed:
                continue
            result = run_one(make_args(args, config), "mtp-hierarchical", repetition, 0)
            result["configuration"] = configuration_name
            results.append(result)
            completed.add(key)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summarize(results), stream, indent=2, sort_keys=True)
                stream.write("\n")
            print("VEGAS_HIERARCHICAL_MATRIX " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
