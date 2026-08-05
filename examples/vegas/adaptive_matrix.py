#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from benchmark import run_one
from opencode_prompt import ensure_prompt


PROMPT = "/tmp/opencode-vegas-technical-conversations.txt"


CONFIGURATIONS = {
    "qwen27-q8-turbo4": {
        "model": "/home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "prompt": PROMPT,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "fixed_gamma": 4,
        "selection_layer": 15,
        "anchor_tokens": 0,
        "refresh_interval": 2,
    },
    "gemma4-q8-turbo4": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "draft_model": "/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/gemma-4-31B-it-qat-q4_0-assistant-Q8_0.gguf",
        "prompt": PROMPT,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "fixed_gamma": 3,
        "selection_layer": 59,
        "anchor_tokens": 0,
        "refresh_interval": 1,
    },
    "qwen35-q8-turbo4": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "prompt": PROMPT,
        "cache_k": "q8_0",
        "cache_v": "turbo4",
        "fixed_gamma": 5,
        "selection_layer": 39,
        "anchor_tokens": 16,
        "refresh_interval": 1,
    },
    "qwen35-q8-q4": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
        "prompt": PROMPT,
        "cache_k": "q8_0",
        "cache_v": "q4_0",
        "fixed_gamma": 5,
        "selection_layer": 39,
        "anchor_tokens": 16,
        "refresh_interval": 1,
    },
}

CONTEXTS = {
    "32k": (32768, 40960),
    "64k": (65536, 73728),
}

VARIANTS = {
    "dense-fixed": ("mtp", False),
    "sparse-fixed": ("mtp-vegas", False),
    "sparse-adaptive": ("mtp-vegas", True),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    return parser.parse_args()


def load_results(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ratio_for(configuration, prompt_tokens):
    if configuration == "qwen27-q8-turbo4":
        return min(0.10, max(0.05, 0.04 + prompt_tokens / 2_000_000.0))
    return 0.03


def make_args(args, configuration_name, configuration, context_name):
    prompt_tokens, context = CONTEXTS[context_name]
    qwen35_long = configuration_name.startswith("qwen35-") and context_name == "64k"
    if qwen35_long:
        context = 66048
    return SimpleNamespace(
        binary=args.binary,
        model=configuration["model"],
        draft_model=configuration["draft_model"],
        prompt=configuration["prompt"],
        prompt_tokens=prompt_tokens,
        context=context,
        predict=args.predict,
        repetitions=args.repetitions,
        modes=[],
        gamma=configuration["fixed_gamma"],
        self_gamma=None,
        mtp_gamma=configuration["fixed_gamma"],
        adaptive_gamma=False,
        adaptive_beta=args.adaptive_beta,
        hier_target=8,
        hier_max_tokens=10,
        hier_max_rounds=3,
        hier_max_corrections=2,
        hier_trace=False,
        mtp_ubatch=64 if qwen35_long else 128,
        ratio=ratio_for(configuration_name, prompt_tokens),
        selection_layer=configuration["selection_layer"],
        anchor_tokens=configuration["anchor_tokens"],
        refresh_interval=configuration["refresh_interval"],
        min_tokens=256,
        batch=1024,
        ubatch=64 if qwen35_long else 128,
        seed=1234,
        temperature=0.0,
        cache_type_k=configuration["cache_k"],
        cache_type_v=configuration["cache_v"],
        draft_cache_type_k=configuration["cache_k"],
        draft_cache_type_v=configuration["cache_v"],
        output=args.output_dir / "results.jsonl",
    )


def summarize(results):
    summary = {}
    for configuration in CONFIGURATIONS:
        summary[configuration] = {}
        for context in CONTEXTS:
            summary[configuration][context] = {}
            for variant in VARIANTS:
                rows = [
                    row for row in results
                    if row["configuration"] == configuration
                    and row["context_name"] == context
                    and row["variant"] == variant
                ]
                if not rows:
                    continue
                summary[configuration][context][variant] = {
                    "n": len(rows),
                    "median_tps": statistics.median(row["tokens_per_second"] for row in rows),
                    "median_accept_rate": statistics.median(row["accept_rate"] for row in rows),
                    "output_hashes": sorted({row["output_hash"] for row in rows}),
                    "adaptive_mean_gamma": statistics.median(
                        row.get("adaptive_mean_gamma", 0.0) for row in rows
                    ),
                }
    return summary


def main():
    args = parse_args()
    ensure_prompt()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.jsonl"
    results = load_results(output_path)
    completed = {
        (row["configuration"], row["context_name"], row["variant"], row["repetition"])
        for row in results
    }

    task_index = 0
    for configuration_name, configuration in CONFIGURATIONS.items():
        for context_name in CONTEXTS:
            for repetition in range(args.repetitions):
                variants = list(args.variants)
                offset = task_index % len(variants)
                variants = variants[offset:] + variants[:offset]
                task_index += 1
                for order, variant in enumerate(variants):
                    key = (configuration_name, context_name, variant, repetition)
                    if key in completed:
                        continue

                    mode, adaptive = VARIANTS[variant]
                    run_args = make_args(args, configuration_name, configuration, context_name)
                    run_args.adaptive_gamma = adaptive
                    result = run_one(run_args, mode, repetition, order)
                    result.update({
                        "configuration": configuration_name,
                        "context_name": context_name,
                        "variant": variant,
                    })
                    results.append(result)
                    completed.add(key)
                    with output_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                    print("VEGAS_ADAPTIVE_MATRIX " + json.dumps(result, sort_keys=True), flush=True)

                    summary_path = args.output_dir / "summary.json"
                    with summary_path.open("w", encoding="utf-8") as stream:
                        json.dump(summarize(results), stream, indent=2, sort_keys=True)
                        stream.write("\n")


if __name__ == "__main__":
    main()
