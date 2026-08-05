#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from benchmark import run_one, summarize


MODELS = {
    "gemma4-31b": {
        "model": "/home/david/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "draft_model": "/home/david/models/gemma-4-31B-it-qat-q4_0-unquantized-assistant/gemma-4-31B-it-qat-q4_0-assistant-Q8_0.gguf",
    },
    "qwen36-35b-a3b": {
        "model": "/home/david/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "draft_model": None,
    },
    "qwen36-27b": {
        "model": "/home/david/models/Qwen3.6-27B-MTP/Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "draft_model": None,
    },
}

CONTEXTS = {
    "16k": (16384, 24576),
    "32k": (32768, 40960),
    "64k": (65536, 73728),
    "128k": (128000, 131072),
}

CACHES = {
    "q8-q4": ("q8_0", "q4_0"),
    "q8-q8": ("q8_0", "q8_0"),
    "q4-q4": ("q4_0", "q4_0"),
    "q8-turbo4": ("q8_0", "turbo4"),
    "q8-turbo3": ("q8_0", "turbo3"),
}

MODES = ["baseline", "mtp", "vegas", "mtp-vegas"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-vegas/bin/llama-vegas")
    parser.add_argument("--prompt", default="tests/test-backend-ops.cpp")
    parser.add_argument("--predict", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--self-gamma", type=int, default=1)
    parser.add_argument("--mtp-gamma", type=int, default=3)
    parser.add_argument("--adaptive-gamma", action="store_true")
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--mtp-ubatch", type=int, default=128)
    parser.add_argument("--long-mtp-ubatch", type=int, default=64)
    parser.add_argument("--ratio", type=float, default=0.03)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--long-ubatch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/vegas/results/mtp-matrix"))
    parser.add_argument("--skip-complete-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    parser.add_argument("--caches", nargs="+", choices=CACHES, default=list(CACHES))
    return parser.parse_args()


def read_results(path):
    if not path.exists():
        return []
    results = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                results.append(json.loads(line))
    return results


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = args.output_dir / "failures.jsonl"
    unavailable_path = args.output_dir / "unavailable.jsonl"
    unavailable = read_results(unavailable_path)
    reference_unavailable = []
    if args.skip_complete_dir is not None:
        reference_unavailable = read_results(args.skip_complete_dir / "unavailable.jsonl")
    failures = 0

    for model_name in args.models:
        model = MODELS[model_name]
        for context_name in args.contexts:
            prompt_tokens, context = CONTEXTS[context_name]
            ubatch = args.long_ubatch if context_name == "128k" else args.ubatch
            mtp_ubatch = args.long_mtp_ubatch if context_name == "128k" else args.mtp_ubatch
            for cache_name in args.caches:
                cache_k, cache_v = CACHES[cache_name]
                if args.skip_complete_dir is not None:
                    reference_output = args.skip_complete_dir / f"{model_name}-{context_name}-{cache_name}.jsonl"
                    reference_modes = {item["mode"] for item in read_results(reference_output)}
                    reference_modes.update(
                        item["mode"] for item in reference_unavailable
                        if item["model_name"] == model_name and item["context_name"] == context_name
                        and item["cache_name"] == cache_name
                    )
                    if set(MODES).issubset(reference_modes):
                        continue
                output = args.output_dir / f"{model_name}-{context_name}-{cache_name}.jsonl"
                results = read_results(output)
                completed = {
                    (item["repetition"], item["mode"])
                    for item in results
                    if item["mode"] in args.modes and item["repetition"] < args.repetitions
                }
                unavailable_modes = {
                    item["mode"] for item in unavailable
                    if item["model_name"] == model_name and item["context_name"] == context_name
                    and item["cache_name"] == cache_name
                }

                run_args = SimpleNamespace(
                    binary=args.binary,
                    model=model["model"],
                    draft_model=model["draft_model"],
                    prompt=args.prompt,
                    prompt_tokens=prompt_tokens,
                    context=context,
                    predict=args.predict,
                    repetitions=args.repetitions,
                    modes=args.modes,
                    gamma=args.mtp_gamma,
                    self_gamma=args.self_gamma,
                    mtp_gamma=args.mtp_gamma,
                    adaptive_gamma=args.adaptive_gamma,
                    adaptive_beta=args.adaptive_beta,
                    mtp_ubatch=mtp_ubatch,
                    ratio=args.ratio,
                    min_tokens=args.min_tokens,
                    selection_layer=None,
                    anchor_tokens=0,
                    refresh_interval=1,
                    batch=args.batch,
                    ubatch=ubatch,
                    seed=args.seed,
                    temperature=args.temperature,
                    cache_type_k=cache_k,
                    cache_type_v=cache_v,
                    draft_cache_type_k=None,
                    draft_cache_type_v=None,
                    output=output,
                )

                for repetition in range(args.repetitions):
                    offset = repetition % len(args.modes)
                    modes = args.modes[offset:] + args.modes[:offset]
                    for order, mode in enumerate(modes):
                        if (repetition, mode) in completed or mode in unavailable_modes:
                            continue
                        try:
                            result = run_one(run_args, mode, repetition, order)
                            result.update({
                                "model_name": model_name,
                                "context_name": context_name,
                                "cache_name": cache_name,
                            })
                            results.append(result)
                            completed.add((repetition, mode))
                            with output.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(result, sort_keys=True) + "\n")
                            print("VEGAS_MATRIX " + json.dumps(result, sort_keys=True), flush=True)
                        except Exception as error:
                            error_text = str(error)
                            failure = {
                                "model_name": model_name,
                                "context_name": context_name,
                                "cache_name": cache_name,
                                "mode": mode,
                                "repetition": repetition,
                                "error": error_text,
                                "time": time.time(),
                            }
                            if "out of memory" in error_text.lower() or "failed to create mtp context" in error_text.lower():
                                unavailable_modes.add(mode)
                                unavailable.append(failure)
                                with unavailable_path.open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps(failure, sort_keys=True) + "\n")
                                print("VEGAS_MATRIX_UNAVAILABLE " + json.dumps(failure, sort_keys=True), flush=True)
                                continue
                            with failures_path.open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(failure, sort_keys=True) + "\n")
                            print("VEGAS_MATRIX_FAILURE " + json.dumps(failure, sort_keys=True), flush=True)
                            failures += 1

                expected = args.repetitions * len(args.modes)
                if len(completed) == expected:
                    summary_path = output.with_suffix(".summary.json")
                    with summary_path.open("w", encoding="utf-8") as stream:
                        json.dump(summarize(results), stream, indent=2, sort_keys=True)
                        stream.write("\n")

    if failures:
        raise SystemExit(f"matrix incomplete: {failures} run(s) failed")


if __name__ == "__main__":
    main()
