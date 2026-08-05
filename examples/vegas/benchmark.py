#!/usr/bin/env python3

import argparse
import json
import random
import statistics
import subprocess
import time
from pathlib import Path


def gpu_state():
    fields = [
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
        "memory.used",
    ]
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    values = subprocess.check_output(command, text=True).strip().split(", ")
    return dict(zip(fields, (float(value) for value in values)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--predict", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=["baseline", "mtp", "vegas", "mtp-vegas"])
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--self-gamma", type=int)
    parser.add_argument("--mtp-gamma", type=int)
    parser.add_argument("--mtp-ubatch", type=int, default=128)
    parser.add_argument("--ratio", type=float, default=0.07)
    parser.add_argument("--ffn-oracle-sparsity", type=float, default=0.0)
    parser.add_argument("--target-ffn-oracle-sparsity", type=float, default=0.0)
    parser.add_argument("--ffn-oracle-block-size", type=int, default=1)
    parser.add_argument("--selection-layer", type=int)
    parser.add_argument("--anchor-tokens", type=int, default=0)
    parser.add_argument("--refresh-interval", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="turbo4")
    parser.add_argument("--draft-cache-type-k")
    parser.add_argument("--draft-cache-type-v")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_for(args, mode):
    draft_cache_type_k = args.draft_cache_type_k or args.cache_type_k
    draft_cache_type_v = args.draft_cache_type_v or args.cache_type_v
    gamma = args.gamma
    if mode == "vegas" and args.self_gamma is not None:
        gamma = args.self_gamma
    if mode in {"mtp", "mtp-vegas", "mtp-auto"} and args.mtp_gamma is not None:
        gamma = args.mtp_gamma

    command = [
        args.binary,
        "-m", args.model,
        "-f", args.prompt,
        "--vegas-prompt-tokens", str(args.prompt_tokens),
        "-n", str(args.predict),
        "-c", str(args.context),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-ngl", "99",
        "-fa", "on",
        "--cache-type-k", args.cache_type_k,
        "--cache-type-v", args.cache_type_v,
        "--temp", str(args.temperature),
        "--seed", str(args.seed),
        "--ignore-eos",
        "--vegas-quiet",
        "--vegas-mode", mode,
        "--vegas-target-ffn-oracle-sparsity", str(args.target_ffn_oracle_sparsity),
        "--vegas-ffn-oracle-block-size", str(args.ffn_oracle_block_size),
    ]
    if args.draft_model and mode in {"mtp", "mtp-vegas", "mtp-auto"}:
        command.extend(["-md", args.draft_model])
    if mode in {"mtp", "mtp-vegas", "mtp-auto"}:
        command.extend([
            "-ngld", "99",
            "--vegas-mtp-ubatch", str(args.mtp_ubatch),
            "--cache-type-k-draft", draft_cache_type_k,
            "--cache-type-v-draft", draft_cache_type_v,
            "--vegas-ffn-oracle-sparsity", str(args.ffn_oracle_sparsity),
        ])
    if mode != "baseline":
        command.extend(["--vegas-gamma", str(gamma)])
    if mode in {"vegas", "mtp-vegas"}:
        command.extend([
            "--vegas-ratio", str(args.ratio),
            "--vegas-min-tokens", str(args.min_tokens),
            "--vegas-anchor-tokens", str(args.anchor_tokens),
        ])
    if mode == "mtp-vegas" and args.selection_layer is not None:
        command.extend(["--vegas-selection-layer", str(args.selection_layer)])
    if mode == "mtp-vegas":
        command.extend(["--vegas-refresh-interval", str(args.refresh_interval)])
    return command


def run_one(args, mode, repetition, order):
    before = gpu_state()
    started = time.time()
    command = command_for(args, mode)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    after = gpu_state()

    marker = "VEGAS_RESULT "
    result_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if completed.returncode != 0 or result_line is None:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"run failed: mode={mode} repetition={repetition} "
            f"exit={completed.returncode}\n{tail}"
        )

    result = json.loads(result_line[len(marker):])
    result.update({
        "repetition": repetition,
        "order": order,
        "wall_seconds": elapsed,
        "gpu_before": before,
        "gpu_after": after,
        "temperature": args.temperature,
        "command": command,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "seed": args.seed,
        "prompt": args.prompt,
        "requested_mode": mode,
        "requested_gamma": (
            args.self_gamma if mode == "vegas" and args.self_gamma is not None else
            args.mtp_gamma if mode in {"mtp", "mtp-vegas", "mtp-auto"} and args.mtp_gamma is not None else
            args.gamma
        ),
        "requested_ratio": args.ratio,
        "requested_ffn_oracle_sparsity": args.ffn_oracle_sparsity,
        "requested_target_ffn_oracle_sparsity": args.target_ffn_oracle_sparsity,
        "requested_ffn_oracle_block_size": args.ffn_oracle_block_size,
        "requested_min_tokens": args.min_tokens,
        "requested_selection_layer": args.selection_layer,
        "requested_anchor_tokens": args.anchor_tokens,
        "requested_refresh_interval": args.refresh_interval,
        "requested_draft_cache_type_k": args.draft_cache_type_k or args.cache_type_k,
        "requested_draft_cache_type_v": args.draft_cache_type_v or args.cache_type_v,
    })
    return result


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(results):
    summary = {}
    for mode in sorted({result["mode"] for result in results}):
        values = [
            result["tokens_per_second"]
            for result in results
            if result["mode"] == mode
        ]
        summary[mode] = {
            "n": len(values),
            "mean_tps": statistics.fmean(values),
            "median_tps": statistics.median(values),
            "stdev_tps": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_tps": min(values),
            "max_tps": max(values),
        }

    by_key = {
        (result["repetition"], result["mode"]): result
        for result in results
    }
    repetitions = sorted({result["repetition"] for result in results})
    for mode in sorted({result["mode"] for result in results} - {"baseline"}):
        if not all((rep, "baseline") in by_key and (rep, mode) in by_key for rep in repetitions):
            continue
        ratios = [
            by_key[(rep, mode)]["tokens_per_second"] /
            by_key[(rep, "baseline")]["tokens_per_second"]
            for rep in repetitions
        ]
        rng = random.Random(0)
        bootstrap = []
        for _ in range(10000):
            sample = [rng.choice(ratios) for _ in ratios]
            bootstrap.append(statistics.fmean(sample))
        summary[f"paired_{mode}_over_baseline"] = {
            "ratios": ratios,
            "mean": statistics.fmean(ratios),
            "ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        }
    return summary


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    results = []
    for repetition in range(args.repetitions):
        modes = list(args.modes)
        if repetition % 2 == 1:
            modes.reverse()
        for order, mode in enumerate(modes):
            result = run_one(args, mode, repetition, order)
            results.append(result)
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            print("VEGAS_BENCH " + json.dumps(result, sort_keys=True), flush=True)

    print("VEGAS_SUMMARY " + json.dumps(summarize(results), sort_keys=True))


if __name__ == "__main__":
    main()
