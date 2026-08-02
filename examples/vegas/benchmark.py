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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--predict", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=["baseline", "vegas"])
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.07)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="turbo4")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_for(args, mode):
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
    ]
    if mode != "baseline":
        command.extend(["--vegas-gamma", str(args.gamma)])
    if mode == "vegas":
        command.extend([
            "--vegas-ratio", str(args.ratio),
            "--vegas-min-tokens", str(args.min_tokens),
        ])
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
        tail = "\n".join(completed.stdout.splitlines()[-20:])
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
        "requested_gamma": args.gamma,
        "requested_ratio": args.ratio,
        "requested_min_tokens": args.min_tokens,
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
    if all((rep, "baseline") in by_key and (rep, "vegas") in by_key for rep in repetitions):
        ratios = [
            by_key[(rep, "vegas")]["tokens_per_second"] /
            by_key[(rep, "baseline")]["tokens_per_second"]
            for rep in repetitions
        ]
        rng = random.Random(0)
        bootstrap = []
        for _ in range(10000):
            sample = [rng.choice(ratios) for _ in ratios]
            bootstrap.append(statistics.fmean(sample))
        summary["paired_vegas_over_baseline"] = {
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
