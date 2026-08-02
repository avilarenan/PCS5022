#!/usr/bin/env python3
"""Microbenchmark correlation evidence against the current Gaussian-TE proxy."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from utility_peft.correlation import residual_correlation_features
from utility_peft.evidence import lagged_transfer_entropy


def _measure(function, repeats: int) -> float:
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=21)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--max-lag", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.channels, args.batch, args.length, args.max_lag, args.repeats) <= 0:
        parser.error("all numeric arguments must be positive")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(17)
    residual = torch.randn(
        args.batch,
        args.channels,
        args.length,
        generator=generator,
    )

    def correlation() -> None:
        residual_correlation_features(residual, max_lag=args.max_lag)

    def transfer_entropy() -> None:
        lagged_transfer_entropy(residual, lag=1)

    correlation()
    transfer_entropy()
    correlation_seconds = _measure(correlation, args.repeats)
    transfer_entropy_seconds = _measure(transfer_entropy, args.repeats)
    payload = {
        "batch": args.batch,
        "channels": args.channels,
        "length": args.length,
        "correlation_max_lag": args.max_lag,
        "correlation_suite_median_s": correlation_seconds,
        "gaussian_te_lag1_median_s": transfer_entropy_seconds,
        "te_over_correlation_speed_ratio": transfer_entropy_seconds
        / max(correlation_seconds, 1e-12),
        "note": (
            "Conservative implementation benchmark: the correlation suite evaluates all "
            "requested lags, while the TE proxy evaluates lag 1 only. Both are quadratic in "
            "channel count here; correlation uses batched tensor operations and TE performs "
            "ordered-pair regression solves."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
