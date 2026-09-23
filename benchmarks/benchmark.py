"""Measure setup, first-call compilation, and synchronized warm toy solves."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from cpmodel_jax import EXAMPLE_NAMES, __version__, load_example, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=EXAMPLE_NAMES, default="full")
    parser.add_argument("--formulation", choices=("augmented", "cost_output"), default="augmented")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    start = perf_counter()
    economy = load_example(args.case, formulation=args.formulation)
    jax.block_until_ready(economy.data)
    setup_seconds = perf_counter() - start
    first = solve(economy)
    times = []
    for _ in range(args.repeats):
        # Deliberately start from the same default state, not the previous root.
        answer = solve(economy)
        times.append(answer.wall_seconds)
    report = dict(
        package_version=__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        jax=jax.__version__,
        numpy=np.__version__,
        devices=[str(device) for device in jax.devices()],
        float64=bool(jax.config.x64_enabled),
        case=args.case,
        formulation=args.formulation,
        unknowns=economy.size,
        setup_seconds=setup_seconds,
        first_call_seconds=first.wall_seconds,
        warm_seconds=times,
        warm_median_seconds=float(np.median(times)),
        newton_steps=int(answer.root.steps),
        diagnostics=answer.diagnostics,
        notes="First call includes compilation. Warm runs use no solution warm start.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
