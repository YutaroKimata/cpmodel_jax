import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from cpmodel_jax import EXAMPLE_NAMES, __version__, load_example, solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=EXAMPLE_NAMES, default="full")
    parser.add_argument("--countries", type=int, default=4)
    parser.add_argument("--sectors", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    start = perf_counter()
    inputs = load_example(args.case, countries=args.countries, sectors=args.sectors)
    setup_seconds = perf_counter() - start
    first = solve(**inputs)
    first_seconds = first["wall_seconds"]
    del first
    times = []
    for _ in range(args.repeats):
        answer = solve(**inputs)
        times.append(answer["wall_seconds"])
        diagnostics, steps = answer["diagnostics"], answer["iterations"]
        del answer
    report = {
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
        "devices": [str(device) for device in jax.devices()],
        "float64": bool(jax.config.x64_enabled),
        "case": args.case,
        "countries": args.countries,
        "sectors": args.sectors,
        "unknowns": args.countries - 1 + 2 * args.countries * args.sectors,
        "setup_seconds": setup_seconds,
        "first_call_seconds": first_seconds,
        "warm_seconds": times,
        "warm_median_seconds": float(np.median(times)),
        "newton_steps": steps,
        "diagnostics": diagnostics,
        "notes": "Solve includes input preparation and economic checks. First call includes JIT. "
        "Warm runs start from default coordinates, not the previous solution.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
