# Local verification — version 0.5.0

Validated on 2026-09-24 with macOS arm64, Python 3.12.14, JAX/jaxlib 0.11.2,
NumPy 2.5.3, CPU, float64.

The refactor uses one implementation file, plain functions/dictionaries, and
Python loops for Newton and backtracking. Economic equations, output names,
closure, and tolerances are preserved.

- Analytic no-shock and one-country equilibria; all six shock examples.
- Jv checked against finite differences; accounting, units, country relabeling,
  zero trade links, unchanged inputs, warm starts, and failure handling.
- Generic quadratic root; successful backtracking and line-search failure.
- All 18 economic outputs checked against both 0.4 formulations for six small
  cases and a 16×10 case (14 comparisons), with `rtol=1e-8`, `atol=1e-10`.
- Standard formulation agreement is within floating-point rounding; the largest
  difference from the alternate reference is below `1e-10` after scaling by
  `max(1, abs(reference))`.

A local comparison on the same M4 Pro CPU timed input preparation, solving,
outcome extraction, and economic checks. Each warm call used the default initial
coordinates. Median of ten warm calls, excluding compilation:

| Countries × sectors | 0.4 | 0.5 |
| --- | ---: | ---: |
| 4 × 3 | 0.60 ms | 0.91 ms |
| 100 × 50 | 32.69 ms | 32.06 ms |

This small timing sample does not establish a speedup. Python control flow adds
about 0.3 ms on the tiny case; the larger case is similar in total runtime.
Old `wall_seconds` excluded preparation, whereas 0.5 includes it; the comparison
above uses externally timed complete calls on both versions.

No new GPU or 1000-country measurement was performed for this refactor.

Packaging checks completed:

- Ruff lint and formatting passed.
- Installed 0.5 wheel tested outside the source tree: 15 tests passed.
- Both quickstart and custom-input examples ran against the installed wheel.
- Wheel and source distribution built successfully; strict metadata checks passed.

GitHub Actions has not been run remotely. No package was published.
