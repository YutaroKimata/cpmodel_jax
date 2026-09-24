# Changelog

## 0.5.0

- Consolidated the implementation into `src/cpmodel_jax.py`.
- Replaced model/options/result classes with functions, keyword arguments, and dictionaries.
- Kept only the wage/price/expenditure equilibrium formulation.
- Replaced compiled control-flow loops with Python Newton/backtracking loops;
  residual evaluation and matrix-free GMRES remain JIT compiled.
- Preserved economic output names, closure, float64 precision, and convergence checks.
- Changed `wall_seconds` to include input preparation. See `docs/api.md` for migration.

## 0.4.0 — 2026-09-23

- Rename all six calibration fields and three policy arguments to descriptive
  economic names; update their stored attributes and validation messages.
- Distinguish `tariff_rates` from `counterfactual_tariff_rates`, and use
  `iceberg_cost_ratio` and `technology_scale_ratio` for multiplicative shocks.
- Expose dimensions as `num_countries` and `num_sectors`.
- Update examples, documentation, tests, and internal coefficient names together.
- Input keywords are a breaking change; output names, data, and equations stay
  unchanged. Old symbolic input aliases are not retained.

## 0.3.0 — 2026-09-23

- Present the package directly as an implementation of the Caliendo–Parro model.
- Generate reproducible synthetic baselines and shocks entirely in Python.
- Replace stored comparison fixtures with analytic equilibrium tests and
  comparisons between the augmented and cost/output formulations.
- Remove external-model export scripts and data from the repository and releases.
- Document the exact synthetic baseline construction and standalone validation.
- Example numerical outputs change because the synthetic dataset changed;
  the equilibrium equations are unchanged.
- Honor both relative and absolute GMRES tolerances in the independent linear
  residual check, preventing false failures near nonlinear convergence.

## 0.2.0 — 2026-09-23

- Rename all 18 output keys to descriptive economic names, including
  `welfare_ratio`, `wage_ratio`, `sector_price_ratio`, and `trade_shares`.
- Use `_ratio` consistently for counterfactual-to-baseline multipliers and
  distinguish tariff-inclusive `trade_value` from `net_trade_value`.
- Rename `economy.pack` keyword arguments to match the output names.
- Document migration from 0.1, units, percentage conversion, and welfare versus
  real wages. Old result keys are removed.
- Preserve model equations while renaming the public outputs.

## 0.1.0 — 2026-09-23

- Standalone `src/`-layout Python distribution of the equation-based CP solver.
- Public calibration, counterfactual, solver options, outcomes, and error API.
- Compiled float64 matrix-free Newton–GMRES with augmented and cost/output formulations.
- Six toy examples and numerical validation tests.
- Installation examples, model/API documentation, benchmark script, and MIT license.
- GitHub Actions for lint, tests, distributions, and installed-wheel validation.

Initial local release preparation; GitHub and PyPI publication are separate.
