# API

All implementation code is in `src/cpmodel_jax.py`.

```python
from cpmodel_jax import load_example, solve

inputs = load_example("full", countries=4, sectors=3)
result = solve(**inputs, tolerance=1e-10)
print(result["welfare_ratio"])
```

`solve` takes six required keyword arguments: `trade_elasticities`,
`final_demand_shares`, `value_added_shares`, `input_output_shares`,
`net_trade_value`, and `tariff_rates`. See the README for shapes and units.
Optional `counterfactual_tariff_rates`, `iceberg_cost_ratio`, and
`technology_scale_ratio` default to no change. Inputs are validated and never
modified, reconciled, or clipped. Share sums must agree within `1e-10`.

The result is a plain dictionary. It contains the same 18 economic outputs as
version 0.4, plus `diagnostics`, `iterations`, `z`, `residual`, `history`, and
`wall_seconds`. Monetary levels are in the input units. Ratios are new/baseline;
`100 * (result["welfare_ratio"] - 1)` gives percentage changes.

For a warm start, pass `z0=previous_result["z"]` for the same country/sector
ordering. `z` holds relative log wages, log prices, and log expenditure ratios.
Changing the model or its dimensions requires a suitable new initial state.

## Numerical options

`solve` forwards numerical keywords to `newton`:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `tolerance` | `1e-10` | Maximum absolute scaled nonlinear residual |
| `max_steps` | `60` | Maximum Newton updates |
| `restart` | `60` | GMRES restart length, capped by unknown count |
| `max_cycles` | `30` | Maximum GMRES restart cycles |
| `max_backtracks` | `25` | Trial lengths per line search |
| `max_log_step` | `1.0` | Maximum absolute coordinate update before backtracking |
| `armijo` | `1e-4` | Required decrease in residual norm |

Economic checks use `max(1e-8, 100*tolerance)` in addition to the nonlinear
residual criterion. Invalid inputs raise `ValueError`; failed convergence or
economic verification raises `RuntimeError`.

`newton(function, data, z0, **options)` also works independently of the CP model.
`function(z, data)` must be a pure JAX-compatible residual with the same vector
shape as `z`. It returns a dictionary containing `z`, `residual`, `steps`, and
`history`. No CP equations are used by this function.

History has one row per completed update and columns listed in `HISTORY_COLUMNS`:
`residual_inf`, `residual_l2`, `forcing_eta`, `linear_relative_residual`,
`step_length`, `backtracks`, `next_residual_inf`, `gmres_info`.

Import enables JAX float64. `solve` is an ordinary host-side Python function;
wrapping it in `jax.grad` is not an implicit equilibrium differentiation API.

## Changes from 0.4

- `load_example` returns an input dictionary; use `solve(**load_example())`.
- Pass custom arrays directly to `solve`; `Calibration` and economy objects were removed.
- Pass numerical options as keywords; `NewtonOptions` was removed.
- Read `result["welfare_ratio"]`, not `result.values["welfare_ratio"]`.
- Read `result["iterations"]`, not `result.root.steps`.
- Only the wage/price/expenditure formulation remains; there is no `formulation` option.
- Catch `RuntimeError` for numerical failures; `SolveError` and status codes were removed.
- `wall_seconds` now includes preparation, previously done by the economy constructor.
