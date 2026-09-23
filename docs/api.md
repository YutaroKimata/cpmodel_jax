# Public API

```python
from cpmodel_jax import (
    Calibration,
    NewtonOptions,
    Equilibrium,
    SolveError,
    AugmentedEconomy,
    CostOutputEconomy,
    load_example,
    solve,
)
```

## Calibration and counterfactuals

`Calibration` copies and validates its six baseline inputs:

```python
baseline = Calibration(
    trade_elasticities=trade_elasticities,
    final_demand_shares=final_demand_shares,
    value_added_shares=value_added_shares,
    input_output_shares=input_output_shares,
    net_trade_value=net_trade_value,
    tariff_rates=tariff_rates,
)
economy = baseline.economy(
    counterfactual_tariff_rates=counterfactual_tariff_rates,
    iceberg_cost_ratio=iceberg_cost_ratio,
    technology_scale_ratio=technology_scale_ratio,
    formulation="augmented",
)
```

The same descriptive names are used for stored calibration attributes. Baseline
arrays are read-only. `tariff_rates` is the baseline rate and
`counterfactual_tariff_rates` is the new rate, both in decimal units. Cost and
technology shocks are ratios. The latter scales the Fréchet distribution
parameter. Array axes and restrictions are in the README.

All three policy arguments default to `None` (unchanged). Preparing an economy
does not solve it. The alternative formulation is `"cost_output"`.
`economy.num_countries` and `economy.num_sectors` expose its dimensions.

Version 0.4 replaces the previous symbolic constructor and policy names; use
these descriptive keywords for both formulations. Output names are unchanged.

`load_example(name="full", *, formulation="augmented", countries=4, sectors=3, seed=20260923)` constructs a fresh
synthetic economy locally. Its baseline is analytically balanced. Valid names
are exposed as `EXAMPLE_NAMES`; a fixed seed reproduces the same inputs.

## Solving

`solve(economy, options=NewtonOptions(), *, z0=None) -> Equilibrium`

Default options:

| Option | Default | Meaning |
| --- | --- | --- |
| `tolerance` | `1e-10` | Infinity norm of scaled nonlinear residual |
| `max_steps` | `60` | Maximum Newton iterations |
| `restart` | `60` | GMRES restart length, capped by unknown count |
| `max_cycles` | `30` | Maximum GMRES restart cycles per Newton step |
| `max_backtracks` | `25` | Maximum trial lengths per line search |
| `max_log_step` | `1.0` | Maximum absolute coordinate step before backtracking |
| `armijo` | `1e-4` | Required residual decrease parameter |

Final economic checks use `max(1e-8, 100*tolerance)` while the scaled nonlinear
residual must meet `tolerance`. These are different tests. Invalid inputs raise
`ValueError`; numerical failure raises `SolveError` with a `.root` attribute.

For an explicit warm start in the default formulation:

```python
first = solve(economy)
z0 = economy.pack(
    wage_ratio=first.values["wage_ratio"],
    sector_price_ratio=first.values["sector_price_ratio"],
    expenditure=first.values["expenditure"],
)
second = solve(economy, z0=z0)
```

For `cost_output`, use `economy.pack(unit_cost_ratio, output)`. Packers require the
numeraire and use levels in the original monetary units. Coordinates differ
between formulations and must not be exchanged.

## Result and convergence history

- `values`: NumPy arrays with descriptive keys; definitions are in the README. `_ratio` means counterfactual divided by baseline, so
  `100 * (values["welfare_ratio"] - 1)` gives the welfare percentage change.
- `diagnostics`: scalar errors for residuals, prices, costs, goods/labor
  markets, expenditure, income, share sums, and numeraire.
- `root`: JAX arrays `z`, `residual`, `steps`, `status`, and `history`.
- `wall_seconds`: synchronized solve, outcomes, and economic checks. It excludes
  economy construction and includes compilation when needed.

History columns are available as `cpmodel_jax.newton.HISTORY_COLUMNS`:
`residual_inf`, `residual_l2`, `forcing_eta`, `linear_relative_residual`,
`step_length`, `backtracks`, `next_residual_inf`, `gmres_info`.
These are nonlinear iteration records, not counts of Krylov iterations.

Root status codes: `0` converged, `1` iteration limit, `2` nonfinite initial
residual, `3` linear solve failure, `4` line search failure, `5` failed economic
verification. Only status zero is returned successfully by `solve`.

## Low-level numerical interface

`cpmodel_jax.newton.newton(function, data, z0, options)` accepts a pure residual
`function(z, data)` with the same vector shape as `z`. It returns a `Root` and
does not raise on numerical failure; call `require_converged(root)` explicitly.
This advanced module is model-independent but is not a stable public API yet.
Likewise, internal `model` and `augmented` helpers may change.

Changing shapes or static solver options can trigger JIT compilation. Same-shape
policy arrays are dynamic inputs. Import enables JAX float64 globally; changing
that setting afterward is unsupported. The host-facing `solve` cannot simply
be wrapped in `jax.grad`: an implicit differentiation interface is future work.
