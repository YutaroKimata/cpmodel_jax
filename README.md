# CPModel-JAX

Exact-hat counterfactuals for the multi-country, multi-sector model of
[Caliendo and Parro (2015)](https://doi.org/10.1093/restud/rdu035).
One Python module: JAX residuals, automatic `Jv`, and matrix-free Newton–GMRES.

## Install and use

Python 3.12+ is required. In another project:

```sh
uv add git+https://github.com/YutaroKimata/cpmodel_jax.git
```

```python
from cpmodel_jax import solve

result = solve(**inputs)
print(100 * (result["welfare_ratio"] - 1))
```

Inputs may be NumPy arrays, JAX arrays, or nested lists. Let `N` be countries
and `J` sectors. Bilateral axes are **importer, exporter, sector**, including
domestic flows; IO axes are **country, output sector, input sector**.

| Input | Shape | Meaning |
| --- | --- | --- |
| `theta` | `(J,)` | Positive sectoral trade elasticities |
| `alpha` | `(N,J)` | Final expenditure shares, summing to one |
| `beta` | `(N,J)` | Positive value-added shares |
| `gamma` | `(N,J,J)` | Intermediate shares in total production cost |
| `net_trade_value` | `(N,N,J)` | Baseline trade values excluding tariffs |
| `tariff_rates` | `(N,N,J)` | Baseline tariff rates; `0.1` means 10% |
| `counterfactual_tariff_rates` | `(N,N,J)` | Optional new tariff rates |
| `iceberg_cost_ratio` | `(N,N,J)` | Optional new/baseline iceberg costs |
| `technology_scale_ratio` | `(N,J)` | Optional new/baseline Fréchet scale parameters |

Omitted policy inputs mean no change; iceberg and technology ratios default to `1.0`.
Final demand shares and total production cost shares each sum to one, checked with
NumPy’s default `allclose` tolerances. Supply valid economic values in the listed shapes.
Zero bilateral flows are allowed; aggregate output, expenditure, value added,
and income must be positive. Tariffs are nonnegative, shock ratios positive,
and domestic iceberg ratios one.

The result is a dictionary of NumPy arrays. Ratios are counterfactual/baseline:
`wage_ratio`, `sector_price_ratio`, `unit_cost_ratio`, `trade_share_ratio`,
`expenditure_ratio`, `trade_value_ratio`, `net_trade_value_ratio`, `output_ratio`,
`income_ratio`, `consumer_price_ratio`, `welfare_ratio`, and `real_wage_ratio`.
Levels are `trade_shares`, `expenditure`, `trade_value`, `net_trade_value`,
`output`, and `income`. Monetary levels retain the input units.
`welfare_ratio` is household income divided by its consumption price index,
relative to baseline.

## Solver

`log_hats` concatenates log wage ratios (`N`), log price ratios (`N*J`),
and log expenditure ratios (`N*J`), in that order. All ratios are counterfactual/baseline.
The residual combines `N-1` labor equations, one nominal value-added condition,
price consistency, and expenditure balance.
Nominal baseline trade deficits stay fixed; baseline-value-added-weighted wages
average one through the explicit condition
`log(sum(value_added0 * wage_ratio) / sum(value_added0)) = 0`. Monetary values stay in the
input units throughout the calculation. Newton uses `jax.linearize` and GMRES with
backtracking, without a dense Jacobian, expenditure LU, or nested fixed-point iteration.

JAX computations use float32; NumPy input preparation uses float64. Numerical
options are `tolerance=1e-5`, `max_iter=60`, and `max_trials=25`. GMRES uses JAX defaults except
`tol=0.01` (avoid oversolving intermediate Newton systems) and
`solve_method="incremental"` (allow early termination within a restart).
Failed line searches, nonfinite residuals, and nonconvergence raise `RuntimeError`.
The final CP result also checks every labor equation, including the omitted one,
within `100*tolerance`, and requires finite outputs and positive household income.

A warm start is `initial_log_hats=previous_result["log_hats"]` with matching country/sector ordering.
`newton(f, data, z0)` is independent of CP and solves `f(z, data)=0`
for a float32 real vector, with options `tol=1e-5`, `max_iter=60`, and
`max_trials=25`. This last option counts trial step lengths per Newton update,
including the initial full step.
It returns `z`, `residual`, and `iterations`.

`solve` returns the economic outputs above plus `iterations`, `log_hats`, `residual`,
`diagnostics` (residual and labor errors), and `wall_seconds`. Timing includes
preparation, checks, and JIT compilation on first use.
CPU execution is validated; GPU execution and differentiation through `solve` are not.

## NAFTA example

From a checkout of this repository:

```sh
uv run examples/nafta_example.py
```

The script downloads the [authors' replication archive](https://faculty.som.yale.edu/lorenzocaliendo/estimates-of-the-trade-and-welfare-effects-of-nafta/)
to `.cache/`, checks its SHA-256, and uses their 31-country, 40-sector no-deficit
baseline. Only intra-NAFTA tariffs change from 1993 to 2005. Technology and
iceberg costs stay fixed. Use `--archive /path/to/Data_and_Codes_CP.zip` offline.

One negative conditional IO entry (Canada, output 11, input 20; `-0.0008459`)
is explicitly zeroed, then nonzero conditional IO vectors are normalized.
The adjusted baseline is not an exact unit-hat equilibrium. This reproduces
Table 2 at its published precision, not the original MATLAB calculation bit for bit.
All 12 entries are checked within 0.005 percentage points: total welfare is
1.3121% for Mexico, -0.0638% for Canada, and 0.0848% for the USA.

**Table 2 uses the authors' finite-change `Welfarelineal.m` decomposition.**
It differs from `100*(welfare_ratio-1)`, which gives about 0.0073%, -0.1101%,
and 0.0742%, respectively. The example prints both measures separately and
implements the decomposition in `paper_welfare`. Source data are downloaded,
not redistributed or covered by this repository's MIT license.

## Development

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m build
```

MIT license; see [LICENSE](LICENSE).
