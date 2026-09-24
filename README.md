# CPModel-JAX

Exact-hat counterfactuals for the multi-country, multi-sector model of
[Caliendo and Parro (2015)](https://doi.org/10.1093/restud/rdu035).
Solves equilibrium equations using JAX automatic differentiation and Newton's method,
without explicitly constructing the Jacobian matrix.
Nominal trade deficits are fixed at baseline levels; world nominal value added
is the numeraire.

## Install and use

Python 3.12+ is required. In your project:

```sh
uv add git+https://github.com/YutaroKimata/cpmodel_jax.git
```

```python
from cpmodel_jax import solve

result = solve(**inputs)
print(100 * (result["welfare_ratio"] - 1))
```

## Inputs

Economic inputs accept NumPy arrays, JAX arrays, or nested lists.
`N` is the number of countries and `J` the number of sectors.
Bilateral axes are **importer, exporter, sector**, including domestic flows;
IO axes are **country, output sector, input sector**.

| Key | Shape | Description |
| --- | --- | --- |
| `theta` | `(J,)` | Positive sectoral trade elasticities |
| `alpha` | `(N,J)` | Final expenditure shares, summing to one per country |
| `beta` | `(N,J)` | Value-added shares in total production cost |
| `gamma` | `(N,J,J)` | Intermediate cost shares; `beta + gamma.sum(2) = 1` |
| `baseline_net_trade_value` | `(N,N,J)` | Baseline trade values excluding tariffs |
| `baseline_tariff_rates` | `(N,N,J)` | Baseline tariff rates; `0.1` means 10% |
| `counterfactual_tariff_rates` | `(N,N,J)` | Counterfactual tariff rates; default: baseline rates |
| `iceberg_cost_ratio` | `(N,N,J)` or scalar | Counterfactual/baseline iceberg costs; default `1.0` |
| `technology_scale_ratio` | `(N,J)` or scalar | Counterfactual/baseline Fréchet scale parameters; default `1.0` |

## Outputs

`solve` returns a dictionary. Economic outputs are NumPy arrays with the same axis
conventions as the inputs. Ratios are counterfactual/baseline; monetary levels
retain the input units.

| Key | Shape | Description |
| --- | --- | --- |
| `wage_ratio` | `(N,)` | Wage ratio |
| `sector_price_ratio` | `(N,J)` | Sectoral price index ratio |
| `unit_cost_ratio` | `(N,J)` | Unit production cost ratio |
| `trade_share_ratio` | `(N,N,J)` | Bilateral expenditure share ratio |
| `expenditure_ratio` | `(N,J)` | Sectoral expenditure ratio |
| `trade_value_ratio` | `(N,N,J)` | Trade value ratio, including tariffs |
| `net_trade_value_ratio` | `(N,N,J)` | Trade value ratio, excluding tariffs |
| `output_ratio` | `(N,J)` | Gross output value ratio |
| `income_ratio` | `(N,)` | Household income ratio |
| `consumer_price_ratio` | `(N,)` | Consumption price index ratio |
| `welfare_ratio` | `(N,)` | Real household income ratio |
| `real_wage_ratio` | `(N,)` | Wage divided by the consumption price index, relative to baseline |
| `counterfactual_trade_shares` | `(N,N,J)` | Counterfactual bilateral expenditure shares |
| `counterfactual_expenditure` | `(N,J)` | Counterfactual sectoral expenditure |
| `counterfactual_trade_value` | `(N,N,J)` | Counterfactual trade values including tariffs |
| `counterfactual_net_trade_value` | `(N,N,J)` | Counterfactual trade values excluding tariffs |
| `counterfactual_output` | `(N,J)` | Counterfactual gross output value |
| `counterfactual_income` | `(N,)` | Counterfactual household income |
| `iterations` | scalar | Number of Newton iterations |
| `log_ratios` | `(N+2*N*J,)` | Log wage, sectoral price, and expenditure ratios, packed for warm starts |
| `residual` | `(N+2*N*J,)` | Equilibrium residuals |
| `diagnostics` | — | Dictionary of maximum residual and relative labor-market errors |
| `wall_seconds` | scalar | Solve time, including JIT compilation on first use |

## NAFTA example

Reproduce the welfare decomposition and real wages in Table 2 of
[Caliendo and Parro (2015)](https://doi.org/10.1093/restud/rdu035).
From a checkout of this repository:

```sh
uv run examples/nafta_example.py
```

The script downloads the authors' data automatically and checks the results
against Table 2 at its published precision.

## Development

```sh
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m build
```

MIT license; see [LICENSE](LICENSE).
