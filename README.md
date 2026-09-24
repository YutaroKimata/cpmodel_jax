# CPModel-JAX

Exact-hat counterfactuals for the multi-country, multi-sector model of
[Caliendo and Parro (2015)](https://doi.org/10.1093/restud/rdu035).
Solves equilibrium equations using JAX automatic differentiation and Newton's method,
without explicitly constructing the Jacobian matrix.

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
For each country and sector, `beta + gamma.sum(2)` must equal one.
Zero bilateral flows are allowed; aggregate output, expenditure, value added,
and income must be positive. Tariffs are nonnegative, shock ratios positive,
and domestic iceberg ratios one.

Nominal trade deficits stay fixed at baseline levels. World nominal value added
is held constant as the price normalization.

## Outputs

The result is a dictionary of NumPy arrays. Ratios are counterfactual/baseline:
`wage_ratio`, `sector_price_ratio`, `unit_cost_ratio`, `trade_share_ratio`,
`expenditure_ratio`, `trade_value_ratio`, `net_trade_value_ratio`, `output_ratio`,
`income_ratio`, `consumer_price_ratio`, `welfare_ratio`, and `real_wage_ratio`.
Levels are `trade_shares`, `expenditure`, `trade_value`, `net_trade_value`,
`output`, and `income`. Monetary levels retain the input units.
`welfare_ratio` is household income divided by its consumption price index,
relative to baseline.

The result also includes `iterations`, `diagnostics`, and `wall_seconds`.
Timing includes JIT compilation on first use.

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
