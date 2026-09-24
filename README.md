# CPModel-JAX

CPModel-JAX is a Python package for computing exact-hat counterfactual equilibria in the multi-country, multi-sector trade model of Caliendo and Parro (2015).

Given baseline trade flows, input-output structure, final expenditure, tariffs, and trade elasticities, it computes counterfactual equilibria under changes in tariffs, iceberg trade costs, and technology. Equilibria are solved in JAX with a Newton–Krylov method.

## Installation

CPModel-JAX requires Python 3.12 or later.

```sh
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install .
```

Run the included example with:

```sh
python examples/quickstart.py
```

## Quick start

```python
from cpmodel_jax import load_example, solve

inputs = load_example("full")
result = solve(**inputs, tolerance=1e-10)

print(result["welfare_ratio"])
print(result["diagnostics"])
```

Available examples are `no_shock`, `tariff`, `iceberg`, `combined`, `technology`, and `full`. They use reproducibly generated synthetic data.

## Using your own data

Pass model inputs directly to `solve`:

```python
from cpmodel_jax import solve

result = solve(
    trade_elasticities=trade_elasticities,
    final_demand_shares=final_demand_shares,
    value_added_shares=value_added_shares,
    input_output_shares=input_output_shares,
    net_trade_value=net_trade_value,
    tariff_rates=tariff_rates,
    counterfactual_tariff_rates=counterfactual_tariff_rates,
    iceberg_cost_ratio=iceberg_cost_ratio,
    technology_scale_ratio=technology_scale_ratio,
)
```

Inputs can also be stored in a dictionary and passed with `solve(**inputs)`. Omitted policy arguments remain at their baseline values.

See [the complete custom-input example](examples/custom_inputs.py).

## Inputs

Let `N` denote the number of countries and `J` the number of sectors.

| Input | Shape | Description |
| --- | --- | --- |
| `trade_elasticities` | `(J,)` | Sectoral trade elasticities |
| `final_demand_shares` | `(N, J)` | Final expenditure shares |
| `value_added_shares` | `(N, J)` | Value-added shares |
| `input_output_shares` | `(N, J, J)` | Intermediate input shares |
| `net_trade_value` | `(N, N, J)` | Baseline bilateral trade values, net of tariffs |
| `tariff_rates` | `(N, N, J)` | Baseline tariff rates |
| `counterfactual_tariff_rates` | `(N, N, J)` | Counterfactual tariff rates |
| `iceberg_cost_ratio` | `(N, N, J)` | Counterfactual-to-baseline iceberg cost ratios |
| `technology_scale_ratio` | `(N, J)` | Counterfactual-to-baseline technology-scale ratios |

Bilateral arrays are ordered as importer, exporter, sector and include domestic flows. Tariffs are rates, so `0.10` means 10%. For `iceberg_cost_ratio` and `technology_scale_ratio`, `1.0` denotes no change.

## Results

Equilibrium outcomes are available directly in the result dictionary.

```python
result["wage_ratio"]
result["sector_price_ratio"]
result["trade_value_ratio"]
result["output_ratio"]
result["income_ratio"]
result["welfare_ratio"]
```

Variables ending in `_ratio` are counterfactual-to-baseline ratios. A value of `1.05` represents a 5% increase.

Level outputs are also available:

```python
result["trade_shares"]
result["expenditure"]
result["trade_value"]
result["net_trade_value"]
result["output"]
result["income"]
```

Solver diagnostics, iteration counts, and elapsed time can be inspected with:

```python
print(result["diagnostics"])
print(result["iterations"])
print(result["wall_seconds"])
```

The implementation is in [`src/cpmodel_jax.py`](src/cpmodel_jax.py): ordinary functions and dictionaries, with Python loops for Newton and backtracking. JAX compiles residual evaluation and the matrix-free GMRES step.

Version 0.5 replaces the class-based API. See the [migration notes](docs/api.md).

## Documentation

See:

- [API guide](docs/api.md) for the public API and result objects.
- [Equations and solver](docs/method.md) for the economic model and numerical method.
- [Validation](docs/validation.md) for tests and numerical validation.

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest
python benchmarks/benchmark.py --repeats 10 --output benchmark-results.json
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.

## Reference

Caliendo, L. and Parro, F. (2015), “Estimates of the Trade and Welfare Effects of NAFTA,” *Review of Economic Studies*, 82(1), 1–44.
https://doi.org/10.1093/restud/rdu035

Citation metadata is available in [CITATION.cff](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE).
