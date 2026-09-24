# Standalone validation

All test and example inputs are generated in Python. The package contains the
model equations, numerical solver, example generator, and validation tests;
there is no external model checkout or stored reference solution to load.

## Analytically balanced synthetic baseline

The example generator chooses positive national value added `v[n]`, final
expenditure shares `a[j]` summing to one, value-added shares `b[j]` between zero
and one, and sector-specific openness `o[j]`. Countries share the sectoral
coefficients while differing in size. Input-output shares are

```
input_output_shares[n,j,k] = (1 - b[j]) * a[k]
B = sum_j a[j] * b[j]
net_trade_value[n,i,j] = a[j]/B * ((1-o[j]) * v[n] * indicator(n=i)
                    + o[j] * v[n] * v[i] / sum(v))
```

Baseline tariffs are zero. Both country margins of bilateral flows are
`v[n] * a[j] / B`; consequently trade is balanced and output equals sector
expenditure. Value added is exactly `v[n]`. Intermediate expenditure is
`v[n] * a[j] * (1-B)/B`, so final plus intermediate demand equals sector
expenditure. This construction requires no equilibrium solve or matrix
factorization. Unit hats are therefore a known no-shock equilibrium.

`load_example` produces fresh NumPy data from a fixed seed. `tariff`, `iceberg`,
`technology`, `combined`, and `full` apply small policy or technology changes;
`no_shock` preserves the analytically balanced baseline. The default is 4
countries and 3 sectors. Tests also use a 16-country, 10-sector synthetic case.
These examples are not estimated economies or empirical evidence.

## Independent checks

- Analytic no-shock results: all ratios equal one and flows/income reproduce
  the constructed baseline, for two model sizes.
- One-country technology shock: with trade elasticity 4 and no intermediate
  inputs, welfare equals the fourth root of the technology-distribution ratio.
- During the 0.5 refactor, all 18 outputs were compared against both previous
  formulations across six scenarios and a 16×10 case. Those comparisons were
  migration checks; the alternative implementation is no longer shipped.
- Off-equilibrium accounting identities, full market clearing, national income,
  expenditure, share normalization, and the numeraire.
- Automatic directional derivatives compared with centered finite differences.
- Money-unit scaling, country relabeling, zero trade links, explicit warm starts,
  unchanged inputs, invalid data, and numerical failure reporting.
- A model-independent quadratic root and backtracking success/failure checks.

## Reproduce

```sh
python -m pip install -e '.[dev]' -c requirements-validation.txt
python -m pytest -q
python benchmarks/benchmark.py --case full --repeats 10 --output benchmark-results.json
```

The constraints record the validated numerical stack. GitHub Actions is
configured for Linux and macOS CPU; local tests do not establish remote CI
results. Tests are also run against an installed wheel outside the source tree.

The benchmark records versions, device, precision, nonlinear residuals,
Newton steps, setup, first-call latency, and all warm samples. Each solve
includes input preparation, synchronizes, and checks economic equations. First-call time includes compilation
but is not a process-startup measurement. Warm calls start from the default
state, not the previous solution. No Krylov iteration counts are inferred from
Newton steps.
