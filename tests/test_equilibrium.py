"""Analytic equilibria and independent economic and numerical properties."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpmodel_jax import Calibration, NewtonOptions, SolveError, solve
from cpmodel_jax.model import state
from cpmodel_jax.newton import newton, require_converged

from .conftest import example_inputs, prepare


@pytest.mark.parametrize(
    "name", ["no_shock", "tariff", "iceberg", "combined", "technology", "full", "synthetic_16x10"]
)
def test_formulations_agree(name):
    inputs = (
        example_inputs("full", countries=16, sectors=10)
        if name == "synthetic_16x10"
        else example_inputs(name)
    )
    results = []
    for formulation in ("augmented", "cost_output"):
        economy = prepare(inputs, formulation)
        result = solve(economy)
        assert max(result.diagnostics.values()) < 1e-8
        keys = (
            ("unit_cost_ratio", "output")
            if formulation == "cost_output"
            else ("wage_ratio", "sector_price_ratio", "expenditure")
        )
        packed = economy.pack(**{key: result.values[key] for key in keys})
        np.testing.assert_allclose(packed, result.root.z, atol=1e-13)
        assert int(solve(economy, z0=packed).root.steps) == 0
        results.append(result.values)
    for key in results[0]:
        np.testing.assert_allclose(
            results[0][key], results[1][key], rtol=1e-8, atol=1e-10, err_msg=key
        )


@pytest.mark.parametrize("formulation", ["augmented", "cost_output"])
@pytest.mark.parametrize("countries,sectors", [(4, 3), (16, 10)])
def test_analytic_no_shock(formulation, countries, sectors):
    inputs = example_inputs("no_shock", countries=countries, sectors=sectors)
    result = solve(prepare(inputs, formulation))
    for key, value in result.values.items():
        if key.endswith("_ratio"):
            np.testing.assert_allclose(value, 1, atol=1e-10, rtol=0, err_msg=key)
    trade = inputs["net_trade_value"]
    np.testing.assert_allclose(result.values["net_trade_value"], trade, rtol=1e-10)
    np.testing.assert_allclose(result.values["expenditure"], trade.sum(1), rtol=1e-10)
    np.testing.assert_allclose(result.values["output"], trade.sum(0), rtol=1e-10)
    np.testing.assert_allclose(
        result.values["income"], (inputs["value_added_shares"] * trade.sum(0)).sum(1), rtol=1e-10
    )


def test_elimination_away_from_equilibrium_and_jvp():
    m = prepare(example_inputs("full"))
    rng = np.random.default_rng(741)
    z = m.initial_state() + 0.02 * rng.normal(size=m.size)
    s = state(z, m.data)
    d = m.data
    net_trade_value = np.asarray(s.shares * s.expenditure[:, None, :] / (1 + d.tariff))
    R = (np.asarray(d.tariff) * net_trade_value).sum((1, 2))
    np.testing.assert_allclose(
        s.income, np.asarray(s.wage * d.value_added + d.deficit) + R, rtol=1e-13
    )
    # The redundant market follows the global accounting identity, even off root.
    np.testing.assert_allclose(np.asarray(s.output).sum(), net_trade_value.sum(), rtol=1e-13)
    np.testing.assert_allclose(np.sum(np.asarray(d.value_added_shares * s.output)), 1, atol=1e-14)
    _, Jv = jax.linearize(m.residual, z)
    v = rng.normal(size=m.size)
    v /= np.linalg.norm(v)
    fd = (np.asarray(m.residual(z + 1e-5 * v)) - np.asarray(m.residual(z - 1e-5 * v))) / 2e-5
    assert np.linalg.norm(np.asarray(Jv(v)) - fd) / np.linalg.norm(fd) < 1e-7


def test_units_and_country_permutation():
    a = example_inputs("full")
    reference = solve(prepare(a)).values
    scaled = solve(prepare({**a, "net_trade_value": a["net_trade_value"] * 1e9})).values
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "trade_shares"):
        np.testing.assert_allclose(scaled[key], reference[key], rtol=1e-8, atol=1e-10)
    permutation = np.array([2, 0, 3, 1])
    permuted = {
        k: (
            v[permutation][:, permutation, :]
            if k
            in (
                "net_trade_value",
                "tariff_rates",
                "counterfactual_tariff_rates",
                "iceberg_cost_ratio",
            )
            else v[permutation]
            if k
            in (
                "final_demand_shares",
                "value_added_shares",
                "input_output_shares",
                "technology_scale_ratio",
            )
            else v
        )
        for k, v in a.items()
    }
    result = solve(prepare(permuted)).values
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "output"):
        np.testing.assert_allclose(result[key], reference[key][permutation], rtol=1e-8, atol=1e-10)


def test_analytic_one_country_and_zero_links():
    one = Calibration(
        trade_elasticities=[4.0],
        final_demand_shares=[[1.0]],
        value_added_shares=[[1.0]],
        input_output_shares=[[[0.0]]],
        net_trade_value=[[[2.0]]],
        tariff_rates=[[[0.0]]],
    )
    r = solve(one.economy(technology_scale_ratio=[[1.21]]))
    np.testing.assert_allclose(r.values["wage_ratio"], 1, atol=1e-14)
    np.testing.assert_allclose(r.values["welfare_ratio"], 1.21**0.25, atol=1e-12)
    net_trade_value = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])[:, :, None]
    sparse = Calibration(
        trade_elasticities=[4.0],
        final_demand_shares=np.ones((3, 1)),
        value_added_shares=np.ones((3, 1)),
        input_output_shares=np.zeros((3, 1, 1)),
        net_trade_value=net_trade_value,
        tariff_rates=np.zeros_like(net_trade_value),
    )
    r = solve(sparse.economy(technology_scale_ratio=[[1.1], [1.0], [0.9]]))
    assert np.all(r.values["net_trade_value"][net_trade_value == 0] == 0)
    assert max(r.diagnostics.values()) < 1e-8


def test_statuses_validation_and_immutable_calibration():
    a = example_inputs("full")
    m = prepare(a)
    with pytest.raises(SolveError, match="iteration limit"):
        solve(m, NewtonOptions(max_steps=1))
    with pytest.raises(SolveError, match="GMRES"):
        solve(m, NewtonOptions(restart=1, max_cycles=1))
    with pytest.raises(ValueError):
        solve(m, z0=np.zeros(m.size - 1))
    with pytest.raises(ValueError):
        m.calibration.value_added_shares[0, 0] = 0.5
    with pytest.raises(ValueError, match="shares"):
        prepare({**a, "final_demand_shares": a["final_demand_shares"] * 1.00001})
    with pytest.raises(ValueError):
        NewtonOptions(restart=1.5)
    with pytest.raises(ValueError):
        prepare({**a, "technology_scale_ratio": np.zeros_like(a["technology_scale_ratio"])})

    # Pure numerical backend has no CP-specific dependence.
    def quadratic(z, data):
        return z * z - data

    root = newton(quadratic, jnp.array([2.0, 3.0]), jnp.ones(2))
    require_converged(root)
    np.testing.assert_allclose(root.z, np.sqrt([2.0, 3.0]), rtol=1e-10)
