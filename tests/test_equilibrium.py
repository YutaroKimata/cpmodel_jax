from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cpmodel_jax
from cpmodel_jax import EXAMPLE_NAMES, _prepare, _state, load_example, newton, residual, solve


def test_version_and_precision():
    assert cpmodel_jax.__version__ == version("cpmodel-jax")
    assert jax.config.x64_enabled


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_examples(name):
    inputs = load_example(name)
    result = solve(**inputs)
    assert max(result["diagnostics"].values()) < 1e-8
    assert np.all(result["welfare_ratio"] > 0)
    assert result["history"].shape == (result["iterations"], 8)
    assert solve(**inputs, z0=result["z"])["iterations"] == 0
    for key, value in inputs.items():
        np.testing.assert_array_equal(value, load_example(name)[key])


@pytest.mark.parametrize("countries,sectors", [(4, 3), (16, 10)])
def test_analytic_no_shock(countries, sectors):
    inputs = load_example("no_shock", countries=countries, sectors=sectors)
    result = solve(**inputs)
    for key, value in result.items():
        if key.endswith("_ratio"):
            np.testing.assert_allclose(value, 1, atol=1e-10, rtol=0, err_msg=key)
    trade = inputs["net_trade_value"]
    np.testing.assert_allclose(result["net_trade_value"], trade, rtol=1e-10)
    np.testing.assert_allclose(result["expenditure"], trade.sum(1), rtol=1e-10)
    np.testing.assert_allclose(result["output"], trade.sum(0), rtol=1e-10)
    np.testing.assert_allclose(
        result["income"], (inputs["value_added_shares"] * trade.sum(0)).sum(1), rtol=1e-10
    )


def test_jvp_and_off_equilibrium_accounting():
    data = _prepare(**load_example())
    rng = np.random.default_rng(741)
    z = jnp.asarray(0.02 * rng.normal(size=27))
    s = _state(z, data)
    net_trade = np.asarray(s["shares"] * s["expenditure"][:, None, :] / (1 + data["tariff"]))
    revenue = (np.asarray(data["tariff"]) * net_trade).sum((1, 2))
    np.testing.assert_allclose(
        s["income"], s["wage"] * data["value_added"] + data["deficit"] + revenue
    )
    np.testing.assert_allclose(s["output"], net_trade.sum(0))
    np.testing.assert_allclose(np.sum(np.asarray(data["value_added"] * s["wage"])), 1, atol=1e-14)
    _, Jv = jax.linearize(lambda x: residual(x, data), z)
    v = rng.normal(size=z.size)
    v /= np.linalg.norm(v)
    fd = (
        np.asarray(residual(z + 1e-5 * v, data)) - np.asarray(residual(z - 1e-5 * v, data))
    ) / 2e-5
    assert np.linalg.norm(np.asarray(Jv(v)) - fd) / np.linalg.norm(fd) < 1e-7


def test_units_and_country_permutation():
    inputs = load_example()
    reference = solve(**inputs)
    scaled = solve(**{**inputs, "net_trade_value": inputs["net_trade_value"] * 1e9})
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "trade_shares"):
        np.testing.assert_allclose(scaled[key], reference[key], rtol=1e-8, atol=1e-10)
    permutation = np.array([2, 0, 3, 1])
    permuted = {}
    for key, value in inputs.items():
        if key in (
            "net_trade_value",
            "tariff_rates",
            "counterfactual_tariff_rates",
            "iceberg_cost_ratio",
        ):
            permuted[key] = value[permutation][:, permutation, :]
        elif key == "trade_elasticities":
            permuted[key] = value
        else:
            permuted[key] = value[permutation]
    result = solve(**permuted)
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "output"):
        np.testing.assert_allclose(result[key], reference[key][permutation], rtol=1e-8, atol=1e-10)


def test_analytic_one_country_and_zero_links():
    one = {
        "trade_elasticities": [4.0],
        "final_demand_shares": [[1.0]],
        "value_added_shares": [[1.0]],
        "input_output_shares": [[[0.0]]],
        "net_trade_value": [[[2.0]]],
        "tariff_rates": [[[0.0]]],
    }
    result = solve(**one, technology_scale_ratio=[[1.21]])
    np.testing.assert_allclose(result["wage_ratio"], 1, atol=1e-14)
    np.testing.assert_allclose(result["welfare_ratio"], 1.21**0.25, atol=1e-12)
    trade = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])[:, :, None]
    result = solve(
        trade_elasticities=[4.0],
        final_demand_shares=np.ones((3, 1)),
        value_added_shares=np.ones((3, 1)),
        input_output_shares=np.zeros((3, 1, 1)),
        net_trade_value=trade,
        tariff_rates=np.zeros_like(trade),
        technology_scale_ratio=[[1.1], [1.0], [0.9]],
    )
    assert np.all(result["net_trade_value"][trade == 0] == 0)
    assert max(result["diagnostics"].values()) < 1e-8


def test_validation_and_failures():
    inputs = load_example()
    with pytest.raises(RuntimeError, match="iteration limit"):
        solve(**inputs, max_steps=1)
    with pytest.raises(RuntimeError, match="GMRES"):
        solve(**inputs, restart=1, max_cycles=1)
    with pytest.raises(ValueError, match="initial"):
        solve(**inputs, z0=np.zeros(2))
    with pytest.raises(ValueError, match="shares"):
        solve(**{**inputs, "final_demand_shares": inputs["final_demand_shares"] * 1.00001})
    with pytest.raises(ValueError):
        solve(**inputs, restart=1.5)
    with pytest.raises(ValueError):
        solve(**{**inputs, "technology_scale_ratio": np.zeros((4, 3))})
    with pytest.raises(ValueError, match="Unknown example"):
        load_example("not-an-example")


def test_generic_newton():
    def quadratic(z, data):
        return z * z - data

    result = newton(quadratic, jnp.array([2.0, 3.0]), jnp.ones(2))
    np.testing.assert_allclose(result["z"], np.sqrt([2.0, 3.0]), rtol=1e-10)
    again = newton(quadratic, jnp.array([2.0, 3.0]), result["z"])
    assert again["steps"] == 0
    assert again["history"].shape == (0, 8)
    with pytest.raises(RuntimeError, match="nonfinite"):
        newton(quadratic, jnp.array([np.nan]), jnp.ones(1))


def test_backtracking():
    def exponential(z, data):
        return jnp.exp(z) - data

    result = newton(exponential, jnp.array([2.0]), jnp.array([-3.0]), max_log_step=100)
    np.testing.assert_allclose(result["z"], np.log(2), atol=1e-10)
    assert np.any(result["history"][:, 5] > 0)
    with pytest.raises(RuntimeError, match="line search"):
        newton(exponential, jnp.array([2.0]), jnp.array([-3.0]), max_log_step=100, max_backtracks=1)
