import io
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpmodel_jax import _prepare, newton, residual, solve


def make_inputs(shock=True):
    countries, sectors = 4, 3
    rng = np.random.default_rng(20260923)
    value_added0 = rng.uniform(0.7, 1.5, countries)
    sector_alpha = rng.uniform(0.5, 1.5, sectors)
    sector_alpha /= sector_alpha.sum()
    sector_beta = rng.uniform(0.4, 0.8, sectors)
    alpha = np.broadcast_to(sector_alpha, (countries, sectors)).copy()
    beta = np.broadcast_to(sector_beta, (countries, sectors)).copy()
    gamma = (1 - beta[:, :, None]) * sector_alpha[None, None, :]
    openness = rng.uniform(0.25, 0.65, sectors)
    domestic = np.diag(value_added0)
    international = np.outer(value_added0, value_added0) / value_added0.sum()
    net_trade_value0 = domestic[:, :, None] * (1 - openness) + international[:, :, None] * openness
    net_trade_value0 *= sector_alpha / np.dot(sector_alpha, sector_beta)
    tariff0 = np.zeros_like(net_trade_value0)
    tariff = tariff0.copy()
    iceberg_cost_ratio = np.ones_like(net_trade_value0)
    technology_scale_ratio = np.ones((countries, sectors))
    if shock:
        tariff = rng.uniform(0.03, 0.12, net_trade_value0.shape)
        tariff[np.arange(countries), np.arange(countries), :] = 0
        iceberg_cost_ratio = rng.uniform(0.92, 0.98, net_trade_value0.shape)
        iceberg_cost_ratio[np.arange(countries), np.arange(countries), :] = 1
        technology_scale_ratio = rng.uniform(0.95, 1.10, (countries, sectors))
    theta = np.linspace(4.0, 6.0, sectors)
    return {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "net_trade_value": net_trade_value0,
        "tariff_rates": tariff0,
        "counterfactual_tariff_rates": tariff,
        "iceberg_cost_ratio": iceberg_cost_ratio,
        "technology_scale_ratio": technology_scale_ratio,
    }


def test_equilibrium():
    inputs = make_inputs()
    original = {key: value.copy() for key, value in inputs.items()}
    result = solve(**inputs)
    assert max(result["diagnostics"].values()) < 2e-5
    assert solve(**inputs, initial_log_hats=result["log_hats"])["iterations"] == 0
    net_trade_value = result["net_trade_value"]
    tariff = inputs["counterfactual_tariff_rates"]
    alpha, beta, gamma = (inputs[key] for key in ("alpha", "beta", "gamma"))
    net_trade_value0 = inputs["net_trade_value"]
    value_added0 = (beta * net_trade_value0.sum(0)).sum(1)
    deficit0 = net_trade_value0.sum((1, 2)) - net_trade_value0.sum((0, 2))
    np.testing.assert_allclose(
        np.dot(value_added0, result["wage_ratio"]) / value_added0.sum(), 1, rtol=2e-5
    )
    np.testing.assert_allclose(
        np.exp(result["log_hats"]),
        np.concatenate(
            (
                result["wage_ratio"],
                result["sector_price_ratio"].ravel(),
                result["expenditure_ratio"].ravel(),
            )
        ),
        rtol=3e-6,
    )
    income = result["wage_ratio"] * value_added0 + deficit0 + (net_trade_value * tariff).sum((1, 2))
    demand = alpha * income[:, None] + np.einsum("nkj,nk->nj", gamma, result["output"])
    np.testing.assert_allclose(result["income"], income, rtol=2e-5)
    np.testing.assert_allclose(result["expenditure"], demand, rtol=2e-5)
    np.testing.assert_allclose(result["output"], net_trade_value.sum(0), rtol=2e-5)
    np.testing.assert_allclose(
        (beta * result["output"]).sum(1), result["wage_ratio"] * value_added0, rtol=2e-5
    )
    np.testing.assert_allclose(result["trade_shares"].sum(1), 1, atol=1e-6)
    unit_cost_ratio = result["wage_ratio"][:, None] ** beta * np.prod(
        result["sector_price_ratio"][:, None, :] ** gamma, axis=2
    )
    trade_shares0 = net_trade_value0 * (1 + inputs["tariff_rates"])
    trade_shares0 /= trade_shares0.sum(1, keepdims=True)
    delivered_cost_ratio = (
        unit_cost_ratio[None, :, :]
        * inputs["iceberg_cost_ratio"]
        * (1 + tariff)
        / (1 + inputs["tariff_rates"])
    )
    sector_price_ratio = (
        trade_shares0
        * inputs["technology_scale_ratio"][None, :, :]
        * delivered_cost_ratio ** -inputs["theta"]
    ).sum(1) ** (-1 / inputs["theta"])
    np.testing.assert_allclose(result["sector_price_ratio"], sector_price_ratio, rtol=2e-5)
    for key, value in inputs.items():
        np.testing.assert_array_equal(value, original[key])


def test_analytic_no_shock():
    inputs = make_inputs(shock=False)
    result = solve(**inputs)
    for key, value in result.items():
        if key.endswith("_ratio"):
            np.testing.assert_allclose(value, 1, atol=3e-6, rtol=0, err_msg=key)
    net_trade_value0 = inputs["net_trade_value"]
    np.testing.assert_allclose(result["net_trade_value"], net_trade_value0, rtol=3e-6)
    np.testing.assert_allclose(result["expenditure"], net_trade_value0.sum(1), rtol=3e-6)
    np.testing.assert_allclose(result["output"], net_trade_value0.sum(0), rtol=3e-6)
    np.testing.assert_allclose(
        result["income"], (inputs["beta"] * net_trade_value0.sum(0)).sum(1), rtol=3e-6
    )


def test_jvp():
    data = _prepare(**make_inputs())
    rng = np.random.default_rng(741)
    countries, sectors = data["beta"].shape
    log_hats = jnp.asarray(0.02 * rng.normal(size=countries + 2 * countries * sectors))
    _, jvp = jax.linearize(lambda x: residual(x, data), log_hats)
    direction = rng.normal(size=log_hats.size)
    direction /= np.linalg.norm(direction)
    direction = jnp.asarray(direction, dtype=jnp.float32)
    finite_difference = (
        np.asarray(residual(log_hats + 0.005 * direction, data))
        - np.asarray(residual(log_hats - 0.005 * direction, data))
    ) / 0.01
    assert (
        np.linalg.norm(np.asarray(jvp(direction)) - finite_difference)
        / np.linalg.norm(finite_difference)
        < 2e-4
    )


def test_nominal_scale_condition():
    data = _prepare(**make_inputs(shock=False))
    countries, sectors = data["beta"].shape
    log_hats = jnp.full(countries + 2 * countries * sectors, jnp.log(1.1))
    expected = np.zeros(log_hats.size)
    expected[countries - 1] = np.log(1.1)
    np.testing.assert_allclose(residual(log_hats, data), expected, atol=1e-6, rtol=0)


def test_units_and_country_permutation():
    inputs = make_inputs()
    reference = solve(**inputs)
    scaled = solve(**{**inputs, "net_trade_value": inputs["net_trade_value"] * 1e9})
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "trade_shares"):
        np.testing.assert_allclose(scaled[key], reference[key], rtol=2e-5, atol=3e-6)
    for key in ("expenditure", "trade_value", "net_trade_value", "output", "income"):
        np.testing.assert_allclose(scaled[key] / 1e9, reference[key], rtol=2e-5, atol=3e-6)
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
        elif key == "theta":
            permuted[key] = value
        else:
            permuted[key] = value[permutation]
    result = solve(**permuted)
    for key in ("wage_ratio", "sector_price_ratio", "welfare_ratio", "output"):
        np.testing.assert_allclose(result[key], reference[key][permutation], rtol=2e-5, atol=3e-6)


def test_analytic_one_country_and_zero_links():
    inputs = {
        "theta": [4.0],
        "alpha": [[1.0]],
        "beta": [[1.0]],
        "gamma": [[[0.0]]],
        "net_trade_value": [[[2.0]]],
        "tariff_rates": [[[0.0]]],
    }
    result = solve(**inputs, technology_scale_ratio=[[1.21]])
    np.testing.assert_allclose(result["wage_ratio"], 1, atol=3e-7)
    np.testing.assert_allclose(result["welfare_ratio"], 1.21**0.25, atol=3e-7)
    net_trade_value0 = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])[:, :, None]
    result = solve(
        theta=[4.0],
        alpha=np.ones((3, 1)),
        beta=np.ones((3, 1)),
        gamma=np.zeros((3, 1, 1)),
        net_trade_value=net_trade_value0,
        tariff_rates=np.zeros_like(net_trade_value0),
        technology_scale_ratio=[[1.1], [1.0], [0.9]],
    )
    assert np.all(result["net_trade_value"][net_trade_value0 == 0] == 0)
    assert max(result["diagnostics"].values()) < 2e-5


def test_validation_and_failures():
    inputs = make_inputs()
    with pytest.raises(RuntimeError, match="did not converge"):
        solve(**inputs, max_iter=1)
    with pytest.raises(ValueError, match="shares"):
        solve(**{**inputs, "alpha": inputs["alpha"] * 1.001})


def test_input_precision():
    inputs = make_inputs()
    reference = solve(**inputs)
    with jax.enable_x64():
        result = solve(**{key: value.astype(np.float32) for key, value in inputs.items()})
        assert jax.config.x64_enabled
        for key, value in reference.items():
            if isinstance(value, np.ndarray):
                assert result[key].dtype == np.float32
                np.testing.assert_allclose(result[key], value, rtol=2e-5, atol=2e-6)


def test_generic_newton():
    def linear(z, data):
        return data["matrix"] @ z - data["target"]

    expected = jnp.array([100.0, -80.0])
    matrix = jnp.array([[2.0, 1.0], [-1.0, 3.0]])
    data = {"matrix": matrix, "target": matrix @ expected}
    result = newton(linear, data, jnp.zeros(2), max_iter=3)
    np.testing.assert_allclose(result["z"], expected, rtol=1e-6)

    def nonlinear(z, data):
        x, y = z
        return jnp.array([x * x + y, x + y * y]) - data

    result = newton(nonlinear, jnp.array([9.0, 23.0]), [-3.0, 4.0])
    np.testing.assert_allclose(result["z"], [-2.0, 5.0], atol=1e-5)


def test_backtracking():
    def exponential(z, data):
        return jnp.exp(z) - data

    result = newton(exponential, jnp.array([2.0]), jnp.array([-3.0]))
    np.testing.assert_allclose(result["z"], np.log(2), atol=3e-6)


def test_newton_failures():
    with pytest.raises(RuntimeError, match="did not converge"):
        newton(lambda z, data: z, None, [np.nan])
    with pytest.raises(RuntimeError, match="line search"):
        newton(lambda z, data: z**2 + 1, None, [0.0])


def test_newton_iteration_budget():
    def linear(z, data):
        return z - data

    root = newton(linear, jnp.array([3.0]), [0.0], max_iter=1)
    np.testing.assert_allclose(root["z"], [3.0], atol=1e-6)
    assert root["iterations"] == 1
    assert newton(linear, jnp.array([3.0]), root["z"], max_iter=0)["iterations"] == 0
    with pytest.raises(RuntimeError, match="did not converge"):
        newton(linear, jnp.array([3.0]), [0.0], max_iter=0)


def test_nafta_archive_validation():
    load_nafta = runpy.run_path(Path(__file__).resolve().parents[1] / "examples/nafta_example.py")[
        "load_nafta"
    ]
    download = Mock(return_value=io.BytesIO(b"invalid archive"))
    with TemporaryDirectory() as directory, patch.dict(load_nafta.__globals__, urlopen=download):
        archive_path = Path(directory) / "data.zip"
        with pytest.raises(ValueError, match="archive"):
            load_nafta(archive_path)
        assert not archive_path.exists()
        archive_path.write_bytes(b"invalid archive")
        with pytest.raises(ValueError, match="archive"):
            load_nafta(archive_path)
        assert archive_path.read_bytes() == b"invalid archive"
        download.assert_called_once()
