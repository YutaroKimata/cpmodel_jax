from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from jax.scipy.special import logsumexp


@jax.jit(static_argnames=("f",))
def _direction(f, data, z):
    residual, jvp = jax.linearize(lambda x: f(x, data), z)
    return gmres(jvp, -residual, tol=0.01, solve_method="incremental")[0]


@jax.jit(static_argnames=("f",))
def _evaluate(f, data, z):
    residual = f(z, data)
    return residual, jnp.max(jnp.abs(residual)), jnp.linalg.norm(residual)


def newton(f, data, z0, *, tol=1e-5, max_iter=60, max_trials=25):
    z = jnp.asarray(z0, dtype=jnp.float32)
    residual, error, residual_norm = _evaluate(f, data, z)
    iteration = 0
    while iteration < max_iter and float(error) > tol:
        direction = _direction(f, data, z)
        step_size = 1.0
        for _ in range(max_trials):
            trial_z = z + step_size * direction
            trial_residual, trial_error, trial_norm = _evaluate(f, data, trial_z)
            if float(trial_norm) <= (1 - 1e-4 * step_size) * float(residual_norm):
                break
            step_size *= 0.5
        else:
            raise RuntimeError("line search failed")
        z, residual, error, residual_norm = (
            trial_z,
            trial_residual,
            trial_error,
            trial_norm,
        )
        iteration += 1
    if not float(error) <= tol:
        raise RuntimeError("Newton did not converge")
    return {"z": z, "residual": residual, "iterations": iteration}


def _prepare(
    theta,
    alpha,
    beta,
    gamma,
    net_trade_value,
    tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=1.0,
    technology_scale_ratio=1.0,
):
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    net_trade_value0 = np.asarray(net_trade_value, dtype=float)
    tariff0 = np.asarray(tariff_rates, dtype=float)
    tariff = (
        tariff0
        if counterfactual_tariff_rates is None
        else np.asarray(counterfactual_tariff_rates, dtype=float)
    )
    iceberg_cost_ratio = np.asarray(iceberg_cost_ratio, dtype=float)
    technology_scale_ratio = np.broadcast_to(
        np.asarray(technology_scale_ratio, dtype=float), beta.shape
    )
    if not np.allclose(alpha.sum(1), 1) or not np.allclose(beta + gamma.sum(2), 1):
        raise ValueError("demand and production shares must sum to one")

    output0 = net_trade_value0.sum(0)
    trade_value0 = net_trade_value0 * (1 + tariff0)
    expenditure0 = trade_value0.sum(1)
    value_added0 = (beta * output0).sum(1)
    deficit0 = net_trade_value0.sum((1, 2)) - net_trade_value0.sum((0, 2))
    income0 = value_added0 + deficit0 + (net_trade_value0 * tariff0).sum((1, 2))

    expenditure_scale = np.maximum(
        expenditure0, alpha * income0[:, None] + np.einsum("nkj,nk->nj", gamma, output0)
    )
    trade_shares0 = trade_value0 / expenditure0[:, None, :]
    with np.errstate(divide="ignore"):
        log_trade_shares0 = np.log(trade_shares0)
    log_cost_shock = np.log(technology_scale_ratio)[None, :, :] - theta * (
        np.log1p(tariff) - np.log1p(tariff0) + np.log(iceberg_cost_ratio)
    )
    data = {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "output0": output0,
        "value_added0": value_added0,
        "deficit0": deficit0,
        "income0": income0,
        "expenditure0": expenditure0,
        "expenditure_scale": expenditure_scale,
        "tariff0": tariff0,
        "tariff": tariff,
        "net_share_factor": 1 / (1 + tariff),
        "log_trade_weights": log_trade_shares0 + log_cost_shock,
        "log_cost_shock": log_cost_shock,
    }
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in data.items()}


def _state(log_hats, data):
    n, j = data["beta"].shape
    log_wage_ratio = log_hats[:n]
    log_sector_price_ratio = log_hats[n : n + n * j].reshape(n, j)
    expenditure = data["expenditure0"] * jnp.exp(log_hats[n + n * j :].reshape(n, j))
    log_unit_cost_ratio = data["beta"] * log_wage_ratio[:, None] + jnp.einsum(
        "njk,nk->nj", data["gamma"], log_sector_price_ratio
    )
    log_trade_weights = data["log_trade_weights"] - data["theta"] * log_unit_cost_ratio[None, :, :]
    log_share_denominator = logsumexp(log_trade_weights, axis=1)
    trade_shares = jnp.exp(log_trade_weights - log_share_denominator[:, None, :])
    net_trade_value = trade_shares * expenditure[:, None, :] * data["net_share_factor"]
    output = net_trade_value.sum(0)
    wage_ratio = jnp.exp(log_wage_ratio)
    income = (
        wage_ratio * data["value_added0"]
        + data["deficit0"]
        + (net_trade_value * data["tariff"]).sum((1, 2))
    )
    return {
        "log_unit_cost_ratio": log_unit_cost_ratio,
        "log_sector_price_ratio": log_sector_price_ratio,
        "log_share_denominator": log_share_denominator,
        "trade_shares": trade_shares,
        "net_trade_value": net_trade_value,
        "wage_ratio": wage_ratio,
        "output": output,
        "income": income,
        "expenditure": expenditure,
    }


@jax.jit
def residual(log_hats, data):
    state = _state(log_hats, data)
    log_wage_ratio_target = jnp.log((data["beta"] * state["output"]).sum(1) / data["value_added0"])
    wage_residual = log_hats[: state["wage_ratio"].size - 1] - log_wage_ratio_target[:-1]
    numeraire_residual = jnp.log(
        (data["value_added0"] * state["wage_ratio"]).sum() / data["value_added0"].sum()
    )
    price_residual = (
        state["log_sector_price_ratio"] + state["log_share_denominator"] / data["theta"]
    )
    demand = data["alpha"] * state["income"][:, None] + jnp.einsum(
        "nkj,nk->nj", data["gamma"], state["output"]
    )
    expenditure_residual = (state["expenditure"] - demand) / data["expenditure_scale"]
    r = jnp.concatenate(
        (
            wage_residual,
            numeraire_residual.reshape(1),
            price_residual.ravel(),
            expenditure_residual.ravel(),
        )
    )
    return jnp.where(jnp.all(state["income"] > 0), r, jnp.full_like(r, jnp.nan))


@jax.jit
def _results(log_hats, data):
    state = _state(log_hats, data)
    trade_share_ratio = jnp.exp(
        data["log_cost_shock"]
        - data["theta"] * state["log_unit_cost_ratio"][None, :, :]
        - state["log_share_denominator"][:, None, :]
    )
    expenditure_ratio = state["expenditure"] / data["expenditure0"]
    trade_value = state["trade_shares"] * state["expenditure"][:, None, :]
    consumer_price_ratio = jnp.exp((data["alpha"] * state["log_sector_price_ratio"]).sum(1))
    income_ratio = state["income"] / data["income0"]
    values = {
        "wage_ratio": state["wage_ratio"],
        "sector_price_ratio": jnp.exp(state["log_sector_price_ratio"]),
        "unit_cost_ratio": jnp.exp(state["log_unit_cost_ratio"]),
        "trade_share_ratio": trade_share_ratio,
        "expenditure_ratio": expenditure_ratio,
        "trade_value_ratio": trade_share_ratio * expenditure_ratio[:, None, :],
        "net_trade_value_ratio": trade_share_ratio
        * expenditure_ratio[:, None, :]
        * (1 + data["tariff0"])
        / (1 + data["tariff"]),
        "output_ratio": state["output"] / data["output0"],
        "income_ratio": income_ratio,
        "consumer_price_ratio": consumer_price_ratio,
        "welfare_ratio": income_ratio / consumer_price_ratio,
        "real_wage_ratio": state["wage_ratio"] / consumer_price_ratio,
        "trade_shares": state["trade_shares"],
        "expenditure": state["expenditure"],
        "trade_value": trade_value,
        "net_trade_value": state["net_trade_value"],
        "output": state["output"],
        "income": state["income"],
    }

    labor_error = jnp.max(
        jnp.abs(
            (data["beta"] * state["output"]).sum(1) / (state["wage_ratio"] * data["value_added0"])
            - 1
        )
    )
    return values, labor_error


def solve(
    *,
    theta,
    alpha,
    beta,
    gamma,
    net_trade_value,
    tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=1.0,
    technology_scale_ratio=1.0,
    initial_log_hats=None,
    tolerance=1e-5,
    max_iter=60,
    max_trials=25,
):
    start = perf_counter()
    data = _prepare(
        theta,
        alpha,
        beta,
        gamma,
        net_trade_value,
        tariff_rates,
        counterfactual_tariff_rates,
        iceberg_cost_ratio,
        technology_scale_ratio,
    )
    n, j = data["beta"].shape
    if initial_log_hats is None:
        initial_log_hats = jnp.concatenate(
            (
                jnp.zeros(n + n * j, dtype=jnp.float32),
                jnp.log(data["expenditure_scale"] / data["expenditure0"]).ravel(),
            )
        )
    root = newton(
        residual,
        data,
        initial_log_hats,
        tol=tolerance,
        max_iter=max_iter,
        max_trials=max_trials,
    )
    log_hats = root["z"]
    values, labor_error = _results(log_hats, data)
    values = {key: np.asarray(value) for key, value in values.items()}
    diagnostics = {
        "residual_inf": float(jnp.max(jnp.abs(root["residual"]))),
        "labor_error": float(labor_error),
    }
    if (
        not diagnostics["labor_error"] <= 100 * tolerance
        or any(not np.all(np.isfinite(v)) for v in values.values())
        or np.any(values["income"] <= 0)
    ):
        raise RuntimeError("equilibrium verification failed")
    return {
        **values,
        "diagnostics": diagnostics,
        "iterations": root["iterations"],
        "log_hats": np.asarray(log_hats),
        "residual": np.asarray(root["residual"]),
        "wall_seconds": perf_counter() - start,
    }
