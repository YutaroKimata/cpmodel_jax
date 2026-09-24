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
    baseline_net_trade_value,
    baseline_tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=1.0,
    technology_scale_ratio=1.0,
):
    theta = np.asarray(theta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    baseline_net_trade_value = np.asarray(baseline_net_trade_value, dtype=float)
    baseline_tariff_rates = np.asarray(baseline_tariff_rates, dtype=float)
    counterfactual_tariff_rates = (
        baseline_tariff_rates
        if counterfactual_tariff_rates is None
        else np.asarray(counterfactual_tariff_rates, dtype=float)
    )
    iceberg_cost_ratio = np.asarray(iceberg_cost_ratio, dtype=float)
    technology_scale_ratio = np.broadcast_to(
        np.asarray(technology_scale_ratio, dtype=float), beta.shape
    )
    if not np.allclose(alpha.sum(1), 1) or not np.allclose(beta + gamma.sum(2), 1):
        raise ValueError("demand and production shares must sum to one")

    baseline_output = baseline_net_trade_value.sum(0)
    baseline_trade_value = baseline_net_trade_value * (1 + baseline_tariff_rates)
    baseline_expenditure = baseline_trade_value.sum(1)
    baseline_value_added = (beta * baseline_output).sum(1)
    baseline_deficit = baseline_net_trade_value.sum((1, 2)) - baseline_net_trade_value.sum((0, 2))
    baseline_income = (
        baseline_value_added
        + baseline_deficit
        + (baseline_net_trade_value * baseline_tariff_rates).sum((1, 2))
    )

    expenditure_scale = np.maximum(
        baseline_expenditure,
        alpha * baseline_income[:, None] + np.einsum("nkj,nk->nj", gamma, baseline_output),
    )
    baseline_trade_shares = baseline_trade_value / baseline_expenditure[:, None, :]
    with np.errstate(divide="ignore"):
        log_baseline_trade_shares = np.log(baseline_trade_shares)
    log_cost_shock = np.log(technology_scale_ratio)[None, :, :] - theta * (
        np.log1p(counterfactual_tariff_rates)
        - np.log1p(baseline_tariff_rates)
        + np.log(iceberg_cost_ratio)
    )
    data = {
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "baseline_output": baseline_output,
        "baseline_value_added": baseline_value_added,
        "baseline_deficit": baseline_deficit,
        "baseline_income": baseline_income,
        "baseline_expenditure": baseline_expenditure,
        "expenditure_scale": expenditure_scale,
        "baseline_tariff_rates": baseline_tariff_rates,
        "counterfactual_tariff_rates": counterfactual_tariff_rates,
        "net_share_factor": 1 / (1 + counterfactual_tariff_rates),
        "log_trade_weights": log_baseline_trade_shares + log_cost_shock,
        "log_cost_shock": log_cost_shock,
    }
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in data.items()}


def _state(log_ratios, data):
    n, j = data["beta"].shape
    log_wage_ratio = log_ratios[:n]
    log_sector_price_ratio = log_ratios[n : n + n * j].reshape(n, j)
    counterfactual_expenditure = data["baseline_expenditure"] * jnp.exp(
        log_ratios[n + n * j :].reshape(n, j)
    )
    log_unit_cost_ratio = data["beta"] * log_wage_ratio[:, None] + jnp.einsum(
        "njk,nk->nj", data["gamma"], log_sector_price_ratio
    )
    log_trade_weights = data["log_trade_weights"] - data["theta"] * log_unit_cost_ratio[None, :, :]
    log_share_denominator = logsumexp(log_trade_weights, axis=1)
    counterfactual_trade_shares = jnp.exp(log_trade_weights - log_share_denominator[:, None, :])
    counterfactual_net_trade_value = (
        counterfactual_trade_shares
        * counterfactual_expenditure[:, None, :]
        * data["net_share_factor"]
    )
    counterfactual_output = counterfactual_net_trade_value.sum(0)
    wage_ratio = jnp.exp(log_wage_ratio)
    counterfactual_income = (
        wage_ratio * data["baseline_value_added"]
        + data["baseline_deficit"]
        + (counterfactual_net_trade_value * data["counterfactual_tariff_rates"]).sum((1, 2))
    )
    return {
        "log_unit_cost_ratio": log_unit_cost_ratio,
        "log_sector_price_ratio": log_sector_price_ratio,
        "log_share_denominator": log_share_denominator,
        "counterfactual_trade_shares": counterfactual_trade_shares,
        "counterfactual_net_trade_value": counterfactual_net_trade_value,
        "wage_ratio": wage_ratio,
        "counterfactual_output": counterfactual_output,
        "counterfactual_income": counterfactual_income,
        "counterfactual_expenditure": counterfactual_expenditure,
    }


@jax.jit
def residual(log_ratios, data):
    state = _state(log_ratios, data)
    log_wage_ratio_target = jnp.log(
        (data["beta"] * state["counterfactual_output"]).sum(1) / data["baseline_value_added"]
    )
    wage_residual = log_ratios[: state["wage_ratio"].size - 1] - log_wage_ratio_target[:-1]
    numeraire_residual = jnp.log(
        (data["baseline_value_added"] * state["wage_ratio"]).sum()
        / data["baseline_value_added"].sum()
    )
    price_residual = (
        state["log_sector_price_ratio"] + state["log_share_denominator"] / data["theta"]
    )
    counterfactual_demand = data["alpha"] * state["counterfactual_income"][:, None] + jnp.einsum(
        "nkj,nk->nj", data["gamma"], state["counterfactual_output"]
    )
    expenditure_residual = (state["counterfactual_expenditure"] - counterfactual_demand) / data[
        "expenditure_scale"
    ]
    r = jnp.concatenate(
        (
            wage_residual,
            numeraire_residual.reshape(1),
            price_residual.ravel(),
            expenditure_residual.ravel(),
        )
    )
    return jnp.where(jnp.all(state["counterfactual_income"] > 0), r, jnp.full_like(r, jnp.nan))


@jax.jit
def _results(log_ratios, data):
    state = _state(log_ratios, data)
    trade_share_ratio = jnp.exp(
        data["log_cost_shock"]
        - data["theta"] * state["log_unit_cost_ratio"][None, :, :]
        - state["log_share_denominator"][:, None, :]
    )
    expenditure_ratio = state["counterfactual_expenditure"] / data["baseline_expenditure"]
    counterfactual_trade_value = (
        state["counterfactual_trade_shares"] * state["counterfactual_expenditure"][:, None, :]
    )
    consumer_price_ratio = jnp.exp((data["alpha"] * state["log_sector_price_ratio"]).sum(1))
    income_ratio = state["counterfactual_income"] / data["baseline_income"]
    values = {
        "wage_ratio": state["wage_ratio"],
        "sector_price_ratio": jnp.exp(state["log_sector_price_ratio"]),
        "unit_cost_ratio": jnp.exp(state["log_unit_cost_ratio"]),
        "trade_share_ratio": trade_share_ratio,
        "expenditure_ratio": expenditure_ratio,
        "trade_value_ratio": trade_share_ratio * expenditure_ratio[:, None, :],
        "net_trade_value_ratio": trade_share_ratio
        * expenditure_ratio[:, None, :]
        * (1 + data["baseline_tariff_rates"])
        / (1 + data["counterfactual_tariff_rates"]),
        "output_ratio": state["counterfactual_output"] / data["baseline_output"],
        "income_ratio": income_ratio,
        "consumer_price_ratio": consumer_price_ratio,
        "welfare_ratio": income_ratio / consumer_price_ratio,
        "real_wage_ratio": state["wage_ratio"] / consumer_price_ratio,
        "counterfactual_trade_shares": state["counterfactual_trade_shares"],
        "counterfactual_expenditure": state["counterfactual_expenditure"],
        "counterfactual_trade_value": counterfactual_trade_value,
        "counterfactual_net_trade_value": state["counterfactual_net_trade_value"],
        "counterfactual_output": state["counterfactual_output"],
        "counterfactual_income": state["counterfactual_income"],
    }

    labor_error = jnp.max(
        jnp.abs(
            (data["beta"] * state["counterfactual_output"]).sum(1)
            / (state["wage_ratio"] * data["baseline_value_added"])
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
    baseline_net_trade_value,
    baseline_tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=1.0,
    technology_scale_ratio=1.0,
    initial_log_ratios=None,
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
        baseline_net_trade_value,
        baseline_tariff_rates,
        counterfactual_tariff_rates,
        iceberg_cost_ratio,
        technology_scale_ratio,
    )
    n, j = data["beta"].shape
    if initial_log_ratios is None:
        initial_log_ratios = jnp.concatenate(
            (
                jnp.zeros(n + n * j, dtype=jnp.float32),
                jnp.log(data["expenditure_scale"] / data["baseline_expenditure"]).ravel(),
            )
        )
    root = newton(
        residual,
        data,
        initial_log_ratios,
        tol=tolerance,
        max_iter=max_iter,
        max_trials=max_trials,
    )
    log_ratios = root["z"]
    values, labor_error = _results(log_ratios, data)
    values = {key: np.asarray(value) for key, value in values.items()}
    diagnostics = {
        "residual_inf": float(jnp.max(jnp.abs(root["residual"]))),
        "labor_error": float(labor_error),
    }
    if (
        not diagnostics["labor_error"] <= 100 * tolerance
        or any(not np.all(np.isfinite(v)) for v in values.values())
        or np.any(values["counterfactual_income"] <= 0)
    ):
        raise RuntimeError("equilibrium verification failed")
    return {
        **values,
        "diagnostics": diagnostics,
        "iterations": root["iterations"],
        "log_ratios": np.asarray(log_ratios),
        "residual": np.asarray(root["residual"]),
        "wall_seconds": perf_counter() - start,
    }
