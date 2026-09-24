from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

__version__ = "0.5.0"
EXAMPLE_NAMES = ("no_shock", "tariff", "iceberg", "combined", "technology", "full")
HISTORY_COLUMNS = (
    "residual_inf",
    "residual_l2",
    "forcing_eta",
    "linear_relative_residual",
    "step_length",
    "backtracks",
    "next_residual_inf",
    "gmres_info",
)


@partial(jax.jit, static_argnames=("function", "restart", "max_cycles", "max_log_step"))
def _newton_direction(function, data, z, restart, max_cycles, max_log_step):
    residual, Jv = jax.linearize(lambda x: function(x, data), z)
    residual_norm = jnp.linalg.norm(residual)
    eta = jnp.clip(jnp.sqrt(jnp.max(jnp.abs(residual))), 1e-10, 0.01)
    absolute_tolerance = 1e-14
    step, info = gmres(
        Jv,
        -residual,
        tol=eta,
        atol=absolute_tolerance,
        restart=min(restart, z.size),
        maxiter=max_cycles,
        solve_method="incremental",
    )
    linear_error = jnp.linalg.norm(Jv(step) + residual) / residual_norm
    linear_tolerance = jnp.maximum(eta, absolute_tolerance / residual_norm)
    valid = (
        (info == 0)
        & jnp.all(jnp.isfinite(step))
        & jnp.isfinite(linear_error)
        & (linear_error <= jnp.maximum(2 * linear_tolerance, 1e-8))
    )
    length = jnp.minimum(1.0, max_log_step / jnp.maximum(jnp.max(jnp.abs(step)), 1e-300))
    return step, eta, linear_error, info, valid, length


@partial(jax.jit, static_argnames=("function",))
def _residual_norms(function, data, z):
    residual = function(z, data)
    return residual, jnp.max(jnp.abs(residual)), jnp.linalg.norm(residual)


def newton(
    function,
    data,
    z0,
    *,
    tolerance=1e-10,
    max_steps=60,
    restart=60,
    max_cycles=30,
    max_backtracks=25,
    max_log_step=1.0,
    armijo=1e-4,
):
    for name, value in (
        ("max_steps", max_steps),
        ("restart", restart),
        ("max_cycles", max_cycles),
        ("max_backtracks", max_backtracks),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(max_log_step) or max_log_step <= 0:
        raise ValueError("max_log_step must be finite and positive")
    if not 0 < armijo < 1:
        raise ValueError("armijo must lie between zero and one")
    z = jnp.asarray(z0, dtype=jnp.float64)
    residual, error, residual_norm = _residual_norms(function, data, z)
    if not np.isfinite(float(error)):
        raise RuntimeError("nonfinite initial residual")
    history = []

    for iteration in range(max_steps + 1):
        if float(error) <= tolerance:
            return {
                "z": z,
                "residual": residual,
                "steps": iteration,
                "history": np.asarray(history).reshape(-1, len(HISTORY_COLUMNS)),
            }
        if iteration == max_steps:
            raise RuntimeError("Newton iteration limit")

        step, eta, linear_error, info, linear_ok, length = _newton_direction(
            function, data, z, restart, max_cycles, max_log_step
        )
        if not bool(linear_ok):
            raise RuntimeError("GMRES failed linear residual check")

        length = float(length)
        for backtracks in range(max_backtracks):
            trial_z = z + length * step
            trial_residual, trial_error, trial_norm = _residual_norms(function, data, trial_z)
            if float(trial_norm) <= (1 - armijo * length) * float(residual_norm):
                break
            length *= 0.5
        else:
            raise RuntimeError("line search failed")

        history.append(
            [
                float(error),
                float(residual_norm),
                float(eta),
                float(linear_error),
                length,
                backtracks,
                float(trial_error),
                int(info),
            ]
        )
        z = trial_z
        residual, error, residual_norm = trial_residual, trial_error, trial_norm


def _array(name, value, shape, *, positive=False, nonnegative=True):
    x = np.asarray(value, dtype=np.float64)
    if x.shape != shape or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    if positive and np.any(x <= 0) or nonnegative and np.any(x < 0):
        raise ValueError(f"{name}: invalid sign")
    return x


def _prepare(
    trade_elasticities,
    final_demand_shares,
    value_added_shares,
    input_output_shares,
    net_trade_value,
    tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=None,
    technology_scale_ratio=None,
):
    shape = np.shape(net_trade_value)
    if len(shape) != 3 or shape[0] != shape[1] or min(shape) < 1:
        raise ValueError("net_trade_value must have shape (N,N,J), with N,J >= 1")
    n, _, j = shape
    theta = _array("trade_elasticities", trade_elasticities, (j,), positive=True)
    alpha = _array("final_demand_shares", final_demand_shares, (n, j))
    beta = _array("value_added_shares", value_added_shares, (n, j), positive=True)
    gamma = _array("input_output_shares", input_output_shares, (n, j, j))
    trade = _array("net_trade_value", net_trade_value, shape)
    tariff0 = _array("tariff_rates", tariff_rates, shape)
    tariff = _array(
        "counterfactual_tariff_rates",
        tariff0 if counterfactual_tariff_rates is None else counterfactual_tariff_rates,
        shape,
    )
    iceberg = _array(
        "iceberg_cost_ratio",
        np.ones(shape) if iceberg_cost_ratio is None else iceberg_cost_ratio,
        shape,
        positive=True,
    )
    technology = _array(
        "technology_scale_ratio",
        np.ones((n, j)) if technology_scale_ratio is None else technology_scale_ratio,
        (n, j),
        positive=True,
    )
    if not np.allclose(alpha.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("final demand shares must sum to one")
    if not np.allclose(beta + gamma.sum(2), 1, atol=1e-10, rtol=0):
        raise ValueError("value added plus IO shares must sum to one")
    if not np.all(iceberg[np.arange(n), np.arange(n)] == 1):
        raise ValueError("domestic iceberg_cost_ratio must be one")

    output0 = trade.sum(0)
    expenditure0 = (trade * (1 + tariff0)).sum(1)
    value_added = (beta * output0).sum(1)
    deficit = trade.sum((1, 2)) - trade.sum((0, 2))
    income0 = value_added + deficit + (trade * tariff0).sum((1, 2))
    for value in (output0, expenditure0, value_added, income0):
        if np.any(value <= 0) or not np.all(np.isfinite(value)):
            raise ValueError(
                "baseline output, expenditure, value added and income must be positive"
            )

    # Normalize monetary units by world baseline value added.
    money_scale = value_added.sum()
    trade = trade / money_scale
    output0 = output0 / money_scale
    expenditure0 = expenditure0 / money_scale
    value_added = value_added / money_scale
    deficit = deficit / money_scale
    income0 = income0 / money_scale
    expenditure_scale = np.maximum(
        expenditure0, alpha * income0[:, None] + np.einsum("nkj,nk->nj", gamma, output0)
    )
    shares0 = trade * (1 + tariff0) / expenditure0[:, None, :]
    with np.errstate(divide="ignore"):
        log_shares0 = np.log(shares0)
    log_shock = np.log(technology)[None, :, :] - theta * (
        np.log1p(tariff) - np.log1p(tariff0) + np.log(iceberg)
    )
    data = {
        "trade_elasticities": theta,
        "final_demand_shares": alpha,
        "value_added_shares": beta,
        "input_output_shares": gamma,
        "log_y0": np.log(output0),
        "value_added": value_added,
        "deficit": deficit,
        "income0": income0,
        "expenditure0": expenditure0,
        "expenditure_scale": expenditure_scale,
        "tariff0": tariff0,
        "tariff": tariff,
        "net_share_factor": 1 / (1 + tariff),
        "log_trade_weights": log_shares0 + log_shock,
        "log_cost_shock": log_shock,
        "money_scale": money_scale,
    }
    return {key: jnp.asarray(value) for key, value in data.items()}


def _state(z, d):
    n, j = d["value_added_shares"].shape
    u = jnp.concatenate((z[: n - 1], jnp.zeros(1)))
    logw = u - logsumexp(jnp.log(d["value_added"]) + u)
    logp = z[n - 1 : n - 1 + n * j].reshape(n, j)
    E = d["expenditure0"] * jnp.exp(z[n - 1 + n * j :].reshape(n, j))
    logc = d["value_added_shares"] * logw[:, None] + jnp.einsum(
        "njk,nk->nj", d["input_output_shares"], logp
    )
    logweights = d["log_trade_weights"] - d["trade_elasticities"] * logc[None, :, :]
    denominator = logsumexp(logweights, axis=1)
    shares = jnp.exp(logweights - denominator[:, None, :])
    net_trade_value = shares * E[:, None, :] * d["net_share_factor"]
    Y = net_trade_value.sum(0)
    w = jnp.exp(logw)
    income = w * d["value_added"] + d["deficit"] + (net_trade_value * d["tariff"]).sum((1, 2))
    return {
        "log_cost": logc,
        "log_output": jnp.log(Y),
        "log_price": logp,
        "log_share_denominator": denominator,
        "shares": shares,
        "wage": w,
        "output": Y,
        "income": income,
        "expenditure": E,
    }


@jax.jit
def residual(z, d):
    s = _state(z, d)
    logtarget_w = jnp.log((d["value_added_shares"] * s["output"]).sum(1) / d["value_added"])
    fw = z[: s["wage"].size - 1] - (logtarget_w[:-1] - logtarget_w[-1])
    fp = s["log_price"] + s["log_share_denominator"] / d["trade_elasticities"]
    demand = d["final_demand_shares"] * s["income"][:, None] + jnp.einsum(
        "nkj,nk->nj", d["input_output_shares"], s["output"]
    )
    fe = (s["expenditure"] - demand) / d["expenditure_scale"]
    r = jnp.concatenate((fw, fp.ravel(), fe.ravel()))
    return jnp.where(jnp.all(s["income"] > 0), r, jnp.full_like(r, jnp.nan))


def _observables(s, d):
    share_multiplier = jnp.exp(
        d["log_cost_shock"]
        - d["trade_elasticities"] * s["log_cost"][None, :, :]
        - s["log_share_denominator"][:, None, :]
    )
    expenditure_ratio = s["expenditure"] / d["expenditure0"]
    trade_value = s["shares"] * s["expenditure"][:, None, :]
    net_trade_value = trade_value * d["net_share_factor"]
    consumer_price_ratio = jnp.exp((d["final_demand_shares"] * s["log_price"]).sum(1))
    income_ratio = s["income"] / d["income0"]
    return {
        "wage_ratio": s["wage"],
        "sector_price_ratio": jnp.exp(s["log_price"]),
        "unit_cost_ratio": jnp.exp(s["log_cost"]),
        "trade_share_ratio": share_multiplier,
        "expenditure_ratio": expenditure_ratio,
        "trade_value_ratio": share_multiplier * expenditure_ratio[:, None, :],
        "net_trade_value_ratio": share_multiplier
        * expenditure_ratio[:, None, :]
        * (1 + d["tariff0"])
        / (1 + d["tariff"]),
        "output_ratio": jnp.exp(s["log_output"] - d["log_y0"]),
        "income_ratio": income_ratio,
        "consumer_price_ratio": consumer_price_ratio,
        "welfare_ratio": income_ratio / consumer_price_ratio,
        "real_wage_ratio": s["wage"] / consumer_price_ratio,
        "trade_shares": s["shares"],
        "expenditure": s["expenditure"] * d["money_scale"],
        "trade_value": trade_value * d["money_scale"],
        "net_trade_value": net_trade_value * d["money_scale"],
        "output": s["output"] * d["money_scale"],
        "income": s["income"] * d["money_scale"],
    }


def _equation_errors(s, d):
    net_trade_value = s["shares"] * s["expenditure"][:, None, :] * d["net_share_factor"]
    sales = net_trade_value.sum(0)
    tariff_revenue = (net_trade_value * d["tariff"]).sum((1, 2))
    true_income = s["wage"] * d["value_added"] + d["deficit"] + tariff_revenue
    true_expenditure = d["final_demand_shares"] * true_income[:, None] + jnp.einsum(
        "nkj,nk->nj", d["input_output_shares"], sales
    )
    true_cost = d["value_added_shares"] * jnp.log(s["wage"])[:, None] + jnp.einsum(
        "njk,nk->nj", d["input_output_shares"], s["log_price"]
    )
    true_logp = (
        -logsumexp(
            d["log_trade_weights"] - d["trade_elasticities"] * s["log_cost"][None, :, :], axis=1
        )
        / d["trade_elasticities"]
    )
    return {
        "cost_error": jnp.max(jnp.abs(s["log_cost"] - true_cost)),
        "price_error": jnp.max(jnp.abs(s["log_price"] - true_logp)),
        "goods_market_error": jnp.max(jnp.abs(s["log_output"] - jnp.log(sales))),
        "labor_error": jnp.max(
            jnp.abs(s["wage"] * d["value_added"] - (d["value_added_shares"] * sales).sum(1))
            / (s["wage"] * d["value_added"])
        ),
        "expenditure_error": jnp.max(
            jnp.abs(s["expenditure"] - true_expenditure) / s["expenditure"]
        ),
        "income_error": jnp.max(jnp.abs(s["income"] - true_income) / s["income"]),
        "share_sum_error": jnp.max(jnp.abs(s["shares"].sum(1) - 1)),
        "numeraire_error": jnp.abs((s["wage"] * d["value_added"]).sum() - 1),
    }


@jax.jit
def _results(z, data):
    state = _state(z, data)
    return _observables(state, data), _equation_errors(state, data)


def solve(
    *,
    trade_elasticities,
    final_demand_shares,
    value_added_shares,
    input_output_shares,
    net_trade_value,
    tariff_rates,
    counterfactual_tariff_rates=None,
    iceberg_cost_ratio=None,
    technology_scale_ratio=None,
    z0=None,
    **solver_options,
):
    start = perf_counter()
    data = _prepare(
        trade_elasticities,
        final_demand_shares,
        value_added_shares,
        input_output_shares,
        net_trade_value,
        tariff_rates,
        counterfactual_tariff_rates,
        iceberg_cost_ratio,
        technology_scale_ratio,
    )
    n, j = data["value_added_shares"].shape
    if z0 is None:
        z0 = jnp.concatenate(
            (
                jnp.zeros(n - 1 + n * j),
                jnp.log(data["expenditure_scale"] / data["expenditure0"]).ravel(),
            )
        )
    z0 = jnp.asarray(z0, dtype=jnp.float64)
    if z0.shape != (n - 1 + 2 * n * j,) or not np.all(np.isfinite(z0)):
        raise ValueError("invalid initial coordinates")

    root = newton(residual, data, z0, **solver_options)
    values, diagnostics = _results(root["z"], data)
    values = {key: np.asarray(value) for key, value in values.items()}
    diagnostics = {key: float(value) for key, value in diagnostics.items()}
    diagnostics["residual_inf"] = float(jnp.max(jnp.abs(root["residual"])))
    tolerance = solver_options.get("tolerance", 1e-10)
    if (
        any(not np.isfinite(v) or v > max(1e-8, 100 * tolerance) for v in diagnostics.values())
        or any(not np.all(np.isfinite(v)) for v in values.values())
        or np.any(values["income"] <= 0)
    ):
        raise RuntimeError("equilibrium verification failed")
    return {
        **values,
        "diagnostics": diagnostics,
        "iterations": root["steps"],
        "z": np.asarray(root["z"]),
        "residual": np.asarray(root["residual"]),
        "history": root["history"],
        "wall_seconds": perf_counter() - start,
    }


def load_example(name="full", *, countries=4, sectors=3, seed=20260923):
    if name not in EXAMPLE_NAMES:
        raise ValueError(f"Unknown example {name!r}; choose one of {EXAMPLE_NAMES}")
    for label, size in (("countries", countries), ("sectors", sectors)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{label} must be a positive integer")
    rng = np.random.default_rng(seed)
    value_added = rng.uniform(0.7, 1.5, countries)
    final_shares = rng.uniform(0.5, 1.5, sectors)
    final_shares /= final_shares.sum()
    labor_shares = rng.uniform(0.4, 0.8, sectors)
    final_demand_shares = np.broadcast_to(final_shares, (countries, sectors)).copy()
    value_added_shares = np.broadcast_to(labor_shares, (countries, sectors)).copy()
    input_output_shares = (1 - value_added_shares[:, :, None]) * final_shares[None, None, :]
    openness = rng.uniform(0.25, 0.65, sectors)
    domestic = np.diag(value_added)
    international = np.outer(value_added, value_added) / value_added.sum()
    trade = domestic[:, :, None] * (1 - openness) + international[:, :, None] * openness
    trade *= final_shares / np.dot(final_shares, labor_shares)
    tariff = np.zeros_like(trade)
    tariff_prime = tariff.copy()
    iceberg_ratio = np.ones_like(trade)
    technology_ratio = np.ones((countries, sectors))
    if name in ("tariff", "combined", "full"):
        tariff_prime = rng.uniform(0.03, 0.12, trade.shape)
        tariff_prime[np.arange(countries), np.arange(countries), :] = 0
    if name in ("iceberg", "combined", "full"):
        iceberg_ratio = rng.uniform(0.92, 0.98, trade.shape)
        iceberg_ratio[np.arange(countries), np.arange(countries), :] = 1
    if name in ("technology", "full"):
        technology_ratio = rng.uniform(0.95, 1.10, (countries, sectors))
    trade_elasticities = np.linspace(4.0, 6.0, sectors)
    return {
        "trade_elasticities": trade_elasticities,
        "final_demand_shares": final_demand_shares,
        "value_added_shares": value_added_shares,
        "input_output_shares": input_output_shares,
        "net_trade_value": trade,
        "tariff_rates": tariff,
        "counterfactual_tariff_rates": tariff_prime,
        "iceberg_cost_ratio": iceberg_ratio,
        "technology_scale_ratio": technology_ratio,
    }
