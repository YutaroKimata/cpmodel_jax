"""Two-block CP exact-hat equilibrium in unit costs and gross output.

All monetary data are scaled by world baseline value added. Price, wage, expenditure and income
are analytical functions of cost/output; only cost consistency and goods
markets are solved numerically. Positive variables use logarithms, without
additive epsilon corrections.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .newton import NewtonOptions, Root, SolveError, newton, require_converged


class Data(NamedTuple):
    trade_elasticities: jax.Array
    final_demand_shares: jax.Array
    value_added_shares: jax.Array
    input_output_shares: jax.Array
    log_y0: jax.Array
    value_added: jax.Array
    deficit: jax.Array
    income0: jax.Array
    expenditure0: jax.Array
    expenditure_scale: jax.Array
    tariff0: jax.Array
    tariff: jax.Array
    net_share_factor: jax.Array
    tariff_share_factor: jax.Array
    log_trade_weights: jax.Array
    log_cost_shock: jax.Array
    free: jax.Array
    money_scale: jax.Array


def _array(name, value, shape, *, positive=False, nonnegative=True):
    x = np.array(value, dtype=np.float64, copy=True)
    if x.shape != shape or not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: expected finite array of shape {shape}")
    if positive and np.any(x <= 0) or nonnegative and np.any(x < 0):
        raise ValueError(f"{name}: invalid sign")
    x.flags.writeable = False
    return x


@dataclass(frozen=True, eq=False)
class Calibration:
    """Structural parameters and baseline data; reusable across policies.

    trade_elasticities[j]: positive sector trade elasticities.
    final_demand_shares[n,j]: sector j share of final spending in country n.
    value_added_shares[n,j]: value-added share of production costs.
    input_output_shares[n,j,k]: share of input sector k in output sector j.
    net_trade_value[n,i,j]: importer n, exporter i, sector j, net of tariff revenue.
    tariff_rates[n,i,j]: baseline tariff rate, e.g. 0.10 for 10%.
    No reconciliation, clipping, or share renormalization occurs implicitly.
    """

    trade_elasticities: np.ndarray
    final_demand_shares: np.ndarray
    value_added_shares: np.ndarray
    input_output_shares: np.ndarray
    net_trade_value: np.ndarray
    tariff_rates: np.ndarray

    def __post_init__(self):
        shape = np.shape(self.net_trade_value)
        if len(shape) != 3 or shape[0] != shape[1] or min(shape) < 1:
            raise ValueError("net_trade_value must have shape (N,N,J), with N,J >= 1")
        n, _, j = shape
        shapes = {
            "trade_elasticities": (j,),
            "final_demand_shares": (n, j),
            "value_added_shares": (n, j),
            "input_output_shares": (n, j, j),
            "net_trade_value": shape,
            "tariff_rates": shape,
        }
        for name, shp in shapes.items():
            object.__setattr__(
                self,
                name,
                _array(
                    name,
                    getattr(self, name),
                    shp,
                    positive=name in ("trade_elasticities", "value_added_shares"),
                ),
            )
        if not np.allclose(self.final_demand_shares.sum(1), 1, atol=1e-10, rtol=0):
            raise ValueError("final demand shares must sum to one (1e-10 tolerance)")
        if not np.allclose(
            self.value_added_shares + self.input_output_shares.sum(2),
            1,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                "value added plus IO shares must sum to one (1e-10 tolerance)"
            )
        Y0 = self.net_trade_value.sum(0)
        E0 = (self.net_trade_value * (1 + self.tariff_rates)).sum(1)
        V0 = (self.value_added_shares * Y0).sum(1)
        D0 = self.net_trade_value.sum((1, 2)) - self.net_trade_value.sum((0, 2))
        I0 = V0 + D0 + (self.net_trade_value * self.tariff_rates).sum((1, 2))
        if any(np.any(x <= 0) or not np.all(np.isfinite(x)) for x in (Y0, E0, V0, I0)):
            raise ValueError(
                "baseline output, expenditure, value added and income must be positive"
            )

    def economy(
        self,
        *,
        counterfactual_tariff_rates=None,
        iceberg_cost_ratio=None,
        technology_scale_ratio=None,
        formulation="augmented",
    ):
        if formulation == "augmented":
            from .augmented import AugmentedEconomy

            cls = AugmentedEconomy
        elif formulation == "cost_output":
            cls = Economy
        else:
            raise ValueError("formulation must be 'augmented' or 'cost_output'")
        return cls(
            self,
            counterfactual_tariff_rates=counterfactual_tariff_rates,
            iceberg_cost_ratio=iceberg_cost_ratio,
            technology_scale_ratio=technology_scale_ratio,
        )


class Economy:
    def __init__(
        self,
        calibration,
        *,
        counterfactual_tariff_rates=None,
        iceberg_cost_ratio=None,
        technology_scale_ratio=None,
    ):
        b = calibration
        self.calibration = b
        self.num_countries, _, self.num_sectors = b.net_trade_value.shape
        n, j = self.num_countries, self.num_sectors
        tariff_rates = _array(
            "counterfactual_tariff_rates",
            b.tariff_rates
            if counterfactual_tariff_rates is None
            else counterfactual_tariff_rates,
            b.net_trade_value.shape,
        )
        iceberg = _array(
            "iceberg_cost_ratio",
            np.ones_like(b.net_trade_value)
            if iceberg_cost_ratio is None
            else iceberg_cost_ratio,
            b.net_trade_value.shape,
            positive=True,
        )
        technology = _array(
            "technology_scale_ratio",
            np.ones((n, j))
            if technology_scale_ratio is None
            else technology_scale_ratio,
            (n, j),
            positive=True,
        )
        if not np.all(iceberg[np.arange(n), np.arange(n)] == 1):
            raise ValueError("domestic iceberg_cost_ratio must be one")
        scale = (b.value_added_shares * b.net_trade_value.sum(0)).sum()
        M0 = b.net_trade_value / scale
        Y0 = M0.sum(0)
        E0 = (M0 * (1 + b.tariff_rates)).sum(1)
        V0 = (b.value_added_shares * Y0).sum(1)
        D0 = M0.sum((1, 2)) - M0.sum((0, 2))
        I0 = V0 + D0 + (b.tariff_rates * M0).sum((1, 2))
        Escale = np.maximum(
            E0,
            b.final_demand_shares * I0[:, None]
            + np.einsum("nkj,nk->nj", b.input_output_shares, Y0),
        )
        pi0 = M0 * (1 + b.tariff_rates) / E0[:, None, :]
        with np.errstate(divide="ignore"):
            logpi0 = np.log(pi0)
        logshock = np.log(technology)[None, :, :] - b.trade_elasticities * (
            np.log1p(tariff_rates) - np.log1p(b.tariff_rates) + np.log(iceberg)
        )
        # Omit the largest baseline goods market, not an arbitrary tiny country.
        self.pivot = int(np.argmax(Y0))
        free = np.delete(np.arange(n * j, dtype=np.int32), self.pivot)
        self.data = Data(
            **{
                k: jnp.asarray(v)
                for k, v in {
                    "trade_elasticities": b.trade_elasticities,
                    "final_demand_shares": b.final_demand_shares,
                    "value_added_shares": b.value_added_shares,
                    "input_output_shares": b.input_output_shares,
                    "log_y0": np.log(Y0),
                    "value_added": V0,
                    "deficit": D0,
                    "income0": I0,
                    "expenditure0": E0,
                    "expenditure_scale": Escale,
                    "tariff0": b.tariff_rates,
                    "tariff": tariff_rates,
                    "net_share_factor": 1 / (1 + tariff_rates),
                    "tariff_share_factor": tariff_rates / (1 + tariff_rates),
                    "log_trade_weights": logpi0 + logshock,
                    "log_cost_shock": logshock,
                    "free": free,
                    "money_scale": scale,
                }.items()
            }
        )
        self.size = 2 * n * j - 1

    def initial_state(self):
        return jnp.zeros(self.size, dtype=jnp.float64)

    def residual(self, z):
        return residual(z, self.data)

    @property
    def functions(self):
        return residual, outcomes, diagnostics

    def pack(self, unit_cost_ratio, output):
        c = _array(
            "unit_cost_ratio",
            unit_cost_ratio,
            (self.num_countries, self.num_sectors),
            positive=True,
        )
        y = _array(
            "output", output, (self.num_countries, self.num_sectors), positive=True
        )
        if not np.isclose(
            (self.calibration.value_added_shares * y).sum()
            / float(self.data.money_scale),
            1,
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("output must satisfy the world-value-added numeraire")
        logratio = np.log(y / float(self.data.money_scale)) - np.asarray(
            self.data.log_y0
        )
        relative = logratio.ravel() - logratio.ravel()[self.pivot]
        return jnp.concatenate(
            (
                jnp.asarray(np.log(c).ravel()),
                jnp.asarray(relative[np.asarray(self.data.free)]),
            )
        )


class State(NamedTuple):
    log_cost: jax.Array
    log_output: jax.Array
    log_price: jax.Array
    log_share_denominator: jax.Array
    shares: jax.Array
    wage: jax.Array
    output: jax.Array
    income: jax.Array
    expenditure: jax.Array
    sales: jax.Array


def state(z, d):
    """All economic identities in one pure function, with named outputs."""
    n, j = d.value_added_shares.shape
    logc = z[: n * j].reshape(n, j)
    relative_y = jnp.zeros(n * j).at[d.free].set(z[n * j :]).reshape(n, j)
    logy = d.log_y0 + relative_y
    logy -= logsumexp(jnp.log(d.value_added_shares) + logy)  # world value added = 1
    Y = jnp.exp(logy)
    wage_income = (d.value_added_shares * Y).sum(1)
    wage = wage_income / d.value_added

    logweights = d.log_trade_weights - d.trade_elasticities * logc[None, :, :]
    denominator = logsumexp(logweights, axis=1)
    logp = -denominator / d.trade_elasticities
    shares = jnp.exp(logweights - denominator[:, None, :])
    net_shares = shares * d.net_share_factor
    tariff_fraction = (shares * d.tariff_share_factor).sum(1)
    intermediate = jnp.einsum("nkj,nk->nj", d.input_output_shares, Y)
    # Solve national income/tariff feedback analytically, one scalar per country.
    retained_final_share = (d.final_demand_shares * net_shares.sum(1)).sum(1)
    income = (
        wage_income + d.deficit + (tariff_fraction * intermediate).sum(1)
    ) / retained_final_share
    expenditure = d.final_demand_shares * income[:, None] + intermediate
    sales = jnp.einsum("nij,nj->ij", net_shares, expenditure)
    return State(
        logc, logy, logp, denominator, shares, wage, Y, income, expenditure, sales
    )


@jax.jit
def residual(z, d):
    s = state(z, d)
    costs = (
        s.log_cost
        - d.value_added_shares * jnp.log(s.wage)[:, None]
        - jnp.einsum("njk,nk->nj", d.input_output_shares, s.log_price)
    )
    markets = s.log_output - jnp.log(s.sales)
    r = jnp.concatenate((costs.ravel(), markets.ravel()[d.free]))
    # Enforce economically meaningful iterates; no clipping or hidden epsilon.
    feasible = jnp.all(s.income > 0) & jnp.all(s.expenditure > 0)
    return jnp.where(feasible, r, jnp.full_like(r, jnp.nan))


def observables(s, d):
    share_multiplier = jnp.exp(
        d.log_cost_shock
        - d.trade_elasticities * s.log_cost[None, :, :]
        - s.log_share_denominator[:, None, :]
    )
    expenditure_ratio = s.expenditure / d.expenditure0
    trade_value = s.shares * s.expenditure[:, None, :]
    net_trade_value = trade_value * d.net_share_factor
    consumer_price_ratio = jnp.exp((d.final_demand_shares * s.log_price).sum(1))
    income_ratio = s.income / d.income0
    return {
        "wage_ratio": s.wage,
        "sector_price_ratio": jnp.exp(s.log_price),
        "unit_cost_ratio": jnp.exp(s.log_cost),
        "trade_share_ratio": share_multiplier,
        "expenditure_ratio": expenditure_ratio,
        "trade_value_ratio": share_multiplier * expenditure_ratio[:, None, :],
        "net_trade_value_ratio": share_multiplier
        * expenditure_ratio[:, None, :]
        * (1 + d.tariff0)
        / (1 + d.tariff),
        "output_ratio": jnp.exp(s.log_output - d.log_y0),
        "income_ratio": income_ratio,
        "consumer_price_ratio": consumer_price_ratio,
        "welfare_ratio": income_ratio / consumer_price_ratio,
        "real_wage_ratio": s.wage / consumer_price_ratio,
        "trade_shares": s.shares,
        "expenditure": s.expenditure * d.money_scale,
        "trade_value": trade_value * d.money_scale,
        "net_trade_value": net_trade_value * d.money_scale,
        "output": s.output * d.money_scale,
        "income": s.income * d.money_scale,
    }


@jax.jit
def outcomes(z, d):
    return observables(state(z, d), d)


def equation_errors(s, d):
    net_trade_value = s.shares * s.expenditure[:, None, :] * d.net_share_factor
    sales = net_trade_value.sum(0)
    tariff_revenue = (net_trade_value * d.tariff).sum((1, 2))
    # Reconstruct the original model equations independently of eliminated identities.
    true_income = s.wage * d.value_added + d.deficit + tariff_revenue
    true_expenditure = d.final_demand_shares * true_income[:, None] + jnp.einsum(
        "nkj,nk->nj", d.input_output_shares, sales
    )
    true_cost = d.value_added_shares * jnp.log(s.wage)[:, None] + jnp.einsum(
        "njk,nk->nj", d.input_output_shares, s.log_price
    )
    true_logp = (
        -logsumexp(
            d.log_trade_weights - d.trade_elasticities * s.log_cost[None, :, :], axis=1
        )
        / d.trade_elasticities
    )
    return {
        "cost_error": jnp.max(jnp.abs(s.log_cost - true_cost)),
        "price_error": jnp.max(jnp.abs(s.log_price - true_logp)),
        "goods_market_error": jnp.max(jnp.abs(s.log_output - jnp.log(sales))),
        "labor_error": jnp.max(
            jnp.abs(s.wage * d.value_added - (d.value_added_shares * sales).sum(1))
            / (s.wage * d.value_added)
        ),
        "expenditure_error": jnp.max(
            jnp.abs(s.expenditure - true_expenditure) / s.expenditure
        ),
        "income_error": jnp.max(jnp.abs(s.income - true_income) / s.income),
        "share_sum_error": jnp.max(jnp.abs(s.shares.sum(1) - 1)),
        "numeraire_error": jnp.abs((s.wage * d.value_added).sum() - 1),
    }


@jax.jit
def diagnostics(z, d):
    return dict(
        residual_inf=jnp.max(jnp.abs(residual(z, d))), **equation_errors(state(z, d), d)
    )


@dataclass
class Equilibrium:
    values: dict
    diagnostics: dict
    root: Root
    wall_seconds: float


def solve(economy, options=NewtonOptions(), *, z0=None):
    """Synchronized solve + all outcomes + full checks, including omitted market."""
    start = perf_counter()
    z = economy.initial_state() if z0 is None else jnp.asarray(z0, dtype=jnp.float64)
    if z.shape != (economy.size,) or not np.all(np.isfinite(z)):
        raise ValueError("invalid initial coordinates")
    function, output_function, diagnostic_function = economy.functions
    root = newton(function, economy.data, z, options)
    require_converged(root)
    values = {
        k: np.asarray(v) for k, v in output_function(root.z, economy.data).items()
    }
    diag = {k: float(v) for k, v in diagnostic_function(root.z, economy.data).items()}
    if (
        any(
            not np.isfinite(v) or v > max(1e-8, 100 * options.tolerance)
            for v in diag.values()
        )
        or any(not np.all(np.isfinite(v)) for v in values.values())
        or np.any(values["income"] <= 0)
    ):
        raise SolveError(root._replace(status=jnp.int32(5)))
    return Equilibrium(values, diag, root, perf_counter() - start)
