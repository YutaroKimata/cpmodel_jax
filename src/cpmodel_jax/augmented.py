"""Sparse-dependency CP formulation: wages, prices and expenditure.

Keeping these economically meaningful auxiliary variables can condition the
Krylov problem better than analytically eliminating them. The same prepared
data, numerical backend, outcomes, and full checks serve both formulations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .model import Economy, State, _array, equation_errors, observables


class AugmentedEconomy(Economy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = self.num_countries - 1 + 2 * self.num_countries * self.num_sectors

    def initial_state(self):
        return jnp.concatenate(
            (
                jnp.zeros(self.num_countries - 1 + self.num_countries * self.num_sectors),
                jnp.log(self.data.expenditure_scale / self.data.expenditure0).ravel(),
            )
        )

    def residual(self, z):
        return residual(z, self.data)

    @property
    def functions(self):
        return residual, outcomes, diagnostics

    def pack(self, wage_ratio, sector_price_ratio, expenditure):
        w = _array("wage_ratio", wage_ratio, (self.num_countries,), positive=True)
        p = _array(
            "sector_price_ratio",
            sector_price_ratio,
            (self.num_countries, self.num_sectors),
            positive=True,
        )
        E = _array("expenditure", expenditure, p.shape, positive=True)
        if not np.isclose(np.dot(w, np.asarray(self.data.value_added)), 1, atol=1e-8, rtol=0):
            raise ValueError("wages must satisfy the value-added numeraire")
        return jnp.asarray(
            np.concatenate(
                (
                    np.log(w[:-1] / w[-1]),
                    np.log(p).ravel(),
                    np.log(
                        E / float(self.data.money_scale) / np.asarray(self.data.expenditure0)
                    ).ravel(),
                )
            )
        )


def state(z, d):
    n, j = d.value_added_shares.shape
    u = jnp.concatenate((z[: n - 1], jnp.zeros(1)))
    logw = u - logsumexp(jnp.log(d.value_added) + u)
    logp = z[n - 1 : n - 1 + n * j].reshape(n, j)
    E = d.expenditure0 * jnp.exp(z[n - 1 + n * j :].reshape(n, j))
    logc = d.value_added_shares * logw[:, None] + jnp.einsum(
        "njk,nk->nj", d.input_output_shares, logp
    )
    logweights = d.log_trade_weights - d.trade_elasticities * logc[None, :, :]
    denominator = logsumexp(logweights, axis=1)
    shares = jnp.exp(logweights - denominator[:, None, :])
    net_trade_value = shares * E[:, None, :] * d.net_share_factor
    Y = net_trade_value.sum(0)
    w = jnp.exp(logw)
    income = w * d.value_added + d.deficit + (net_trade_value * d.tariff).sum((1, 2))
    return State(logc, jnp.log(Y), logp, denominator, shares, w, Y, income, E, Y)


@jax.jit
def residual(z, d):
    s = state(z, d)
    logtarget_w = jnp.log((d.value_added_shares * s.output).sum(1) / d.value_added)
    fw = z[: s.wage.size - 1] - (logtarget_w[:-1] - logtarget_w[-1])
    fp = s.log_price + s.log_share_denominator / d.trade_elasticities
    demand = d.final_demand_shares * s.income[:, None] + jnp.einsum(
        "nkj,nk->nj", d.input_output_shares, s.output
    )
    fe = (s.expenditure - demand) / d.expenditure_scale
    r = jnp.concatenate((fw, fp.ravel(), fe.ravel()))
    return jnp.where(jnp.all(s.income > 0), r, jnp.full_like(r, jnp.nan))


@jax.jit
def outcomes(z, d):
    return observables(state(z, d), d)


@jax.jit
def diagnostics(z, d):
    return dict(residual_inf=jnp.max(jnp.abs(residual(z, d))), **equation_errors(state(z, d), d))
