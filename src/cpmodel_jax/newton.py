"""Model-independent, fully compiled, matrix-free Newton--GMRES.

Only the initial and final state cross the Python/device boundary. No full
Jacobian is constructed. Trace buffers have fixed shapes for compilation.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres


@dataclass(frozen=True)
class NewtonOptions:
    tolerance: float = 1e-10
    max_steps: int = 60
    restart: int = 60
    max_cycles: int = 30
    max_backtracks: int = 25
    max_log_step: float = 1.0
    armijo: float = 1e-4

    def __post_init__(self):
        for name in ("max_steps", "restart", "max_cycles", "max_backtracks"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(self.max_log_step) or self.max_log_step <= 0:
            raise ValueError("max_log_step must be finite and positive")
        if not 0 < self.armijo < 1:
            raise ValueError("armijo must lie between zero and one")


class Root(NamedTuple):
    z: jax.Array
    residual: jax.Array
    steps: jax.Array
    status: jax.Array  # 0 converged, 1 limit, 2 nonfinite, 3 GMRES, 4 line search
    history: jax.Array


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


@partial(jax.jit, static_argnames=("function", "options"))
def newton(function, data, z0, options=NewtonOptions()):
    """Pure array API, suitable for repeated policies of a fixed shape."""

    def norm_inf(r):
        return jnp.max(jnp.abs(r))

    def condition(state):
        z, r, iteration, status, history = state
        return (status == 0) & (iteration < options.max_steps) & (norm_inf(r) > options.tolerance)

    def body(state):
        z, r, iteration, status, history = state
        _, Jv = jax.linearize(lambda x: function(x, data), z)
        eta = jnp.clip(jnp.sqrt(norm_inf(r)), 1e-10, 0.01)
        linear_atol = 1e-14
        residual_norm = jnp.linalg.norm(r)
        step, info = gmres(
            Jv,
            -r,
            tol=eta,
            atol=linear_atol,
            restart=min(options.restart, z.size),
            maxiter=options.max_cycles,
            solve_method="incremental",
        )
        linear_error = jnp.linalg.norm(Jv(step) + r) / residual_norm
        linear_ok = (info == 0) & jnp.all(jnp.isfinite(step)) & jnp.isfinite(linear_error)
        # GMRES stops at max(relative target, absolute target). The independent
        # check must honor both, especially when the nonlinear residual is small.
        linear_tolerance = jnp.maximum(eta, linear_atol / residual_norm)
        linear_ok &= linear_error <= jnp.maximum(2 * linear_tolerance, 1e-8)
        length = jnp.minimum(
            1.0, options.max_log_step / jnp.maximum(jnp.max(jnp.abs(step)), 1e-300)
        )

        def search(_):
            trial_r = function(z + length * step, data)

            def accepted(a, rr):
                return jnp.all(jnp.isfinite(rr)) & (
                    jnp.linalg.norm(rr) <= (1 - options.armijo * a) * jnp.linalg.norm(r)
                )

            def search_condition(s):
                a, rr, backtracks = s
                return (~accepted(a, rr)) & (backtracks < options.max_backtracks - 1)

            def shrink(s):
                a, rr, backtracks = s
                a = a * 0.5
                return a, function(z + a * step, data), backtracks + 1

            a, rr, backtracks = jax.lax.while_loop(
                search_condition, shrink, (length, trial_r, jnp.int32(0))
            )
            return a, rr, backtracks, accepted(a, rr)

        a, rr, backtracks, accepted = jax.lax.cond(
            linear_ok, search, lambda _: (jnp.float64(0), r, jnp.int32(0), jnp.bool_(False)), None
        )
        status = jnp.where(~linear_ok, 3, jnp.where(~accepted, 4, 0))
        history = history.at[iteration].set(
            jnp.array(
                [
                    norm_inf(r),
                    jnp.linalg.norm(r),
                    eta,
                    linear_error,
                    a,
                    backtracks,
                    norm_inf(rr),
                    info,
                ],
                dtype=jnp.float64,
            )
        )
        return (
            jnp.where(status == 0, z + a * step, z),
            jnp.where(status == 0, rr, r),
            iteration + 1,
            status,
            history,
        )

    r0 = function(z0, data)
    state = (
        z0,
        r0,
        jnp.int32(0),
        jnp.where(jnp.all(jnp.isfinite(r0)), 0, 2),
        jnp.full((options.max_steps, len(HISTORY_COLUMNS)), jnp.nan),
    )
    z, r, steps, status, history = jax.lax.while_loop(condition, body, state)
    status = jnp.where((status == 0) & (norm_inf(r) > options.tolerance), 1, status)
    return Root(z, r, steps, status, history)


class SolveError(RuntimeError):
    def __init__(self, root):
        reasons = {
            1: "Newton iteration limit",
            2: "nonfinite initial residual",
            3: "GMRES failed linear residual check",
            4: "line search failed",
        }
        super().__init__(reasons.get(int(root.status), "equilibrium verification failed"))
        self.root = root


def require_converged(root):
    if int(root.status) != 0:
        raise SolveError(root)
