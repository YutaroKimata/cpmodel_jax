# Model and numerical method

## Mathematical specification

Axes: `net_trade_value[n,i,j]` is importer/exporter/sector; `input_output_shares[n,j,k]` is
country/output-sector/input-sector. Value-added shares and IO shares sum to one;
final expenditure shares sum to one, both to a `1e-10` validation tolerance.
Zero bilateral flows are permitted and remain zero. Country-sector baseline
output/expenditure and country baseline income/value added must be positive.
Tariffs must be nonnegative; iceberg and technology hats are strictly positive.
Domestic iceberg hats are one. Share corrections, baseline reconciliation,
and negative-value clipping are never performed implicitly.

All money is internally divided by world baseline value added. Let `V0`, `D`,
`Y0`, `E0`, `I0` denote quantities in these units; `sum(V0)=1`. The closure is
fixed nominal baseline trade deficits, with `sum(V0*w_hat)=1`. This is the model
closure and normalization. Results restore the original
monetary units. This implements exact-hat counterfactuals, not levels from uncalibrated
technology/endowment primitives.

Precompute log bilateral weights from baseline expenditure shares and policy:

\[
a_{nij}=\log\pi^0_{nij}+\log\hat\lambda_{ij}
             -\theta_j\log\hat\kappa_{nij}.
\]

Then, for any current unit cost `c_hat`,

\[
L_{nj}=\operatorname{logsumexp}_i(a_{nij}-\theta_j\log\hat c_{ij}),
\quad \log\hat p^*_{nj}=-L_{nj}/\theta_j,
\quad \pi'_{nij}=\exp(a_{nij}-\theta_j\log\hat c_{ij}-L_{nj}).
\]

Zeros in `pi0` become `-inf` log weights, not positive pseudo-trade. Unit costs
are Cobb--Douglas in log wages/prices, with no `log(exp(x)+epsilon)` detour.

### Equilibrium equations

Unknowns: `N-1` relative log wages, `N*J` log prices and `N*J` log expenditure
ratios. Weighted log normalization imposes the numeraire. Unit costs are
`value_added_shares*log(w_hat) + input_output_shares @ log(p_hat)`. From current shares/expenditure, compute
net sales `Y`, tariff revenue `R`, and household income `I=w_hat*V0+D+R`.
The residuals are relative labor-market clearing, price consistency, and
`E - final_demand_shares*I - gamma_transpose @ Y`, with a fixed accounting-based row scale.
There is no price iteration, no expenditure matrix and no expenditure solve.

Trade shares are normalized while Newton moves off equilibrium. Price
consistency is separately checked, so this is an equivalent equilibrium
system, not a change to trade demand. All original market/accounting
conditions are checked on the final result.

## Numerical execution and scope

Newton and backtracking use ordinary Python `for` loops. Each Newton direction
uses `jax.linearize` and GMRES inside a JIT-compiled function. Residual evaluation
and final outcomes/checks are also compiled. There is no full Jacobian,
expenditure LU, or dense fallback.

GMRES convergence is checked by explicitly evaluating the linear residual.
A backtracking search reduces the step until the residual norm decreases.
The final solution must pass every original economic equation, including the
omitted labor condition. Numerical failures raise `RuntimeError`.

Python reads scalar convergence information each iteration. This costs some
synchronization compared with a fully compiled loop, but keeps the control flow
readable. Restarted GMRES stores Krylov vectors and a small Hessenberg matrix;
bilateral arrays still require `O(N^2 J)` storage.

`wall_seconds` includes input validation/preparation, the solve, outcome
extraction, and economic checks. Compilation is included on the first use of a
new shape or static GMRES setting. Histories record nonlinear and linear errors,
step lengths, and backtracking counts, not Krylov iteration counts.

This is CPU-validated float64 code. GPU performance and differentiation through
the equilibrium solution have not been implemented or validated. Disconnected
trade networks can have additional unidentified nominal scales. No automatic
continuation, preconditioner, or fixed-point fallback is included.
