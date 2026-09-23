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
monetary units. Neither formulation solves the model in levels from uncalibrated
technology/endowment primitives; both implement its exact-hat counterfactual.

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

### Default: augmented equations

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

### Alternative: costs and output only

Unknowns: `N*J` log unit costs and `N*J-1` relative log output ratios. For free
coordinates `u`, set the pivot coordinate to zero and use

\[
Y_{nj}=\frac{Y^0_{nj}e^{u_{nj}}}
                 {\sum_{ik}\beta_{ik}Y^0_{ik}e^{u_{ik}}},
\quad \hat w_n=\frac{\sum_j\beta_{nj}Y_{nj}}{V^0_n}.
\]

Price and shares come directly from the log-sum-exp above. Intermediate demand
is `B_nj=sum_k gamma_nkj*Y_nk`. Let

\[
t_{nj}=\sum_i\frac{\tau'_{nij}}{1+\tau'_{nij}}\pi'_{nij}.
\]

The equations `E=final_demand_shares*I+B` and `I=w_hat*V0+D+sum_j t_j E_j` give an **exact
scalar elimination per country**:

\[
I_n=\frac{\sum_j\beta_{nj}Y_{nj}+D_n+\sum_jt_{nj}B_{nj}}
           {1-\sum_j\alpha_{nj}t_{nj}},\qquad E_{nj}=\alpha_{nj}I_n+B_{nj}.
\]

The code evaluates the denominator as
`sum_j alpha_nj * sum_i pi_nij/(1+tau_nij)` for numerical stability.
Only two sets of equations remain:

\[
F_c=\log\hat c-\beta\log\hat w-\gamma\log\hat p=0,
\quad F_Y=\log Y-\log\left(\sum_n\pi'_{nij}E_{nj}/(1+\tau'_{nij})\right)=0.
\]

One goods equation is redundant: the accounting identities imply
`sum(sales)=sum(Y)` for any candidate state (including off equilibrium), since
`sum(D)=0`. We omit the largest baseline-output sector and check **all** goods
markets after solving. Choosing the largest sector avoids amplifying a small
omitted absolute discrepancy into the relative error of a tiny sector.

This reduces the dimension by N, but couples equations through the eliminated
variables. It is available as an alternative formulation for research. Fewer
unknowns do not guarantee fewer Krylov operations.
Krylov counts are not exposed by the JAX API, so this is an explanation of the
possible mechanism, not a measured iteration-count attribution.

## Numerical execution and scope

Both Newton and backtracking run inside compiled `jax.lax.while_loop` loops.
Each Newton step uses `jax.linearize` and GMRES with an independently checked
linear residual. There is no full Jacobian, expenditure LU, or dense fallback.
Restarted GMRES stores Krylov vectors and a small Hessenberg matrix; matrix-free
does not mean memory-free. Dense bilateral data still require `O(N^2 J)` storage.

No Python/device synchronization occurs inside a nonlinear iteration. Result
extraction, full economic checks, and final synchronization are included in
`wall_seconds`. Compilation is included on the first use of a new shape/static
option. Changing options can trigger compilation; policy arrays of the same
shape use dynamic data. Histories store nonlinear errors, linear solve errors,
step lengths and backtracking counts, not unavailable Krylov iteration counts.

This is CPU-validated, float64 code. No GPU performance claim or implicit
gradient-through-equilibrium API is provided. In particular, JAX's dynamic
`while_loop` is not a reverse-mode differentiable equilibrium layer; Jv inside
Newton is a different capability. Disconnected trade networks can have
additional unidentified nominal scales. No automatic continuation,
preconditioner, rebalancing, or hidden fixed-point fallback is included.

