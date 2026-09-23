"""A complete two-country, one-sector example with descriptive input names."""

import numpy as np

from cpmodel_jax import Calibration, solve


def main():
    # Bilateral values include domestic sales, measured before tariffs.
    baseline = Calibration(
        trade_elasticities=np.array([4.0]),
        final_demand_shares=np.ones((2, 1)),
        value_added_shares=np.ones((2, 1)),
        input_output_shares=np.zeros((2, 1, 1)),
        net_trade_value=np.array([[2.0, 1.0], [1.0, 2.0]])[:, :, None],
        tariff_rates=np.zeros((2, 2, 1)),
    )
    new_tariffs = baseline.tariff_rates.copy()
    new_tariffs[0, 1, 0] = 0.10  # Country 0 taxes imports from country 1 at 10%.
    iceberg_costs = np.array([[1.0, 0.95], [0.95, 1.0]])[:, :, None]
    economy = baseline.economy(
        counterfactual_tariff_rates=new_tariffs,
        iceberg_cost_ratio=iceberg_costs,
        technology_scale_ratio=np.array([[1.05], [1.0]]),
    )
    result = solve(economy)
    print("Welfare percentage changes:", 100 * (result.values["welfare_ratio"] - 1))
    print("Maximum equation error:", max(result.diagnostics.values()))


if __name__ == "__main__":
    main()
