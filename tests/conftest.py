"""Helpers for independently generated synthetic economies."""

from cpmodel_jax import Calibration
from cpmodel_jax.examples import _example_inputs


def example_inputs(name="full", countries=4, sectors=3):
    return _example_inputs(name, countries=countries, sectors=sectors)


def prepare(inputs, formulation="cost_output"):
    baseline = Calibration(
        **{
            key: inputs[key]
            for key in (
                "trade_elasticities",
                "final_demand_shares",
                "value_added_shares",
                "input_output_shares",
                "net_trade_value",
                "tariff_rates",
            )
        }
    )
    return baseline.economy(
        formulation=formulation,
        **{
            key: inputs[key]
            for key in (
                "counterfactual_tariff_rates",
                "iceberg_cost_ratio",
                "technology_scale_ratio",
            )
        },
    )
