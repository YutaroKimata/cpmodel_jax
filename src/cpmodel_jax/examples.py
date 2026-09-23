"""Reproducible synthetic economies constructed from accounting identities."""

import numpy as np

from .model import Calibration

EXAMPLE_NAMES = ("no_shock", "tariff", "iceberg", "combined", "technology", "full")


def _example_inputs(name="full", *, countries=4, sectors=3, seed=20260923):
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
    # Both margins equal national value added. Sector-specific home bias keeps
    # bilateral flows balanced without solving a calibration system.
    openness = rng.uniform(0.25, 0.65, sectors)
    domestic = np.diag(value_added)
    international = np.outer(value_added, value_added) / value_added.sum()
    trade = domestic[:, :, None] * (1 - openness) + international[:, :, None] * openness
    # Rank-one IO structure gives the exact expenditure/output multiplier.
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
    # Baseline elasticities are identical across scenarios with the same seed.
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


def load_example(name="full", *, formulation="augmented", countries=4, sectors=3, seed=20260923):
    """Construct a fresh synthetic counterfactual; defaults to four countries.

    The baseline satisfies the model's accounting identities analytically.
    Names are in :data:`EXAMPLE_NAMES`. These are numerical examples, not
    empirical estimates. All inputs are generated locally by NumPy.
    """
    arrays = _example_inputs(name, countries=countries, sectors=sectors, seed=seed)
    baseline = Calibration(
        **{
            key: arrays[key]
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
            key: arrays[key]
            for key in (
                "counterfactual_tariff_rates",
                "iceberg_cost_ratio",
                "technology_scale_ratio",
            )
        },
    )
