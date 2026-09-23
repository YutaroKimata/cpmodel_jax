"""Public API and synthetic examples work from an installed distribution."""

from importlib.metadata import version

import jax
import numpy as np
import pytest

import cpmodel_jax
from cpmodel_jax import AugmentedEconomy, load_example, solve


def test_version_and_precision():
    assert cpmodel_jax.__version__ == version("cpmodel-jax")
    assert jax.config.x64_enabled


@pytest.mark.parametrize("name", cpmodel_jax.EXAMPLE_NAMES)
def test_installed_examples(name):
    economy = load_example(name)
    assert isinstance(economy, AugmentedEconomy)
    result = solve(economy)
    assert max(result.diagnostics.values()) < 1e-8
    assert np.all(result.values["welfare_ratio"] > 0)
    again = load_example(name)
    np.testing.assert_array_equal(
        again.calibration.net_trade_value, economy.calibration.net_trade_value
    )


def test_unknown_example():
    with pytest.raises(ValueError, match="Unknown example"):
        load_example("not-an-example")
