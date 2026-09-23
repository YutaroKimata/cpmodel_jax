"""Caliendo–Parro exact-hat equilibria with float64 matrix-free Newton–Krylov.

Importing this package enables JAX's process-wide 64-bit configuration.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .augmented import AugmentedEconomy
from .examples import EXAMPLE_NAMES, load_example
from .model import Calibration, Equilibrium, solve
from .model import Economy as CostOutputEconomy
from .newton import NewtonOptions, SolveError

__version__ = "0.4.0"

__all__ = [
    "AugmentedEconomy",
    "Calibration",
    "CostOutputEconomy",
    "Equilibrium",
    "EXAMPLE_NAMES",
    "NewtonOptions",
    "SolveError",
    "load_example",
    "solve",
]
