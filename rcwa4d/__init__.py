"""RCWA4D: rigorous coupled-wave analysis for layered structures with incommensurate periodicities."""

from .beams import SummedRCWA, field_fourier_to_real, get_real_space_bases, pk_to_pte_ptm
from .fourier import convmat2D
from .smatrix import redheffer_star
from .solver import RCWA, rcwa

__version__ = "0.2.0"

__all__ = [
    "RCWA",
    "rcwa",
    "SummedRCWA",
    "convmat2D",
    "redheffer_star",
    "pk_to_pte_ptm",
    "get_real_space_bases",
    "field_fourier_to_real",
    "__version__",
]
