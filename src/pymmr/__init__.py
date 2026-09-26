"""Skill rating algorithms: Glicko-2 and a Bayesian approximation method."""

from . import glicko2, weng11a
from .glicko2 import Glicko2Rating, Match
from .weng11a import WengLinRating

__all__ = [
    "Glicko2Rating",
    "Match",
    "WengLinRating",
    "glicko2",
    "weng11a",
]
