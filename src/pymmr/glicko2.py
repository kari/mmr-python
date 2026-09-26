"""Glicko-2 rating algorithm.

An implementation of Mark E. Glickman's Glicko-2 system, see
http://www.glicko.net/glicko/glicko2.pdf for the reference paper.

Ratings are immutable: :func:`rate` returns a new :class:`Glicko2Rating`
and never modifies its arguments.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import NamedTuple

# Constants for converting between the Glicko scale (center 1500) and the
# internal Glicko-2 scale.
_CENTER = 1500.0
_SCALE = 173.7178

# Convergence tolerance for the volatility iteration.
_TOLERANCE = 1e-6

__all__ = ["Glicko2Rating", "Match", "expected", "rate"]


@dataclass(frozen=True, slots=True)
class Glicko2Rating:
    """A player's rating: ``rating``, rating deviation ``rd`` and ``volatility``."""

    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Return the confidence interval for the rating with coverage 1 - ``alpha``.

        By default returns the 95% confidence interval as a
        ``(lower, upper)`` tuple.
        """
        z = NormalDist().inv_cdf(1 - alpha / 2)
        return (self.rating - z * self.rd, self.rating + z * self.rd)


class Match(NamedTuple):
    """A game played by the rated player.

    The ``score`` is 1.0 for a win, 0.5 for a draw and 0.0 for a loss.
    """

    opponent: Glicko2Rating
    score: float


def _g(phi: float) -> float:
    return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)


def _expected_score(mu: float, mu_opponent: float, phi_opponent: float) -> float:
    return 1 / (1 + math.exp(-_g(phi_opponent) * (mu - mu_opponent)))


def rate(
    rating: Glicko2Rating,
    matches: Sequence[Match],
    *,
    tau: float = 0.5,
) -> Glicko2Rating:
    """Return a new rating after playing ``matches`` in one rating period.

    ``tau`` constrains how quickly the volatility may change; the Glicko-2
    paper recommends a value between 0.3 and 1.2 but does not offer a default.

    An empty ``matches`` list counts as a rating period without competition:
    the rating and volatility are unchanged and the rating deviation grows.
    """
    mu = (rating.rating - _CENTER) / _SCALE
    phi = rating.rd / _SCALE

    if not matches:
        phi_star = math.hypot(phi, rating.volatility)
        return replace(rating, rd=_SCALE * phi_star)

    mu_js = [(m.opponent.rating - _CENTER) / _SCALE for m in matches]
    phi_js = [m.opponent.rd / _SCALE for m in matches]

    v_inv = sum(
        _g(pj) ** 2 * _expected_score(mu, mj, pj) * (1 - _expected_score(mu, mj, pj))
        for mj, pj in zip(mu_js, phi_js, strict=False)
    )
    v = 1 / v_inv
    delta = v * sum(
        _g(pj) * (m.score - _expected_score(mu, mj, pj))
        for mj, pj, m in zip(mu_js, phi_js, matches, strict=False)
    )

    a = math.log(rating.volatility**2)

    def f(x: float) -> float:
        # The f(x) from the Glicko-2 paper, solved with the Illinois algorithm.
        exp_x = math.exp(x)
        return (
            exp_x * (delta**2 - phi**2 - v - exp_x) / (2 * (phi**2 + v + exp_x) ** 2)
            - (x - a) / tau**2
        )

    A = a
    if delta**2 > phi**2 + v:
        B = math.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    f_A, f_B = f(A), f(B)
    while abs(B - A) > _TOLERANCE:
        C = A + (A - B) * f_A / (f_B - f_A)
        f_C = f(C)
        if f_C * f_B < 0:
            A, f_A = B, f_B
        else:
            f_A /= 2
        B, f_B = C, f_C

    sigma_new = math.exp(A / 2)
    phi_star = math.hypot(phi, sigma_new)
    phi_new = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
    mu_new = mu + phi_new**2 * sum(
        _g(pj) * (m.score - _expected_score(mu, mj, pj))
        for mj, pj, m in zip(mu_js, phi_js, matches, strict=False)
    )

    return Glicko2Rating(_SCALE * mu_new + _CENTER, _SCALE * phi_new, sigma_new)


def expected(player: Glicko2Rating, opponent: Glicko2Rating) -> float:
    """Expected score of a match based on the players' ratings and RDs.

    Uses the formula from the original Glicko paper,
    http://www.glicko.net/glicko/glicko.pdf.
    """
    q = math.log(10) / 400
    g = 1 / math.sqrt(1 + 3 * q**2 * (player.rd**2 + opponent.rd**2) / math.pi**2)
    return 1 / (1 + 10 ** (-g * (player.rating - opponent.rating) / 400))
