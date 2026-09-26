"""Tests for :mod:`pymmr.glicko2`."""

import math

import pytest

from pymmr import Glicko2Rating
from pymmr.glicko2 import Match, expected, rate


def test_rate_glicko2_paper_example() -> None:
    """Example from the Glicko-2 paper."""
    new = rate(
        Glicko2Rating(1500, 200),
        [
            Match(Glicko2Rating(1400, 30), 1),
            Match(Glicko2Rating(1550, 100), 0),
            Match(Glicko2Rating(1700, 300), 0),
        ],
        tau=0.5,
    )
    assert new.rating == pytest.approx(1464.06, abs=0.01)
    assert new.rd == pytest.approx(151.52, abs=0.01)
    assert new.volatility == pytest.approx(0.05999, abs=0.00001)


def test_rate_one_match_at_a_time() -> None:
    """Single-match rating periods approach the paper example."""
    p = Glicko2Rating(1500, 200)
    p = rate(p, [Match(Glicko2Rating(1400, 30), 1)], tau=0.5)
    p = rate(p, [Match(Glicko2Rating(1550, 100), 0)], tau=0.5)
    p = rate(p, [Match(Glicko2Rating(1700, 300), 0)], tau=0.5)
    assert p.rating == pytest.approx(1464.06, abs=1)
    assert p.rd == pytest.approx(151.52, abs=0.5)
    assert p.volatility == pytest.approx(0.05999, abs=0.00001)


def test_rate_draw_against_equal_opponent() -> None:
    """A draw between equal players keeps the rating and shrinks the RD."""
    new = rate(Glicko2Rating(1500, 200), [Match(Glicko2Rating(1500, 200), 0.5)])
    assert new.rating == pytest.approx(1500, abs=0.5)
    assert new.rd < 200


def test_rate_does_not_modify_input() -> None:
    """Updating returns a new rating; the input rating is untouched."""
    rating = Glicko2Rating(1500, 200)
    rate(rating, [Match(Glicko2Rating(1400, 30), 1)], tau=0.5)
    assert rating == Glicko2Rating(1500, 200)


def test_rate_empty_period_grows_rd() -> None:
    """A player who did not compete keeps rating and volatility, RD grows."""
    new = rate(Glicko2Rating(1500, 200), [])
    assert new.rating == 1500
    assert new.rd == pytest.approx(math.sqrt(200**2 + (0.06 * 173.7178) ** 2), abs=0.01)
    assert new.volatility == 0.06


def test_rate_large_upset() -> None:
    """A result far from expectation takes the upper bound of the paper's step 5."""
    new = rate(Glicko2Rating(1500, 60), [Match(Glicko2Rating(2500, 60), 1)])
    assert new.rating > 1500
    assert new.volatility > 0.06


def test_rate_volatility_lower_bound_extension() -> None:
    """Exercise the paper's step 5 fallback that extends the lower bound.

    Reachable only with extreme parameters: many drawn matches make v tiny
    relative to the volatility, and a large tau keeps the extension negative.
    """
    opponents = [Match(Glicko2Rating(1500, 1), 0.5) for _ in range(1000)]
    new = rate(Glicko2Rating(1500, 1, volatility=2.0), opponents, tau=5.0)
    assert new.rating == pytest.approx(1500, abs=0.01)
    assert new.rd > 0
    assert new.volatility < 2.0


def test_expected() -> None:
    """Values from the original Glicko paper."""
    assert expected(Glicko2Rating(1400, 80), Glicko2Rating(1500, 150)) == pytest.approx(
        0.376, abs=0.001
    )
    assert expected(
        Glicko2Rating(1500, 350), Glicko2Rating(1500, 350)
    ) == pytest.approx(0.5, abs=0.1)
    assert expected(
        Glicko2Rating(1400, 350), Glicko2Rating(1500, 350)
    ) == pytest.approx(0.423, abs=0.001)


def test_confidence_interval() -> None:
    """95% and 90% confidence intervals for a rating."""
    assert Glicko2Rating(1500, 30).confidence_interval() == pytest.approx(
        (1441, 1559), abs=1
    )
    assert Glicko2Rating(1500, 30).confidence_interval(alpha=0.1) == pytest.approx(
        (1450.7, 1549.3), abs=0.5
    )
