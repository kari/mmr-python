import pytest

from glicko2 import Player


def test_update_glicko2():
    """Tests the update function, example from Glicko-2 paper"""
    p = Player(1500, 200)
    p.update([Player(1400, 30), Player(1550, 100), Player(1700, 300)], [1, 0, 0], 0.5)
    assert p.rating == pytest.approx(1464.06, abs=0.01)
    assert p.rd == pytest.approx(151.52, abs=0.01)
    assert p.volatility == pytest.approx(0.05999, abs=0.00001)


def test_update_glicko2_iterative():
    """Tests the update function using the iterative version of the Glicko-2 paper example.
    Should return something close to the above"""
    p = Player(1500, 200)
    p.update(Player(1400, 30), 1, 0.5)
    p.update(Player(1550, 100), 0, 0.5)
    p.update(Player(1700, 300), 0, 0.5)
    assert p.rating == pytest.approx(1464.06, abs=1)
    assert p.rd == pytest.approx(151.52, abs=0.5)
    assert p.volatility == pytest.approx(0.05999, abs=0.00001)


def test_expected_outcome():
    assert Player.expected_outcome(
        Player(1400, 80), Player(1500, 150)
    ) == pytest.approx(0.376, abs=0.001) # Example from the Glicko paper
    assert Player.expected_outcome(
        Player(1500, 350), Player(1500, 350)
    ) == pytest.approx(0.5, abs=0.1)
    assert Player.expected_outcome(
        Player(1400, 350), Player(1500, 350)
    ) == pytest.approx(0.423, abs=0.001)


def test_true_expected_outcome():
    assert Player.true_expected_outcome(
        Player(1500, 350, 0.06, 1500), Player(1500, 350, 0.06, 1500)
    ) == pytest.approx(0.5)
    assert Player.true_expected_outcome(
        Player(1400, 80, 0.06, 1400), Player(1500, 150, 0.06, 1500)
    ) == pytest.approx(0.429, abs=0.001)


def test_ci():
    p = Player(1500, 30)
    assert p.ci() == pytest.approx((1441, 1559), abs=1)
