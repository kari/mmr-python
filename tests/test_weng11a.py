"""Tests for :mod:`pymmr.weng11a`."""

import pytest

from pymmr import WengLinRating
from pymmr.weng11a import probs, rate


def test_rate_winner_gains() -> None:
    """The winner's mu rises, the loser's falls, uncertainty shrinks."""
    teams = [[WengLinRating()], [WengLinRating()]]
    new = rate(teams, [1, 2])
    assert new[0][0].mu > new[1][0].mu  # winner has higher mu
    assert new[0][0].mu > WengLinRating().mu
    assert new[1][0].mu < WengLinRating().mu
    assert new[0][0].sigma < WengLinRating().sigma  # uncertainty shrinks


def test_rate_two_player_teams() -> None:
    """All players on a winning team gain, all on the losing team lose."""
    teams = [[WengLinRating(), WengLinRating()], [WengLinRating(), WengLinRating()]]
    new = rate(teams, [1, 2])
    assert all(player.mu > WengLinRating().mu for player in new[0])
    assert all(player.mu < WengLinRating().mu for player in new[1])
    assert all(player.sigma < WengLinRating().sigma for team in new for player in team)


def test_rate_does_not_modify_input() -> None:
    """Updating returns new ratings; the input teams are untouched."""
    teams = [[WengLinRating()], [WengLinRating()]]
    rate(teams, [1, 2])
    assert teams == [[WengLinRating()], [WengLinRating()]]


def test_rate_tie_keeps_mu() -> None:
    """Two equally ranked teams draw: means stay, uncertainty shrinks."""
    teams = [[WengLinRating()], [WengLinRating()]]
    new = rate(teams, [1, 1])
    assert new[0][0].mu == pytest.approx(WengLinRating().mu)
    assert new[1][0].mu == pytest.approx(WengLinRating().mu)
    assert new[0][0].sigma < WengLinRating().sigma


def test_rate_three_teams_ordering() -> None:
    """Finishing order is reflected in the updated mus."""
    teams = [[WengLinRating()] for _ in range(3)]
    new = rate(teams, [1, 2, 3])
    assert new[0][0].mu > new[1][0].mu > new[2][0].mu


def test_rate_single_team_is_unchanged() -> None:
    """With no opponents there is nothing to update against."""
    teams = [[WengLinRating()]]
    assert rate(teams, [1]) == teams


def test_rate_rank_count_mismatch() -> None:
    """A rank count that does not match the team count raises ValueError."""
    with pytest.raises(ValueError):
        rate([[WengLinRating()], [WengLinRating()]], [1])


def test_probs() -> None:
    """Two equally rated teams each win half the time."""
    teams = [[WengLinRating()], [WengLinRating()]]
    assert probs(teams) == pytest.approx([0.5, 0.5])


def test_probs_sums_to_one() -> None:
    """Winning probabilities over all teams sum to 1."""
    teams = [[WengLinRating(30)], [WengLinRating(25)], [WengLinRating(20)]]
    assert sum(probs(teams)) == pytest.approx(1.0)


def test_probs_extreme_ratings_no_overflow() -> None:
    """The old exp-ratio form overflowed to nan for large rating differences."""
    teams = [[WengLinRating(5000)], [WengLinRating(25)]]
    p = probs(teams)
    assert p[0] == pytest.approx(1.0, abs=1e-12)
    assert p[1] == pytest.approx(0.0, abs=1e-12)
