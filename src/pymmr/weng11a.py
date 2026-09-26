"""Bayesian approximation method for online ranking.

An implementation of Algorithm 1 (Bradley-Terry model, full pairing) from
"A Bayesian Approximation Method for Online Ranking" by Weng and Lin (2011),
http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf — the same
model popularized by Microsoft's TrueSkill.

Ratings are immutable: :func:`rate` returns new ratings and never modifies
its arguments.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Skill uncertainty of the scoring system: beta = 25/6.
BETA_SQUARED = (25 / 6) ** 2

# Lower bound for the variance update multiplier, keeping the variance positive.
KAPPA = 0.0001

__all__ = ["WengLinRating", "probs", "rate"]


@dataclass(frozen=True, slots=True)
class WengLinRating:
    """A player's rating: skill estimate ``mu`` and skill deviation ``sigma``.

    The defaults follow the paper: ``mu = 25``, ``sigma = 25/3``, with skill
    uncertainty ``BETA_SQUARED = (25/6)**2``.
    """

    mu: float = 25.0
    sigma: float = 25 / 3


def _team_stats(
    teams: Sequence[Sequence[WengLinRating]],
) -> tuple[list[float], list[float]]:
    """Team mean skill and total variance for each team."""
    means = [sum(player.mu for player in team) for team in teams]
    variances = [sum(player.sigma**2 for player in team) for team in teams]
    return means, variances


def probs(teams: Sequence[Sequence[WengLinRating]]) -> list[float]:
    """Return the winning probability of each team implied by current ratings.

    Returns one probability per team; the list sums to 1. For two teams the
    probability equals the expected score of a single match between them.
    """
    means, variances = _team_stats(teams)
    k = len(teams)

    unnormalized = []
    for i in range(k):
        p = 1.0
        for q in range(k):
            if q == i:
                continue
            c = math.sqrt(variances[i] + variances[q] + 2 * BETA_SQUARED)
            p *= 1 / (1 + math.exp(-(means[i] - means[q]) / c))
        unnormalized.append(p)

    total = sum(unnormalized)
    return [p / total for p in unnormalized]


def rate(
    teams: Sequence[Sequence[WengLinRating]],
    ranks: Sequence[float],
) -> list[list[WengLinRating]]:
    """Return new ratings after all teams have played one rating period.

    ``teams`` holds one list of players per team and ``ranks`` gives each
    team's finishing position: a lower rank means a better result (rank 1
    beats rank 2) and equal ranks count as a draw. Raises ``ValueError`` if
    ``ranks`` does not have one rank per team.
    """
    if len(ranks) != len(teams):
        raise ValueError("need one rank per team")

    means, variances = _team_stats(teams)
    k = len(teams)

    new_teams = []
    for i in range(k):
        omega = 0.0
        delta = 0.0
        for q in range(k):
            if q == i:
                continue
            c = math.sqrt(variances[i] + variances[q] + 2 * BETA_SQUARED)
            p_iq = 1 / (1 + math.exp(-(means[i] - means[q]) / c))
            p_qi = 1 - p_iq

            if ranks[q] > ranks[i]:
                s = 1.0
            elif ranks[q] < ranks[i]:
                s = 0.0
            else:
                s = 0.5

            gamma = math.sqrt(variances[i]) / c
            omega += variances[i] / c * (s - p_iq)
            delta += gamma * variances[i] / c**2 * p_iq * p_qi

        new_team = []
        for player in teams[i]:
            share = player.sigma**2 / variances[i]
            new_mu = player.mu + share * omega
            new_variance = player.sigma**2 * max(1 - share * delta, KAPPA)
            new_team.append(WengLinRating(new_mu, math.sqrt(new_variance)))
        new_teams.append(new_team)

    return new_teams
