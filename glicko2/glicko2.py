"""
Glicko-2 implementation in Python

For reference see http://www.glicko.net/glicko/glicko2.pdf
"""

import math
from typing import Self, Tuple

from scipy.stats import norm, logistic


class Player:
    """Player class to hold a player's rating, RD, volatility and "true" skill (for simulation purposes)"""

    def __init__(
        self,
        rating: float = 1500,
        rd: float = 350,
        volatility: float = 0.06,
        skill: float | None = None,
    ) -> None:
        self.rating = rating
        self.rd = rd
        self.volatility = volatility
        if skill is not None:
            self.skill = skill
        else:
            self.skill = norm.rvs(rating, rd)

    def update(
        self, opponents: list[Self] | Self, s: list[float] | float, tau: float
    ) -> None:
        """Updates a player's rating against opponents with outcomes `s`
        (1 = win, 0.5 = tie, 0 = loss)

        Glicko-2 paper recommends `tau` is in range 0.3 - 1.2 but does not offer a default.
        """
        mu = (self.rating - 1500) / 173.7178
        phi = self.rd / 173.7178
        sigma = self.volatility

        if isinstance(opponents, Player):
            opponents = [opponents]

        if not isinstance(s, list):
            s = [s]

        mu_j = list(map(lambda o: (o.rating - 1500) / 173.7178, opponents))
        phi_j = list(map(lambda o: o.rd / 173.7178, opponents))

        def g(phi: float) -> float:
            return 1 / math.sqrt(1 + 3 * math.pow(phi, 2) / math.pow(math.pi, 2))

        def E(mu: float, mu_i: float, phi_i: float) -> float:
            return 1 / (1 + math.exp(-g(phi_i) * (mu - mu_i)))

        def f(x):
            return math.exp(x) * (
                math.pow(Delta, 2) - math.pow(phi, 2) - v - math.exp(x)
            ) / (2 * math.pow(math.pow(phi, 2) + v + math.exp(x), 2)) - (
                x - a
            ) / math.pow(
                tau, 2
            )

        v = math.pow(
            sum(
                map(
                    lambda phi_i, mu_i: math.pow(g(phi_i), 2)
                    * E(mu, mu_i, phi_i)
                    * (1 - E(mu, mu_i, phi_i)),
                    phi_j,
                    mu_j,
                )
            ),
            -1,
        )

        Delta = v * sum(
            map(
                lambda phi_i, mu_i, s_i: g(phi_i) * (s_i - E(mu, mu_i, phi_i)),
                phi_j,
                mu_j,
                s,
            )
        )

        a = math.log(math.pow(sigma, 2))
        epsilon = 0.000001

        A = a
        if math.pow(Delta, 2) > math.pow(phi, 2) + v:
            B = math.log(math.pow(Delta, 2) - math.pow(phi, 2) - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k = k + 1
            B = a - k * tau

        f_A = f(A)
        f_B = f(B)
        while abs(B - A) > epsilon:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = f(C)
            if f_C * f_B < 0:
                A = B
                f_A = f_B
            else:
                f_A = f_A / 2
            B = C
            f_B = f_C

        sigma_new = math.exp(A / 2)

        phi_star = math.sqrt(math.pow(phi, 2) + math.pow(sigma_new, 2))
        phi_new = 1 / math.sqrt(1 / math.pow(phi_star, 2) + 1 / v)
        mu_new = mu + math.pow(phi_new, 2) * sum(
            map(
                lambda phi_i, mu_i, s_i: g(phi_i) * (s_i - E(mu, mu_i, phi_i)),
                phi_j,
                mu_j,
                s,
            )
        )

        self.rating = 173.7178 * mu_new + 1500
        self.rd = 173.7178 * phi_new
        self.volatility = sigma_new

    @staticmethod
    def expected_outcome(player1: "Player", player2: "Player") -> float:
        """Return expected outcome of a match based on player ratings.

        From original Glicko paper, http://www.glicko.net/glicko/glicko.pdf
        """
        q = math.log(10) / 400

        def g(RD):
            return 1 / math.sqrt(
                1 + 3 * math.pow(q, 2) * math.pow(RD, 2) / math.pow(math.pi, 2)
            )

        def E(r_i, r_j, RD_i, RD_j):
            return 1 / (
                1
                + math.pow(
                    10,
                    (
                        -g(math.sqrt(math.pow(RD_i, 2) + math.pow(RD_j, 2)))
                        * (r_i - r_j)
                        / 400
                    ),
                )
            )

        return E(player1.rating, player2.rating, player1.rd, player2.rd)

    def ci(self, alpha=0.05) -> Tuple[float, float]:
        """Calculate the confidence interval for a player's rating with coverage 1-`alpha`.
        By default calculates the 95% confidence interval."""

        q = norm.ppf(1 - alpha / 2).item(0)
        return (self.rating - q * self.rd, self.rating + q * self.rd)

    @staticmethod
    def true_expected_outcome(
        player1: "Player", player2: "Player", s: float = 350
    ) -> float:
        """Calculate expected outcome using "hidden, true" skill attribute.

        Uses a logistic distribution, other option would be to use a normal distribution. `s` = standard deviation of the scoring system.

        Using logistic gives higher probabilities to "improbable" events and thus might be more suitable for real-life scenarios.
        """

        return logistic.cdf(player1.skill - player2.skill, scale=s).item(0)
