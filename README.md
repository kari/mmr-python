# pymmr — Skill rating algorithms for Python

Pure-Python implementations of two skill rating systems:

- [Glicko-2](http://www.glicko.net/glicko/glicko2.pdf) by Mark E. Glickman (1999)
- [A Bayesian Approximation Method for Online Ranking](http://jmlr.csail.mit.edu/papers/volume12/weng11a/weng11a.pdf) by Weng and Lin (2011)

No dependencies beyond the standard library. Ratings are immutable
dataclasses: updating a rating always returns a new one and never modifies
its arguments. The code is fully type-annotated and checked with mypy in
strict mode.

## Installation

Requires Python 3.11+.

You can install this straight from the repository:

```bash
uv add git+https://github.com/kari/mmr-python
```

## Usage — Glicko-2

```python
from mmr import Glicko2Rating, Match
from mmr.glicko2 import expected, rate

player = Glicko2Rating()  # rating 1500, rd 350, volatility 0.06

player = rate(
    player,
    [
        Match(Glicko2Rating(1400, 30), 1.0),  # win
        Match(Glicko2Rating(1550, 100), 0.5),  # draw
        Match(Glicko2Rating(1700, 300), 0.0),  # loss
    ],
    tau=0.5,
)

# A rating period without matches grows the rating deviation:
player = rate(player, [])

expected(player, Glicko2Rating(1500, 350))  # expected score of a match
player.confidence_interval()  # 95% confidence interval
```

`tau` controls how quickly the volatility may change; the Glicko-2 paper
recommends a value between 0.3 and 1.2.

## Usage — Bayesian approximation (weng11a)

Ratings are `(mu, sigma)` pairs; teams are lists of players. `ranks` gives
each team's finishing position, lower is better, equal ranks count as draws.

```python
from mmr import WengLinRating
from mmr.weng11a import probs, rate

teams = [[WengLinRating()], [WengLinRating()]]  # one player per team

rate(teams, ranks=[1, 2])  # team 0 finished ahead of team 1
# [[WengLinRating(mu=27.6, sigma=8.07)], [WengLinRating(mu=22.4, sigma=8.07)]]

probs(teams)  # [0.5, 0.5] — win probability of each team, sums to 1
```

## Development

```bash
uv sync                                  # create .venv, install dev dependencies
uv run pytest                            # run the test suite
uv run pytest --cov=mmr                  # ... with a coverage report
uv run ruff check . && uv run ruff format --check .
uv run mypy                              # type-check with mypy --strict
```

## License

[MPL-2.0](LICENSE)
