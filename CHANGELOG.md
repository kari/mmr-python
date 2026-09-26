# Changelog

## 0.1.0 (2026-09-26)

Initial packaged release. Breaking changes relative to the original scripts:

- Functional API with immutable `Glicko2Rating` / `WengLinRating`
  dataclasses: updating a rating returns a new one, nothing is mutated.
- No dependencies: scipy is replaced by the standard library
  (`statistics.NormalDist`, closed-form logistic CDF).
- The simulation helpers (`sample_true_skill`, `expected_from_skill`) and the
  auto-sampled `skill` attribute were removed; use `random.gauss(r.rating,
  r.rd)` and `1 / (1 + exp(-(a - b) / scale))` directly in simulation code.
- weng11a ratings are named `(mu, sigma)` fields instead of `(mean, variance)`
  tuples, and the default `sigma` is corrected to 25/3 (the old default was a
  variance of 25/9 due to an operator precedence slip).
- Glicko-2: a rating period without matches now grows the rating deviation,
  as described in the paper.
- weng11a win probabilities use an overflow-safe logistic form, so very
  large rating differences no longer produce nan.
- Developer tooling follows current best practices: mypy in strict mode,
  expanded ruff lint rules (docstrings, annotations), and strict pytest
  settings.
