import pytest

import weng11a


def test_update():
    ratings = [[(25, 25 / 3**2)], [(25, 25 / 3**2)]]
    ranks = [1, 2]
    new_ranks = w.update(ratings, ranks)
    assert new_ranks[0][0][0] > new_ranks[1][0][0]  # winner has higher mu


def test_probs():
    ratings = [[(25, 25 / 3**2)], [(25, 25 / 3**2)]]
    assert w.probs(ratings) == pytest.approx([0.5, 0.5])
