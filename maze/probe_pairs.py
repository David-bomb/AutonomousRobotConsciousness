"""При запасе на одну ошибку проверить, различит ли вторая станция оставшиеся ходы."""
from dataclasses import replace
from functools import lru_cache

import numpy as np

from .interface import Move
from .shared_trunk import trunk_votes
from .specific_paths import LastReturnPaths
from .street_trunk import alternatives


@lru_cache(maxsize=96)
def paired_directions(view, readings, old, prior, sites):
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    result = np.zeros((len(Move), len(Move)))
    for weight, grid in alternatives(view.memory, goal):
        pair = trunk_votes(replace(view, memory=grid), readings, old, prior, sites, True,
                           require_match=True, joint_first=True)
        if pair is not None:
            result += weight * pair
    return result


def two_attempts(pairs, options, first):
    """Один ход и, при ошибке, одна повторная попытка после наблюдения второго датчика."""
    remaining = [i for i in options if i != first]
    success = pairs[first].sum()
    if not remaining:
        return success, 0.
    other_down = pairs[remaining, first]
    totals = pairs[remaining].sum(axis=1)
    # На дереве с единственной ближайшей станцией её длина пути уменьшится
    # ровно на своём направлении. Эти два исхода распознаются после хода.
    recovery = other_down.max() + (totals - other_down).max()
    return success + recovery, recovery - totals.max()


class ProbePairs(LastReturnPaths):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        own, other = (view.fuel, view.health) if kind == 0 else (view.health, view.fuel)
        if len(options) < 3 or readings[kind] is None or min(own, other-1) - readings[kind] not in (2, 3):
            return parent
        pairs = paired_directions(view, readings, old, prior, sites)
        matrix = pairs.T if kind else pairs
        moves = tuple(Move)
        legal = [moves.index(m) for m in options]
        values = {m: two_attempts(matrix, legal, moves.index(m)) for m in options}
        best = max(options, key=lambda m: (values[m][0], m == parent))
        scale = matrix.sum()
        if values[best][1] <= 1e-12 * scale or values[best][0] <= values[parent][0] + 1e-12 * scale:
            return parent
        return best
