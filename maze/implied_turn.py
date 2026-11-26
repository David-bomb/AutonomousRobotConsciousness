"""Проверка условия «следующий ход сокращает длину» по уже полученным данным."""
from functools import lru_cache
from math import exp, log

from .cell_hypotheses import occupancy
from .deduced_roads import deductions
from .grid import destination
from .street_trunk import StreetTrunk


@lru_cache(maxsize=256)
def support(memory, origin, neighbor, goal, current, proposed, sites):
    # Текущие длины измерены. Предложенные значения существуют только
    # внутри этой гипотезы. Среда не вызывается.
    inferred = deductions(memory, neighbor, goal, proposed, origin, current, sites, True)
    if inferred is None:
        return 0.0
    if inferred == memory:
        return 1.0
    _, chances = occupancy(memory)
    evidence = 0.
    for i, (seen, forced) in enumerate(zip("".join(memory), "".join(inferred))):
        if seen == "?" and forced != "?":
            evidence += log(chances[i] if forced == "." else 1 - chances[i])
    return exp(evidence)


class ImpliedTurn(StreetTrunk):
    def direction_votes(self, view, readings, old, prior, sites):
        rows = super().direction_votes(view, readings, old, prior, sites)
        if not all(d is not None and d > 0 for d in readings):
            return rows
        x, y = view.position
        goal = x + view.dx, y + view.dy
        result = [dict(row) for row in rows]
        for kind, row in enumerate(result):
            for move in row:
                q = destination(view.memory, view.position, move)
                if not q or not row[move]:
                    continue
                proposed = [None, None]
                proposed[kind] = readings[kind] - 1
                one = support(view.memory, view.position, q, goal, readings, tuple(proposed), sites)
                if one:
                    both = support(view.memory, view.position, q, goal, readings,
                                   tuple(d - 1 for d in readings), sites)
                    row[move] *= (one + both) / 2
                else:
                    row[move] = 0.
        return result if all(any(row.values()) for row in result) else rows
