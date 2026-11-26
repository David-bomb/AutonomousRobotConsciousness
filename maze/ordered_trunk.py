"""Учитывать выбранный порядок двух остановок при оценке начального участка дороги."""
from dataclasses import replace
from functools import lru_cache
from math import inf

from .frontier_reserve import _lengths, _optimistic
from .shared_trunk import trunk_votes
from .specific_paths import SpecificPaths
from .street_trunk import alternatives


@lru_cache(maxsize=96)
def ordered_votes(view, readings, old, prior, sites, allow_goal):
    stocks = view.fuel, view.health
    if any(d is None or not 0 < d <= stock for d, stock in zip(readings, stocks)):
        return None
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    lower = _lengths(_optimistic(view.memory), goal)
    start = _lengths(_optimistic(view.memory), view.position)
    budgets = [True, True]
    if allow_goal:
        for k in range(2):
            budgets[k] = not any(start.get(p, inf) <= readings[k] and
                                 lower.get(p, inf) <= min(100, stocks[1-k]-readings[k]) for p in sites[k])
    rows = [{}, {}]
    for weight, memory in alternatives(view.memory, goal):
        for k in range(2):
            result = trunk_votes(replace(view, memory=memory), readings, old, prior, sites, budgets[k],
                                 require_match=True, first_kind=k)
            if result:
                for move, score in result[k].items():
                    rows[k][move] = rows[k].get(move, 0.) + weight*score
    return rows


class OrderedTrunk(SpecificPaths):
    allow_goal = False

    def direction_votes(self, view, readings, old, prior, sites):
        rows = ordered_votes(view, readings, old, prior, sites, self.allow_goal)
        if rows and all(any(row.values()) for row in rows):
            return rows
        parent = super().direction_votes(view, readings, old, prior, sites)
        if rows is None:
            return parent
        return [row if any(row.values()) else original for row, original in zip(rows, parent)]


class FinishTrunk(OrderedTrunk):
    allow_goal = True
