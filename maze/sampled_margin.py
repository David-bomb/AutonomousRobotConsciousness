"""Оценить запас на прямой выезд к финишу в нескольких возможных сетях дорог."""
from math import isfinite

from .exit_flow import CorridorFlow
from .last_mission import mission_fields


class SampledMargin(CorridorFlow):
    def pick(self, view, goal, arrivals):
        choices = tuple(arrivals)
        if len({a.point for a in choices}) < 2 or any(a.point == goal for a in choices):
            return super().pick(view, goal, choices)
        width = len(view.memory[0])
        index = lambda p: p[1] * width + p[0]
        origin = index(view.position)
        sites, worlds = mission_fields(view.memory, goal)
        groups = [[i for i, (kind, _) in enumerate(sites) if kind == symbol] for symbol in "FR"]
        votes = dict.fromkeys(choices, 0.)
        for fields in worlds:
            error = sum(abs(min(fields[i, origin] for i in ids) - measured)
                        for ids, measured in zip(groups, (view.fuel_distance, view.repair_distance))
                        if ids and measured is not None)
            if not isfinite(error):
                continue
            slack = {a: min(a.fuel, a.health) - fields[-1, index(a.point)] for a in choices}
            best = max(slack.values())
            if best < 0:
                continue
            shortest = min(a.length for a in choices if slack[a] == best)
            winners = [a for a in choices if slack[a] == best and a.length == shortest]
            weight = 1 / ((1 + error)**2 * len(winners))
            for a in winners:
                votes[a] += weight
        best = max(votes.values())
        return super().pick(view, goal, [a for a in choices if votes[a] == best])
