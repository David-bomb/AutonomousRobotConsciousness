"""Оценить, сможет ли робот распознать следующие повороты до выбранной станции."""
from functools import lru_cache
from math import inf, isfinite

from .conditional_mission import possible_worlds
from .frontier_reserve import _optimistic
from .grid import destination
from .interface import Move
from .proven_approach import service_field
from .street_trunk import StreetTrunkReserve


def continuation_chance(neighbors, field, lower):
    """После убывающего шага выбирать поровну среди всё ещё возможных выходов."""
    @lru_cache(None)
    def chance(previous, point):
        if field[point] == 0:
            return 1.
        # Сосед предыдущей клетки не может быть на два хода ближе к станции.
        shortcuts = neighbors(previous)
        options = [q for q in neighbors(point) if q != previous and q not in shortcuts
                   and lower[q] <= field[point] - 1]
        if not options:
            return 0.
        return sum(chance(point, q) for q in options if field[q] == field[point] - 1) / len(options)
    return chance


def local_roads(opened, width, height):
    @lru_cache(None)
    def neighbors(p):
        x, y = p % width, p // width
        out = set()
        for move in Move:
            dx, dy = move.value
            q = p + dx + width * dy
            if (0 <= x + dx < width and 0 <= y + dy < height and opened[q]
                    and (not dx or not dy or opened[p + dx] and opened[p + width * dy])):
                out.add(q)
        return out
    return neighbors


def survival_factors(view, readings, old, prior, sites, options, kind, exact=False):
    width, height = len(view.memory[0]), len(view.memory)
    def index(point):
        return point[0] + width * point[1]
    origin = index(view.position)
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    markers, worlds = possible_worlds(view.memory, goal)
    ids = [[j for j, (k, _) in enumerate(markers) if k == symbol] for symbol in "FR"]
    known = service_field(_optimistic(view.memory), sites[kind])
    lower = [known.get((p % width, p // width), inf) for p in range(width * height)]
    starts = {m: index(destination(view.memory, view.position, m)) for m in options}
    support = dict.fromkeys(options, 0.)
    arrivals = support.copy()
    for opened, fields, _ in worlds:
        nearest = [fields[group].min(axis=0) if group else None for group in ids]
        selected = nearest[kind]
        if selected is None or not isfinite(selected[origin]):
            continue
        error = delta = 0.
        for i, field in enumerate(nearest):
            if readings[i] is None or field is None:
                continue
            if not isfinite(field[origin]):
                error = inf
                break
            error += abs(field[origin] - readings[i])
            if prior is not None and old[i] is not None:
                delta += abs(field[index(prior)] - field[origin] - (old[i] - readings[i]))
        if not isfinite(error + delta) or (exact and error + delta != 0):
            continue
        weight = 1. / ((1. + error) * (1. + delta)) ** 2
        neighbors = local_roads(opened, width, height)
        chance = continuation_chance(neighbors, selected, lower)
        for m, q in starts.items():
            if selected[q] == selected[origin] - 1:
                support[m] += weight
                arrivals[m] += weight * chance(origin, q)
        chance.cache_clear()
    total = sum(support.values())
    if not total:
        return {}
    pooled = sum(arrivals.values()) / total
    return {m: arrivals[m] / support[m] if support[m] else pooled for m in options}


class LegSurvival(StreetTrunkReserve):
    exact = False

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        score = self.direction_votes(view, readings, old, prior, sites)[kind]
        stock = (view.fuel, view.health)[kind]
        factors = {}
        if readings[kind] is not None and 0 <= stock - readings[kind] < 2:
            factors = survival_factors(view, readings, old, prior, sites, options, kind, self.exact)
        return max(options, key=lambda m: (score[m] * factors.get(m, 1.), score[m], m == default))


class MatchingSurvival(LegSurvival):
    """Учитывает трудность следующих поворотов только в сетях с точными показаниями."""
    exact = True
