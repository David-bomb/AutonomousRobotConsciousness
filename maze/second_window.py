"""При выборе выезда учитывать, можно ли после первого осмотра доехать до другого окна."""
from functools import lru_cache
from itertools import product

from .cell_hypotheses import occupancy
from .first_window import MarginWindow, first_events
from .frontier_automaton import _frontiers
from .frontier_reserve import _lengths, _optimistic
from .grid import cell, destination
from .interface import Move
from .simple_exit_paths import SimpleExitPaths
from .specific_paths import LastReturnPaths, SpecificPaths


@lru_cache(maxsize=512)
def closed_probability(memory, point, goal):
    """Шанс, что окно не откроет ни одного ранее неизвестного хода из этой клетки."""
    _, chance = occupancy(memory)
    width = len(memory[0])
    x, y = point
    requirements = []
    optimistic = _optimistic(memory)
    for move in Move:
        if destination(memory, point, move) is not None:
            continue
        q = destination(optimistic, point, move)
        if q is None:
            continue
        dx, dy = move.value
        cells = {q, (x + dx, y), (x, y + dy)} if dx and dy else {q}
        requirements.append({p for p in cells if cell(memory, *p) == "?"})
    if not requirements:
        return 1.
    hidden = sorted(set.union(*requirements))
    probs = [1. if p == goal else chance[p[1]*width+p[0]] for p in hidden]
    total = 0.
    for bits in product((False, True), repeat=len(hidden)):
        opened = {p for p, bit in zip(hidden, bits) if bit}
        if any(group <= opened for group in requirements):
            continue
        mass = 1.
        for prob, bit in zip(probs, bits):
            mass *= prob if bit else 1. - prob
        total += mass
    return total


class SecondWindow(SpecificPaths):
    discount_closed = True
    closed_chance = staticmethod(closed_probability)

    def sortie_curves(self, view, goal, points):
        return self.probabilities(view.memory, goal, points)

    def chance(self, view, goal, point, fuel, health, curves):
        return curves[point][min(fuel,health)]

    def backup(self, view, goal, a, curves):
        stock = min(a.fuel, a.health)
        lengths = _lengths(view.memory, a.point)
        alternate = max((self.chance(view,goal,p,a.fuel-lengths[p],a.health-lengths[p],curves) for p in curves
                         if p != a.point and lengths.get(p, 101) < stock), default=0.)
        discount = self.closed_chance(view.memory, a.point, goal) if self.discount_closed else 1.
        return discount * alternate

    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        parent = super().pick(view, goal, choices)
        if len({a.point for a in choices}) < 2 or any(a.point == goal for a in choices):
            return parent
        # Сохраняем решение о заезде, если сервис изменил выбор первого окна.
        if parent != SimpleExitPaths.pick(self, view, goal, choices):
            return parent
        points = tuple(sorted(p for p, _, _ in first_events(view.memory, view.position,
                                                              frozenset(_frontiers(view.memory)))))
        curves = self.sortie_curves(view, goal, points)
        scores = []
        for a in choices:
            scores.append(self.chance(view,goal,a.point,a.fuel,a.health,curves) + self.backup(view, goal, a, curves))
        best = max(scores)
        finalists = [a for a, score in zip(choices, scores) if score == best]
        return parent if parent in finalists else MarginWindow.pick(self, view, goal, finalists)


class OpenSecondWindow(SecondWindow):
    """Верхняя оценка, когда неверный первый выезд распознаётся сразу по новому окну."""
    discount_closed = False


class ReturnWindows(SecondWindow, LastReturnPaths):
    """Пересмотр направления после последней пробы и выбор выезда с запасным окном."""
