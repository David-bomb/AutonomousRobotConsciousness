"""Доехать по карте памяти до первого нового наблюдения с учётом заездов на сервис."""
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count

from .frontier_automaton import _arrive, _frontiers
from .frontier_reserve import _lengths, _optimistic
from .grid import cell, destination, locations, wave
from .interface import Grid, Move, Point
from .pilot_automaton import Mode
from .sight_automaton import with_wall
from .street_trunk import StreetTrunkReserve


@dataclass(frozen=True)
class Arrival:
    point: Point
    length: int
    fuel: int
    health: int
    first: Move


@lru_cache(maxsize=256)
def first_events(memory: Grid, origin: Point, terminals: frozenset[Point]):
    return tuple((p, length, move) for p, length, move in wave(memory, [origin], stop_at=terminals)
                 if length and p in terminals)


def window_arrivals(view):
    """Пути обрываются у первого нового окна или финиша, проход через сервис пополняет запас."""
    memory = view.memory
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    exits = frozenset(_frontiers(memory)) | {goal}
    stations = {p for kind in "FR" for p in locations(memory, kind)}
    terminals = exits | stations
    serial = count()
    pending = [(0, next(serial), view.position, view.fuel, view.health, None)]
    reached = {}
    arrivals = {}
    while pending:
        length, _, point, fuel, health, first = heappop(pending)
        saved = reached.setdefault(point, [])
        # Обход идёт в порядке длины пути. Более позднее прибытие с обоими
        # запасами не выше прежних не улучшит ни одного продолжения.
        if any(f >= fuel and h >= health for f, h in saved):
            continue
        saved.append((fuel, health))
        if first is not None and point in exits:
            options = arrivals.setdefault(point, [])
            options[:] = [a for a in options if a.length < length or a.fuel > fuel or a.health > health]
            options.append(Arrival(point, length, fuel, health, first))
            continue
        for p, distance, move in first_events(memory, point, terminals):
            stocks = _arrive("B" if p == goal else cell(memory, *p), distance, fuel, health)
            if stocks is not None:
                heappush(pending, (length + distance, next(serial), p, *stocks, first or move))
    for options in arrivals.values():
        yield from options


class GoalWindow(StreetTrunkReserve):
    """В последнем выезде едет к достижимому первому окну с кратчайшим продолжением до цели."""
    reserve_in_rank = False

    def remaining_length(self, memory, goal, point, field):
        return field.get(point, float("inf"))

    def pick(self, view, goal, arrivals):
        remaining = _lengths(_optimistic(view.memory), goal)
        def rank(arrival):
            lower = self.remaining_length(view.memory, goal, arrival.point, remaining)
            stock = min(arrival.fuel, arrival.health) if self.reserve_in_rank else 0
            return arrival.point != goal, lower - stock, arrival.length, arrival.point, -arrival.fuel, -arrival.health

        return min(arrivals, key=rank, default=None)

    def _reconsider_sortie(self, view, goal, stops, planned):
        best = self.pick(view, goal, window_arrivals(view))
        if best is None:
            return None
        self.activity = "Первое новое наблюдение на пути к финишу"
        self._fallback.mode = Mode.GOAL
        self._fallback.returning = False
        self._fallback.last_lengths = {Mode.FUEL: view.fuel_distance, Mode.REPAIR: view.repair_distance}
        return best.first


class MarginWindow(GoalWindow):
    """Учитывает также меньший из двух запасов при прибытии к выбранному окну."""
    reserve_in_rank = True


@lru_cache(maxsize=256)
def length_after_wall(memory, goal, point):
    """Наибольшая длина пути к цели при закрытии одной неизвестной клетки окна."""
    grid = _optimistic(memory)
    field = _lengths(grid, goal)
    if point not in field:
        return float("inf")
    length = field[point]
    # Стена вне одного найденного кратчайшего пути сохраняет этот путь.
    # Для диагонали угловые клетки тоже должны оставаться открытыми.
    used = set()
    current = point
    while current != goal:
        move, nxt = next((m, q) for m in Move
                         if (q := destination(grid, current, m)) is not None and field[q] == field[current] - 1)
        used.add(nxt)
        dx, dy = move.value
        if dx and dy:
            used.update(((current[0] + dx, current[1]), (current[0], current[1] + dy)))
        current = nxt
    for x, y in used:
        if ((x, y) != goal and memory[y][x] == "?"
                and max(abs(x - point[0]), abs(y - point[1])) <= 1):
            blocked = with_wall(grid, (x, y))
            length = max(length, _lengths(blocked, goal).get(point, float("inf")))
    return length


class WallWindow(MarginWindow):
    """Сохраняет запас на обход одной неожиданной стены в следующем окне."""
    def remaining_length(self, memory, goal, point, field):
        return length_after_wall(memory, goal, point)
