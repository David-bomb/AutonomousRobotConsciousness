"""Эксперимент: подтверждённое обслуживание и разведка по памяти клеток."""
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count

from .grid import cell, destination, locations
from .interface import Grid, Move, Observation, Point
from .pilot_automaton import Mode, separation
from .resource_automaton import Automaton as ResourceAutomaton
from .resource_automaton import known_paths as _paths
from .resource_automaton import possible_directions, service_order


CONFIRMED_GOAL = "Подтверждённый путь к финишу"


@lru_cache(maxsize=96)
def _frontiers(grid: Grid) -> tuple[Point, ...]:
    return tuple((x, y) for y, row in enumerate(grid) for x, kind in enumerate(row)
                 if kind not in "#?"
                 and any(cell(grid, x + dx, y + dy) == "?" for dx, dy in (m.value for m in Move)))


@lru_cache(maxsize=96)
def _escape_lengths(grid: Grid, stops):
    # Симметричные дороги: длина от точки до станции равна обратной длине.
    distances = {}
    for target, kind in stops:
        for p, (d, _) in _paths(grid, target).items():
            if d:
                distances.setdefault(p, []).append((target, kind, d))
    return distances


def _arrive(kind, distance, fuel, health):
    if distance > min(fuel, health):
        return None
    f = 100 if kind == "F" else fuel - distance
    h = 100 if kind == "R" else health - distance
    return (f, h) if min(f, h) > 0 or kind == "B" else None


def _service_states(grid, origin, f, h, stops):
    serial = count()
    queue = [(0, next(serial), origin, f, h, None)]
    seen = {}
    while queue:
        length, _, point, f, h, first = heappop(queue)
        stocks = seen.setdefault(point, [])
        if any(old_f >= f and old_h >= h for old_f, old_h in stocks):
            continue
        stocks.append((f, h))
        roads = _paths(grid, point)
        yield length, f, h, first, roads
        for target, kind in stops:
            distance, move = roads.get(target, (0, None))
            if not distance or (after := _arrive(kind, distance, f, h)) is None:
                continue
            heappush(queue, (length + distance, next(serial), target, *after, first or move))


@lru_cache(maxsize=128)
def _survey(view: Observation, goal: Point, stops, stalled: bool = False):
    grid = view.memory
    frontiers = _frontiers(grid)
    escape = _escape_lengths(grid, stops)
    # Кандидаты оцениваются отдельно от физики: путь к подцели и от неё
    # рассчитывается только по открытым дорогам.
    best_goal = best_frontier = None
    for length, fuel, health, first, roads in _service_states(
            grid, view.position, view.fuel, view.health, stops):
        d, move = roads.get(goal, (0, None))
        if d and _arrive("B", d, fuel, health) is not None:
            row = (length + d, first or move)
            if best_goal is None or row[0] < best_goal[0]:
                best_goal = row
        if best_goal and length >= best_goal[0]:
            break
        for point in frontiers:
            if point == view.position or point not in roads:
                continue
            d, move = roads[point]
            after = _arrive(".", d, fuel, health)
            if after is None:
                continue
            # Требуем подтвердить не только ближайшую остановку, но и
            # переход к сервису второго типа. Иначе можно заправиться и встать.
            safe = False
            for site, kind, back in escape.get(point, ()):
                service_stocks = _arrive(kind, back, *after)
                if service_stocks is None:
                    continue
                if any(other != kind and _arrive(other, onward, *service_stocks)
                       for _, other, onward in escape.get(site, ())):
                    safe = True
                    break
            remaining = separation(point, goal)
            if not safe and stalled:
                # Запасной короткий выход в сторону близкого финиша: после
                # возврата к известному сервису ещё остаётся резерв на обход.
                safe = any(_arrive(kind, back, *after) and min(after) > 2 * (back + remaining)
                           for _, kind, back in escape.get(point, ()))
            if not safe:
                continue
            key = (length + d + remaining, remaining, point)
            if best_frontier is None or key < best_frontier[0]:
                best_frontier = key, first or move
    if best_goal:
        return CONFIRMED_GOAL, best_goal[1]
    if best_frontier:
        return "Разведка с подтверждённым путём к сервису", best_frontier[1]
    return None


def _near_goal(view, goal):
    # Один ход разведки, если цель находится сразу за окном восприятия.
    # Это ограниченная гипотеза о выходе, а не гарантия проходимости.
    if separation(view.position, goal) != 2 or min(view.fuel, view.health) <= 4:
        return None
    if cell(view.memory, *goal) != "?":
        return None
    optimistic = tuple(row.replace("?", ".") for row in view.memory)
    for move in Move:
        point = destination(view.memory, view.position, move)
        if point is not None and any(destination(optimistic, point, step) == goal
                                     for step in Move):
            return move
    return None


class FrontierAutomaton:
    """Тот же вход Observation; никакого доступа к полной карте или Environment."""
    STREAK_CAP = 2

    def __init__(self):
        self._fallback = self._make_fallback()
        self.activity = "Планирование"
        self._unseen = None
        self._service_streak = 0

    def _make_fallback(self):
        return ResourceAutomaton()

    def choose(self, view: Observation) -> Move:
        if view.last_move is None:
            self._fallback = self._make_fallback()
        # Число неизвестных клеток монотонно: не нужны копия карты,
        # история координат или пометки посещённых ветвей.
        unseen = sum(row.count("?") for row in view.memory)
        if view.last_move is None or unseen != self._unseen:
            self._service_streak = 0
            self._unseen = unseen
        elif view.moved and cell(view.memory, *view.position) in "FR":
            self._service_streak = min(self.STREAK_CAP, self._service_streak + 1)
        x, y = view.position
        goal = x + view.dx, y + view.dy
        stops = tuple((p, kind) for kind in "FR" for p in locations(view.memory, kind))
        plan = _survey(view, goal, stops, self._service_streak >= 2)
        if plan is None or plan[0] != CONFIRMED_GOAL:
            probe = _near_goal(view, goal)
            if probe is not None:
                plan = "Проверка близкого финиша", probe
        if plan is None:
            plan = self._extra_plan(view, goal, stops)
        if plan is not None:
            self._fallback.mode = Mode.GOAL
            self._fallback.returning = False
            self._fallback.last_lengths = {Mode.FUEL: view.fuel_distance,
                                           Mode.REPAIR: view.repair_distance}
            self.activity, move = plan
            return move

        return self._fallback_move(view, goal, stops)

    def _extra_plan(self, view, goal, stops):
        return None

    def _fallback_move(self, view, goal, stops):
        active = self._fallback.mode
        returning = self._fallback.returning
        service = service_order(view, active, self._fallback.service_offset)
        previous = self._fallback.last_lengths.get(service.mode) if service and view.moved else None
        move = self._fallback.choose(view)
        self.activity = self._fallback.activity
        if (service and service.stock - service.distance <= 1 and not returning
                and (previous is None or previous >= service.distance)
                and not any(_paths(view.memory, view.position).get(p, (None,))[0] == service.distance
                            for p in service.sites)):
            options = possible_directions(view, service, previous)
            if move in options:
                # При нулевом запасе не отдаём предпочтение маршруту с самым
                # сильным воображаемым сокращением относительно реального замера.
                move = max(options, key=lambda m: (options[m], m == move))
        return move
