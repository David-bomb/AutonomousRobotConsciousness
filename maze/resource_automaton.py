"""Выбор потребности и проверка пути к сервису по фактическому показанию датчика."""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

from .grid import destination, locations, wave
from .interface import Grid, Move, Observation, Point
from .pilot_automaton import Mode, PilotAutomaton, separation


@dataclass(frozen=True)
class Service:
    mode: Mode
    stock: int
    distance: int
    sites: tuple[Point, ...]


def service_order(view, active, service_offset: int = 0):
    scout_budget = min(50, 2 * max(abs(view.dx), abs(view.dy))) + service_offset
    demands = []
    for mode, stock, other, distance, symbol in (
        (Mode.FUEL, view.fuel, view.health, view.fuel_distance, "F"),
        (Mode.REPAIR, view.health, view.fuel, view.repair_distance, "R"),
    ):
        if distance is None or not 0 < distance <= stock or distance >= other:
            continue
        enough_to_explore = stock-distance > scout_budget
        small_gain = 100-stock <= distance
        if mode != active and (enough_to_explore or small_gain):
            continue
        # Любая ближайшая станция лежит в пределах измеренной длины даже по прямой сетке.
        sites = tuple(s for s in locations(view.memory, symbol) if separation(view.position, s) <= distance)
        if sites:
            demands.append(Service(mode, stock, distance, sites))
    if len(demands) < 2:
        return next(iter(demands), None)

    def priority(pair):
        first, second = pair
        other_sites = locations(view.memory, "R" if second.mode == Mode.REPAIR else "F")
        between = min(separation(a, b) for a in first.sites for b in other_sites)
        too_late = max(second.distance, first.distance + between) > second.stock
        return too_late, first.mode != active, first.stock + first.distance, first.distance

    a, b = demands
    return min(((a, b), (b, a)), key=priority)[0]


@lru_cache(maxsize=256)
def possible_directions(view, service, previous):
    """Нижняя оценка через каждый первый ход. Поиск идёт лишь на карте памяти."""
    optimistic = tuple(row.replace("?", ".") for row in view.memory)
    origin = view.position
    legal = {p: m for m in Move if (p := destination(view.memory, origin, m)) is not None}
    if previous is not None and view.moved:
        dx, dy = view.last_move.value
        last = (origin[0]-dx, origin[1]-dy)
        # Предыдущий реальный замер даёт нижнюю границу в соседних клетках.
        # Для шага по кратчайшей дороге новое показание должно быть ровно D-1.
        if previous >= service.distance:
            legal.pop(last, None)
        if previous > service.distance:
            for move in Move:
                legal.pop(destination(view.memory, last, move), None)
    queue = deque((p, 0) for p in service.sites)
    seen = set(service.sites) | {origin}
    options = {}
    while queue:
        p, length = queue.popleft()
        if p in legal:
            options[legal[p]] = length + 1
            if len(options) == len(legal):
                break
        if length + 1 >= service.distance:
            continue
        for m in Move:
            q = destination(optimistic, p, m)
            if q is not None and q not in seen:
                seen.add(q)
                queue.append((q, length+1))
    return options


@lru_cache(maxsize=256)
def known_paths(memory: Grid, position: Point):
    return {p: (d, first) for p, d, first in wave(memory, [position])}


def confirmed_route(view, goal, paths):
    """Кратчайшее посильное продолжение по открытым дорогам через известные сервисы."""
    if goal not in paths:
        return None
    if paths[goal][0] <= min(view.fuel, view.health):
        return Mode.GOAL, paths[goal][1]
    stops = [(p, mode) for symbol, mode in (("F", Mode.FUEL), ("R", Mode.REPAIR))
             for p in locations(view.memory, symbol) if p in paths]
    stops.append((goal, Mode.GOAL))
    queue = [(0, 0, view.position, view.fuel, view.health, None)]
    serial = 0
    seen = {}
    while queue:
        length, _, point, fuel, health, first = heappop(queue)
        if point == goal:
            return first
        stocks = seen.setdefault(point, [])
        if any(f >= fuel and h >= health for f, h in stocks):
            continue
        stocks.append((fuel, health))
        roads = known_paths(view.memory, point)
        for site, mode in stops:
            distance, move = roads[site]
            if distance == 0 or distance > min(fuel, health):
                continue
            f = 100 if mode == Mode.FUEL else fuel-distance
            h = 100 if mode == Mode.REPAIR else health-distance
            if mode != Mode.GOAL and min(f, h) == 0:
                continue
            serial += 1
            heappush(queue, (length+distance, serial, site, f, h, first or (mode, move)))
    return None


class Automaton(PilotAutomaton):
    def __init__(self, service_offset: int = 0):
        super().__init__(service_offset)
        self.last_lengths = {}
        self.returning = False

    def choose(self, view: Observation) -> Move:
        previous_lengths = self.last_lengths if view.moved else {}
        self.last_lengths = {Mode.FUEL: view.fuel_distance, Mode.REPAIR: view.repair_distance}
        x, y = view.position
        goal = (x+view.dx, y+view.dy)
        paths = known_paths(view.memory, view.position)
        route = confirmed_route(view, goal, paths)
        if route is not None:
            self.mode, move = route
            self.returning = False
            return move
        active = self.mode if view.last_move is not None else Mode.GOAL
        service = service_order(view, active, self.service_offset)
        self.mode = service.mode if service else Mode.GOAL
        if view.last_move is None or self.mode != active:
            self.returning = False
        if service is None:
            return self._toward(view, paths, [goal])

        # Равенство с датчиком удостоверяет кратчайший уже открытый маршрут.
        for site in service.sites:
            if site in paths and paths[site][0] == service.distance:
                self.returning = False
                return paths[site][1]

        previous = previous_lengths.get(service.mode)
        if previous is not None and service.distance > previous:
            # Обратный шаг гарантированно восстанавливает меньшее показание.
            self.returning = self.mode == active
            return opposite(view.last_move)
        options = possible_directions(view, service, previous)
        if not options:
            return self._toward(view, paths, service.sites)
        if self.returning:
            self.returning = False
            # Последняя физическая команда известна из наблюдения. После возврата
            # проверяем следующий допустимый выход в общем порядке восьми направлений.
            moves = list(Move)
            index = moves.index(opposite(view.last_move))
            return next(m for m in moves[index+1:] + moves[:index+1] if m in options)
        return min(Move, key=lambda m: options.get(m, float("inf")))


def opposite(move):
    dx, dy = move.value
    return Move((-dx, -dy))
