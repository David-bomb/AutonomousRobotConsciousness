"""Доработка frontier: резерв до развилки, проверка двух запасов и выход из застоя."""
from functools import lru_cache
from heapq import heappop, heappush

from .frontier_automaton import FrontierAutomaton, _arrive, _frontiers, _paths, _service_states
from .grid import cell, destination, wave
from .interface import Move
from .pilot_automaton import Mode, separation
from .resource_automaton import Automaton as ResourceAutomaton
from .resource_automaton import possible_directions, service_order


@lru_cache(maxsize=128)
def _optimistic(grid):
    return tuple(row.replace("?", ".") for row in grid)


@lru_cache(maxsize=128)
def _lengths(grid, origin):
    return {p: d for p, d, _ in wave(grid, [origin])}


def _service_floors(view, previous, point):
    """Не запрашивает датчик для point: использует неравенство треугольника."""
    out = {}
    for kind, mode, current in (("F", Mode.FUEL, view.fuel_distance),
                                ("R", Mode.REPAIR, view.repair_distance)):
        if current is None:
            continue
        floor = max(0, current - 1)
        old = previous.get(mode)
        if old is not None and view.moved:
            dx, dy = view.last_move.value
            prior = view.position[0] - dx, view.position[1] - dy
            px, py = point[0] - prior[0], point[1] - prior[1]
            move = next((m for m in Move if m.value == (px, py)), None)
            distance = 0 if prior == point else (
                1 if move is not None and destination(view.memory, prior, move) == point else 2)
            floor = max(floor, old - distance)
        out[kind] = floor
    return out


def _relaxed_route(grid, point, goal, fuel, health, stops, floors):
    """Нижняя оценка успешного маршрута; None исключает ход, не подтверждает другие."""
    queue = [(point, fuel, health, True)]
    seen = {}
    while queue:
        p, f, h, first = queue.pop()
        if p == goal:
            return True
        previous = seen.setdefault((p, first), [])
        if any(of >= f and oh >= h for of, oh in previous):
            continue
        previous[:] = [(of, oh) for of, oh in previous if not (f >= of and h >= oh)]
        previous.append((f, h))
        for site, kind in stops:
            distance = _lengths(grid, site).get(p)
            if distance is None or not distance:
                continue
            if first:
                distance = max(distance, floors.get(kind, 0))
            after = _arrive(kind, distance, f, h)
            if after is not None:
                queue.append((site, *after, False))
    return False


@lru_cache(maxsize=96)
def _estimated_lengths(memory, sites):
    """Открытые рёбра предпочтительнее неизвестных; это не показание датчика."""
    optimistic = _optimistic(memory)
    queue = [(0, p) for p in sites]
    seen = {}
    while queue:
        cost, point = heappop(queue)
        if point in seen:
            continue
        seen[point] = cost
        for m in Move:
            q = destination(optimistic, point, m)
            if q is None or q in seen:
                continue
            known = cell(memory, *point) not in "#?" and destination(memory, point, m) == q
            heappush(queue, (cost + (1 if known else 2), q))
    return seen


def _prioritize_direction(view, service, previous, default):
    options = possible_directions(view, service, previous)
    if default not in options or len(options) < 2:
        return default
    roads = _paths(view.memory, view.position)
    if any(roads.get(site, (float("inf"),))[0] <= service.stock
           for site in service.sites):
        return default
    lengths = _estimated_lengths(view.memory, tuple(sorted(service.sites)))
    def rank(move):
        q = destination(view.memory, view.position, move)
        x, y = q
        discovered = sum(cell(view.memory, x + i, y + j) == "?"
                         for i in (-1, 0, 1) for j in (-1, 0, 1))
        estimate = lengths.get(q, float("inf"))
        return estimate, -discovered, move != default
    return min(options, key=rank)


class ReserveFrontier(FrontierAutomaton):
    """Законченные гипотезы сравниваются отдельно; v1 не подменяется."""
    STREAK_CAP = 4
    RESERVE = 2
    # Три пары «проба, возврат»: в этом диапазоне добавляем проверку
    # совместной достижимости. За его пределами оставляем базовый поиск.
    CHECK_MARGIN = 3 * RESERVE

    def _make_fallback(self):
        # Одна обзорная проба и обратный ход расходуют две единицы.
        return ResourceAutomaton(self.RESERVE)

    def _accept_escape(self, view, goal, stops, station, total, stocks):
        return True

    def _extra_plan(self, view, goal, stops):
        if self._service_streak < self.STREAK_CAP:
            return None
        candidates = _frontiers(view.memory)
        if not candidates:
            return None
        # После повторных обслуживаний выбираем достижимое новое наблюдение,
        # а не ещё один заезд сам по себе. Проверяем возможность хотя бы
        # вернуться к известному сервису по уже открытой дороге.
        best = None
        for length, f, h, first, paths in _service_states(
                view.memory, view.position, view.fuel, view.health, stops):
            for point in candidates:
                if point not in paths or point == view.position:
                    continue
                d, move = paths[point]
                after = _arrive(".", d, f, h)
                if after is None:
                    continue
                for station, kind in stops:
                    ret = _paths(view.memory, station).get(point)
                    if ret is None or not ret[0]:
                        continue
                    stocks = _arrive(kind, ret[0], *after)
                    if stocks is None or not self._accept_escape(
                            view, goal, stops, station, length + d + ret[0], stocks):
                        continue
                    # Близкое новое окно важнее близости к финишу при застое.
                    score = (length + d, -min(stocks), separation(point, goal), point)
                    if best is None or score < best[0]:
                        best = score, first or move
        return ("Выход из застоя с возвратом к сервису", best[1]) if best else None

    def _may_rank(self, view, service):
        return True

    def _fallback_move(self, view, goal, stops):
        previous = self._fallback.last_lengths.copy()
        was_returning = self._fallback.returning
        service = service_order(view, self._fallback.mode, self.RESERVE)
        move = super()._fallback_move(view, goal, stops)
        if service is not None and not was_returning:
            old = previous.get(service.mode) if view.moved else None
            if (service.stock - service.distance <= self.CHECK_MARGIN
                    and (old is None or old >= service.distance) and self._may_rank(view, service)):
                move = _prioritize_direction(view, service, old, move)
        margins = (stock - distance for stock, distance in
                   ((view.fuel, view.fuel_distance), (view.health, view.repair_distance))
                   if distance is not None)
        if min(margins, default=100) > self.CHECK_MARGIN:
            return move
        grid = _optimistic(view.memory)
        targets = (*stops, (goal, "B"))
        options = []
        for choice in Move:
            q = destination(view.memory, view.position, choice)
            if q is None:
                continue
            kind = "B" if q == goal else cell(view.memory, *q)
            after = _arrive(kind, 1, view.fuel, view.health)
            if after is None:
                continue
            floors = _service_floors(view, previous, q)
            if not _relaxed_route(grid, q, goal, *after, targets, floors):
                continue
            if choice == move:
                return move
            # Из оставшихся ходов предпочтём подтверждённое пополнение;
            # иначе — наибольший остаток у наиболее стеснённого ресурса.
            slack = min(stock - floors.get(kind, 0)
                        for kind, stock in zip("FR", after))
            options.append(((q == goal, kind in "FR", slack), choice))
        if not options:
            return move
        self._fallback.returning = False
        self.activity = "Обход несовместимого порядка обслуживания"
        return max(options, key=lambda row: row[0])[1]
