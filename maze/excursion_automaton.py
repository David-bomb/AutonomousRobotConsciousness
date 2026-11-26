"""Разведывательные поездки из района с двумя возобновляемыми ресурсами."""
from functools import lru_cache

from .frontier_automaton import CONFIRMED_GOAL, _arrive, _paths, _service_states
from .frontier_reserve import ReserveFrontier, _lengths, _optimistic
from .grid import cell, destination, locations
from .interface import Move
from .pilot_automaton import Mode


@lru_cache(maxsize=128)
def shelters(memory, stops):
    """Две станции образуют воспроизводимый цикл, если хватает на путь туда и обратно."""
    out = {}
    for p, kind in stops:
        roads = _paths(memory, p)
        nearest = min((roads[q][0] for q, other in stops if other != kind and q in roads),
                      default=101)
        if 2*nearest <= 100:
            out[p] = (kind, nearest)
    return out


@lru_cache(maxsize=128)
def lookouts(memory, goal):
    optimistic = _optimistic(memory)
    to_goal = _lengths(optimistic, goal)
    out = {}
    for y, row in enumerate(memory):
        for x, symbol in enumerate(row):
            if symbol in "#?":
                continue
            p = (x, y)
            guesses = []
            for move in Move:
                q = destination(optimistic, p, move)
                if q in to_goal and destination(memory, p, move) is None:
                    guesses.append(1+to_goal[q])
            if guesses:
                out[p] = min(guesses)
    return out


@lru_cache(maxsize=256)
def excursion(view, goal, stops, maximize_range=False, outer_rank=None, extra_lookouts=None,
              onward_estimate=None, return_sites=shelters):
    grid = view.memory
    origin = view.position
    safe_sites = return_sites(grid, stops)
    if not any(site in _paths(grid, origin) for site in safe_sites):
        return None
    candidates = lookouts(grid, goal)
    side_views = extra_lookouts(grid, goal) if extra_lookouts else {}
    if side_views:
        candidates = {**side_views, **candidates}
    final_trips = {}
    outside = outer_rank(grid,goal,stops,min(_paths(grid,origin))) if outer_rank else None
    finish = best = None
    for traveled, f, h, first, roads in _service_states(grid, origin, view.fuel, view.health, stops):
        if goal in roads:
            length, move = roads[goal]
            if length and _arrive("B", length, f, h):
                row = (traveled+length, first or move)
                if finish is None or row[0] < finish[0]:
                    finish = row
        if finish:
            if traveled >= finish[0]:
                break
            continue
        for point, onward in candidates.items():
            if point not in roads or point == origin:
                continue
            length, move = roads[point]
            # Если точка обзора сама является сервисом, учесть пополнение
            # и при первом прибытии, и при подходе через другой сервис.
            after = (f, h) if not length else _arrive(cell(grid, *point), length, f, h)
            if after is None:
                continue
            safe = False
            for site, (kind, shuttle) in safe_sites.items():
                back = _paths(grid, site).get(point)
                if back is None:
                    continue
                upon = _arrive(kind, back[0], *after)
                if upon and upon[1 if kind == "F" else 0] >= shuttle:
                    safe = True
                    break
            if safe:
                rank = (0, point in side_views, traveled+length+onward, onward, point)
            elif point in side_views:
                continue
            elif onward_estimate:
                # Одно и то же окно можно достичь через разные станции.
                # Выбрать запас до расчёта скрытого участка, который нужен
                # лишь после исчерпания всех поездок с проверенным возвратом.
                offer = (-min(after), traveled+length)
                if point not in final_trips or offer < final_trips[point][0]:
                    final_trips[point] = offer, first or move, onward
                continue
            elif outside is not None and point in outside:
                # При окончательном выезде путь к выбранной цели должен
                # продолжаться вне уже обследованной связной области.
                margin = max(min(after[0]-f,after[1]-h,cap) for f,h,cap in outside[point])
                rank = (1, 0, -margin, onward, traveled+length, point)
            elif outside is not None:
                rank = (1, 1, onward-min(after), onward, traveled+length, point)
            elif maximize_range:
                rank = (1, onward-min(after), onward, traveled+length, point)
            else:
                # За рубежом возврата сначала выбирается точка с наименьшей
                # нехваткой запаса до финиша, затем короткий подход к ней.
                deficit = max(0, onward-min(after))
                rank = (1, deficit, traveled+length+onward, -min(after), onward, point)
            if best is None or rank < best[0]:
                best = rank, first or move, safe
    if finish:
        return CONFIRMED_GOAL, finish[1]
    if final_trips and best is None:
        lengths_after_exit = onward_estimate(grid,goal)
        def final_rank(point):
            (minus_stock,leg),_,onward = final_trips[point]
            estimate = lengths_after_exit.get(point,onward)
            return estimate+minus_stock, estimate, leg, point
        point = min(final_trips,key=final_rank)
        return "Выезд за рубеж возврата",final_trips[point][1]
    if best:
        return ("Разведка с возобновляемым возвратом" if best[2] else "Выезд за рубеж возврата"), best[1]
    return None


class Excursion(ReserveFrontier):
    maximize_range = False
    outer_rank = None
    extra_lookouts = None
    onward_estimate = None
    return_sites = staticmethod(shelters)

    def _before_base(self,view):
        return super().choose(view)

    def _reconsider_sortie(self,view,goal,stops,planned):
        return None

    def choose(self, view):
        if view.last_move is None:
            self._service_streak = 0
            self._unseen = None
            self._fallback = self._make_fallback()
        goal = view.position[0]+view.dx, view.position[1]+view.dy
        stops = tuple((p, k) for k in "FR" for p in locations(view.memory, k))
        plan = excursion(view, goal, stops, self.maximize_range, self.outer_rank,
                         self.extra_lookouts, self.onward_estimate, self.return_sites)
        if plan is None:
            return self._before_base(view)
        if plan[0] == "Выезд за рубеж возврата":
            alternative = self._reconsider_sortie(view, goal, stops, plan[1])
            if alternative is not None:
                return alternative
        self._fallback.mode = Mode.GOAL
        self._fallback.returning = False
        self._fallback.last_lengths = {Mode.FUEL: view.fuel_distance, Mode.REPAIR: view.repair_distance}
        self.activity, move = plan
        return move


class RangeExcursion(Excursion):
    """В поездке без подтверждённого возврата бережёт запас для последующего обхода."""
    maximize_range = True
