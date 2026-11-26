"""Четыре переключаемых правила на информационном входе frontier-reserve (стенд E-0006)."""
from functools import lru_cache

from maze.frontier_automaton import CONFIRMED_GOAL, _arrive, _escape_lengths, _frontiers, _paths
from maze.frontier_reserve import ReserveFrontier, _lengths, _optimistic, _relaxed_route, _service_floors
from maze.grid import cell, destination, locations
from maze.interface import Move
from maze.pilot_automaton import Mode
from maze.resource_automaton import Service, opposite, possible_directions


@lru_cache(maxsize=128)
def stops(grid, goal):
    return tuple((p,k) for k in "FR" for p in locations(grid,k)) + ((goal,"B"),)


def gain(grid, p):
    x, y = p
    return sum(cell(grid, x+i, y+j) == "?" for i in (-1,0,1) for j in (-1,0,1))


def neighbors(v):
    return [(m,q) for m in Move if (q := destination(v.memory, v.position, m)) is not None]


@lru_cache(maxsize=256)
def distance_fields(grid, target):
    known = _paths(grid,target) if cell(grid,*target) != "?" else {}
    return known, _lengths(_optimistic(grid), target)


@lru_cache(maxsize=2048)
def route_move(v, goal, proposed, activity):
    """Сравнить финиш, один заезд и оба порядка двух заездов по памяти."""
    targets = stops(v.memory,goal)
    finish = targets[-1]
    services = targets[:-1]
    orders = [(finish,)]
    for first in services:
        orders.append((first,finish))
        orders.extend((first,second,finish) for second in services if first[1] != second[1])
    best = None
    for order in orders:
        point, stocks, score = v.position, (v.fuel,v.health), 0
        for index,(site,kind) in enumerate(order):
            known, possible = distance_fields(v.memory,site)
            confirmed = point in known
            distance = known[point][0] if confirmed else possible.get(point)
            if not distance:
                break
            if index == 0 and kind in "FR":
                reading = v.fuel_distance if kind == "F" else v.repair_distance
                if reading is not None:
                    distance = max(distance,reading)
            stocks = _arrive(kind,distance,*stocks)
            if stocks is None:
                break
            score += distance if confirmed else 2*distance
            point = site
        else:
            if best is None or score < best[0]:
                best = score, order[0]
    if best is None:
        return proposed
    site, kind = best[1]
    known, possible = distance_fields(v.memory,site)
    use_known = v.position in known
    if not use_known and ((kind == "F" and activity == Mode.FUEL.value)
                          or (kind == "R" and activity == Mode.REPAIR.value)):
        return proposed  # Сохраняем наведение по фактическим измерениям к тому же типу сервиса.
    fields = known if use_known else possible
    descending = [m for m,q in neighbors(v) if q in fields and
                  (known[q][0] < known[v.position][0] if use_known else possible[q] < possible[v.position])]
    return proposed if not descending or proposed in descending else descending[0]


@lru_cache(maxsize=2048)
def scout_move(v, goal, proposed, readings):
    previous = dict(readings)
    distances = {"F":v.fuel_distance, "R":v.repair_distance}
    if any(d is None for d in distances.values()):
        return proposed
    margins = {"F":v.fuel-distances["F"], "R":v.health-distances["R"]}
    if min(margins.values()) < 2:
        return proposed
    kind = min(margins, key=margins.get)
    if not distances[kind]:
        return proposed
    mode, stock = (Mode.FUEL,v.fuel) if kind == "F" else (Mode.REPAIR,v.health)
    sites = tuple(locations(v.memory,kind))
    roads = _paths(v.memory,v.position)
    if any(site in roads and roads[site][0] == distances[kind] for site in sites):
        return proposed
    choices = possible_directions(v, Service(mode,stock,distances[kind],sites), previous.get(mode))
    if len(choices) < 2:
        return proposed
    gains = {m:gain(v.memory,q) for m,q in neighbors(v)}
    for m in sorted(choices, key=lambda move: -gains[move]):
        if gains[m] <= gains[proposed]:
            break
        q = destination(v.memory,v.position,m)
        if q == goal:
            return m
        out = _arrive(cell(v.memory,*q),1,v.fuel,v.health)
        back = out and _arrive(cell(v.memory,*v.position),1,*out)
        if back and _relaxed_route(_optimistic(v.memory),v.position,goal,*back,stops(v.memory,goal),distances):
            return m
    return proposed


@lru_cache(maxsize=512)
def first_window(v):
    """Дойти до нового окна по известной дороге до следующего пополнения."""
    roads = v.memory
    sites = tuple((p,k) for k in "FR" for p in locations(roads,k))
    # Вход в сервис запрещён на отрезке разведки; для диагонали боковой сервис остаётся дорогой.
    paths = {v.position:(0,None)}
    queue = [v.position]
    for p in queue:
        d, first = paths[p]
        for move in Move:
            q = destination(roads,p,move)
            if q is not None and q not in paths and cell(roads,*q) not in "FR":
                paths[q] = d+1, first or move
                queue.append(q)
    escape = _escape_lengths(roads,sites)
    best = None
    for q in _frontiers(roads):
        if q == v.position or q not in paths:
            continue
        d, first = paths[q]
        after = _arrive(".",d,v.fuel,v.health)
        if after and any(_arrive(kind,back,*after) for _,kind,back in escape.get(q,())):
            rank = (d,-gain(roads,q),q)
            if best is None or rank < best[0]:
                best = rank, first
    return best[1] if best else None


@lru_cache(maxsize=2048)
def guarded(v, goal, readings, proposed):
    previous = dict(readings)
    grid = _optimistic(v.memory)
    options = sorted(neighbors(v), key=lambda item:item[0] != proposed)
    for move,q in options:
        stocks = _arrive("B" if q == goal else cell(v.memory,*q),1,v.fuel,v.health)
        if stocks and _relaxed_route(grid,q,goal,*stocks,stops(v.memory,goal),_service_floors(v,previous,q)):
            return move
    return proposed


class ExperimentalReserve(ReserveFrontier):
    def __init__(self, rules):
        super().__init__()
        self.rules = rules
        self.return_probe = False

    def choose(self, v):
        previous = tuple(self._fallback.last_lengths.items()) if v.moved else ()
        pending = self.return_probe and v.moved
        self.return_probe = False
        base = super().choose(v)
        if self.activity == CONFIRMED_GOAL:
            return base
        goal = (v.position[0]+v.dx, v.position[1]+v.dy)
        move = base
        returning = proposed_probe = False
        if pending:
            move = opposite(v.last_move)
            returning = True
        elif "T" in self.rules and self._service_streak >= 2 and (target := first_window(v)) is not None:
            move = target
        elif self.activity in (Mode.FUEL.value, Mode.REPAIR.value, Mode.GOAL.value):
            if "P" in self.rules:
                move = route_move(v,goal,move,self.activity)
            if "S" in self.rules:
                scout = scout_move(v,goal,move,previous)
                proposed_probe = scout != move
                move = scout
        final = guarded(v,goal,previous,move) if "G" in self.rules else move
        self.return_probe = proposed_probe and final == move
        if final != base or returning:
            self._fallback.returning = False
        return final
