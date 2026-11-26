"""Проверить, возможен ли финиш после хода хоть при одной допустимой достройке дорог."""
from functools import lru_cache

from .compatible_loops import layout, measured_constraints
from .exact_worlds import edges, layers
from .frontier_automaton import _arrive
from .frontier_reserve import _service_floors
from .grid import cell, destination
from .pilot_automaton import Mode
from .route_witness import geometry
from .second_window import ReturnWindows


@lru_cache(maxsize=256)
def shortcut(memory, goal, position, readings, prior, old, opened):
    """Вернуть клетки короткого пути, хотя бы одна из которых должна быть стеной."""
    w, h = len(memory[0]), len(memory)
    shifts, _ = geometry(w, h)
    gates = edges(w, h)
    mutable = sum(1 << i for i, c in enumerate("".join(memory)) if c == "?")
    mutable &= ~(1 << (goal[1]*w + goal[0]))
    best = None
    for markers, bounds in measured_constraints(memory, position, readings, prior, old):
        reach = layers(opened, shifts, markers, max(d for _,d in bounds))
        for start, d in bounds:
            if not reach[d] & start:
                return ()
            if not d or not reach[d-1] & start:
                continue
            length = next(t for t, seen in enumerate(reach) if start & seen)
            needed = 0
            p = start.bit_length()-1
            for t in range(length, 0, -1):
                choices = ((q, gate) for q,gate in gates[p] if gate & opened == gate and reach[t-1] & (1 << q))
                p, gate = min(choices, key=lambda edge: (edge[1] & mutable).bit_count())
                needed |= gate & mutable
            # Начало измерения всегда лежит в уже известном окне.
            if not needed:
                return ()
            if best is None or needed.bit_count() < best.bit_count():
                best = needed
    if best is None:
        return None
    bits = []
    while best:
        bit = best & -best
        bits.append(bit)
        best ^= bit
    return tuple(bits)


def viable_in_layout(view, goal, opened, move, old):
    q = destination(view.memory, view.position, move)
    if q is None:
        return False
    if q == goal:
        return True
    after = _arrive(cell(view.memory, *q), 1, view.fuel, view.health)
    if not after:
        return False
    model = layout(view.memory, goal, opened)
    start = q[1]*len(view.memory[0]) + q[0]
    if model.fields[-1,start] <= min(after):
        return True
    previous = dict(zip((Mode.FUEL, Mode.REPAIR),old))
    floors = _service_floors(view, previous, q)
    for j,(kind, _) in enumerate(model.sites[:-1]):
        d = max(floors.get(kind, 0), model.fields[j,start])
        if d <= min(after):
            stocks = _arrive(kind, int(d), *after)
            if stocks and stocks[int(kind == "F")] >= model.needed[j]:
                return True
    return False


def completion_status(view, goal, readings, prior, old, move, limit=32):
    full = sum(1 << i for i,c in enumerate("".join(view.memory)) if c != "#")
    pending = [full]
    seen = set()
    while pending:
        mask = pending.pop()
        if mask in seen:
            continue
        if len(seen) >= limit:
            return "budget", None, len(seen)
        seen.add(mask)
        if not viable_in_layout(view, goal, mask, move, old):
            continue
        block = shortcut(view.memory, goal, view.position, readings, prior, old, mask)
        if block is None:
            # Совместимая с датчиками геометрия допускает продолжение.
            return "witness", mask, len(seen)
        pending.extend(mask ^ bit for bit in block)
    return "impossible", None, len(seen)


def still_possible(view, goal, readings, prior, old, move, limit=32):
    status, _, _ = completion_status(view, goal, readings, prior, old, move, limit)
    return status != "impossible"


class RouteShield(ReturnWindows):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        own, other = (view.fuel, view.health) if kind == 0 else (view.health, view.fuel)
        if readings[kind] is None or min(own,other-1) - readings[kind] >= 2:
            return parent
        goal = view.position[0] + view.dx, view.position[1] + view.dy
        for move in dict.fromkeys((parent, default, *options)):
            if still_possible(view, goal, readings, prior, old, move):
                return move
        return parent
