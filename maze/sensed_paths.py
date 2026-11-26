"""Проверять предполагаемую дорогу к финишу уже полученными показаниями до станций."""
from functools import lru_cache

import numpy as np

from .compatible_loops import measured_constraints
from .exact_worlds import layers
from .exit_flow import eventual_exits
from .frontier_reserve import _lengths, _optimistic
from .grid import locations
from .passage_flow import passage_graph
from .pilot_automaton import Mode
from .route_witness import geometry
from .second_window import ReturnWindows
from .simple_exit_paths import sampled_walks, simple_exit_probabilities


@lru_cache(maxsize=32)
def road_samples(memory, goal, points):
    w = len(memory[0])
    known = sum(1 << i for i,c in enumerate("".join(memory)) if c not in "#?")
    result = []
    for point, path in sampled_walks(memory, goal, points, passage_graph, trace=True):
        opened = known | (1 << path[0])
        stops = []
        for a,b in zip(path,path[1:]):
            opened |= 1 << b
            if a//w != b//w and a%w != b%w:
                opened |= (1 << (a//w*w+b%w)) | (1 << (b//w*w+a%w))
        for t, i in enumerate(reversed(path)):
            c = memory[i//w][i%w]
            if t and c in "FR":
                stops.append((t,c))
        result.append((point, len(path)-1, opened, tuple(stops)))
    return tuple(result)


@lru_cache(maxsize=96)
def sensed_curves(memory, goal, points, pos, readings, prior, old):
    initial = simple_exit_probabilities(memory, goal, points)
    if goal in points or not any(c[-1] for c in initial.values()):
        return initial
    widths = len(memory[0]), len(memory)
    shifts,_ = geometry(*widths)
    constraints = []
    for (source, bounds) in measured_constraints(memory,pos,readings,prior,old):
        relevant = []
        # Если даже полностью открытая область не даёт короткого пути,
        # добавление проходимых клеток этого наблюдения не нарушит.
        for p,d in bounds:
            i = p.bit_length()-1
            lower = _lengths(_optimistic(memory),(i%widths[0],i//widths[0]))
            if any(lower.get(q,101) < d for k in "FR" for q in locations(memory,k)
                   if source & (1 << (q[1]*widths[0]+q[0]))):
                relevant.append((p,d))
        if relevant:
            constraints.append((source,relevant))
    if not constraints:
        return initial
    shares = eventual_exits(memory,goal,points)
    n = dict.fromkeys(points,0)
    counts = {p:np.zeros(101) for p in points}
    discounted = set()
    for p, length, opened, _ in road_samples(memory,goal,points):
        n[p] += 1
        if length > 100:
            continue
        for source,bounds in constraints:
            reach = layers(opened,shifts,source,max(d-1 for _,d in bounds))
            if any(reach[d-1] & b for b,d in bounds):
                discounted.add(p)
                break
        else:
            counts[p][length] += 1
    if not discounted:
        return initial
    return {p:shares[p]*counts[p].cumsum()/n[p] if p in discounted else curve
            for p,curve in initial.items()}


class SensedPaths(ReturnWindows):
    def sortie_curves(self, view, goal, points):
        prior = None
        if view.moved:
            dx,dy = view.last_move.value
            prior = view.position[0]-dx, view.position[1]-dy
        old = tuple(self._fallback.last_lengths.get(m) for m in (Mode.FUEL,Mode.REPAIR))
        return sensed_curves(view.memory,goal,points,view.position,
                             (view.fuel_distance,view.repair_distance),prior,old)
