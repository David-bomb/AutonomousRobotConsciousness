"""Оставить планы помещений с проходами, воспроизводящими фактические длины путей."""
from collections import Counter
from functools import lru_cache

import numpy as np

from .compatible_loops import compatible_interval, layout, measured_constraints
from .exact_worlds import pack
from .frontier_automaton import _arrive
from .grid import destination
from .room_roads import room_connections, room_partition, separator_prior
from .route_witness import geometry
from .second_window import ReturnWindows


@lru_cache(maxsize=64)
def room_samples(memory, goal, origin, readings, prior, old):
    h, w = len(memory), len(memory[0])
    board = np.frombuffer("".join(memory).encode(), dtype="S1").reshape(h,w)
    seen = (board != b"?") & (board != b"#")
    if not (seen[1:,1:] & seen[:-1,1:] & seen[1:,:-1] & seen[:-1,:-1]).any():
        return ()
    priors = separator_prior(board), separator_prior(board.T)
    constraints = measured_constraints(memory, origin, readings, prior, old)
    shifts, _ = geometry(w,h)
    rng = np.random.default_rng(42)
    accepted = Counter()
    for _ in range(48):
        opened, known, wall, links, count = room_partition(board, goal, priors, rng)
        extra = []
        for needed, optional, choices in room_connections(links, count):
            if not optional:
                continue
            y,x = choices[int(rng.integers(len(choices)))]
            if needed:
                opened[y,x] = True
            else:
                extra.append(y*w + x)
        masks = [pack((opened | known) & ~wall)]
        for i in rng.permutation(extra):
            masks.append(masks[-1] | (1 << int(i)))
        lo, end = compatible_interval(masks, shifts, constraints)
        if lo < end:
            accepted[masks[int(rng.integers(lo, end))]] += (end-lo)/len(masks)
    return tuple(sorted(accepted.items()))


class RoomLengths(ReturnWindows):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        measured = readings[kind]
        if measured is None or not 0 <= (view.fuel,view.health)[kind] - measured < 2:
            return parent
        goal = view.position[0] + view.dx, view.position[1] + view.dy
        width = len(view.memory[0])
        def index(p):
            return p[1]*width + p[0]
        stock = _arrive("FR"[kind], measured, view.fuel, view.health)
        if stock is None:
            return parent
        origin = index(view.position)
        positions = {m:index(destination(view.memory, view.position, m)) for m in options}
        votes = Counter()
        usable = 0
        for mask, weight in room_samples(view.memory, goal, view.position, readings, prior, old):
            model = layout(view.memory, goal, mask)
            suitable = [j for j,(k,_) in enumerate(model.sites[:-1]) if k == "FR"[kind]
                        and model.fields[j,origin] == measured and stock[1-kind] >= model.needed[j]]
            good = [m for m,q in positions.items() if any(model.fields[j,q] == measured-1 for j in suitable)]
            if good:
                usable += 1
                for m in good:
                    votes[m] += weight / len(good)
        return max(options, key=lambda m:(votes[m], m == parent)) if usable >= 2 else parent
