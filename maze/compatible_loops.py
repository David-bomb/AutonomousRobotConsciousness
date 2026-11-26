"""Оставить варианты дорог, у которых длины до обеих станций совпадают с измерениями."""
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .conditional_mission import required_after_stop
from .exact_worlds import layers, pack
from .frontier_automaton import _arrive
from .grid import cell, destination, locations
from .interface import Move
from .pattern_roads import cells_from_parts, patterns, road_parts, scaffold
from .road_hypotheses import lattice
from .route_witness import geometry
from .simple_exit_paths import MissionSimplePaths


def compatible_interval(masks, shifts, constraints):
    """При последовательном открытии проходов все измеренные длины монотонно убывают."""
    @lru_cache(None)
    def compare(i):
        too_long = too_short = False
        for source, readings in constraints:
            reach = layers(masks[i], shifts, source, max(d for _, d in readings))
            for point, d in readings:
                too_long |= not bool(reach[d] & point)
                too_short |= d > 0 and bool(reach[d - 1] & point)
        return too_long, too_short

    indices = range(len(masks))
    first = bisect_left(indices, True, key=lambda i: not compare(i)[0])
    end = bisect_left(indices, True, lo=first, key=lambda i: compare(i)[1])
    return first, end


def measured_constraints(memory, origin, readings, prior, previous):
    width = len(memory[0])
    index = lambda p: p[1] * width + p[0]
    constraints = []
    for k, kind in enumerate("FR"):
        if readings[k] is None:
            continue
        bounds = [(1 << index(origin), readings[k])]
        if prior is not None and previous[k] is not None:
            bounds.append((1 << index(prior), previous[k]))
        constraints.append((sum(1 << index(p) for p in locations(memory, kind)), bounds))
    return constraints


@lru_cache(maxsize=96)
def compatible_samples(memory, goal, origin, readings, prior, previous):
    width, height = len(memory[0]), len(memory)
    index = lambda p: p[1] * width + p[0]
    shifts, _ = geometry(width, height)
    constraints = measured_constraints(memory, origin, readings, prior, previous)
    board, models, probability = patterns(memory, goal)
    rng = np.random.default_rng(42)
    accepted = Counter()
    for _ in range(32):
        model = models[rng.choice(len(models), p=probability)]
        if model is None:
            continue
        _, _, _, doors, forced, _, _ = scaffold(memory, model)
        tree, _ = road_parts(memory, model, rng)
        opened = cells_from_parts(memory, model, board, tree | forced, index(goal))
        masks = [pack(opened)]
        for e in rng.permutation(np.flatnonzero(~(tree | forced))):
            mask = masks[-1] | sum(1 << int(p) for p in doors[e])
            if mask != masks[-1]:
                masks.append(mask)
        lo, end = compatible_interval(masks, shifts, constraints)
        if lo < end:
            sample = masks[int(rng.integers(lo, end))]
            accepted[sample] += (end - lo) / len(masks)
    return tuple(sorted(accepted.items()))


def compatible_masks(memory, goal, origin, readings, prior, previous):
    return tuple(mask for mask, _ in compatible_samples(memory, goal, origin, readings, prior, previous))


@dataclass
class Layout:
    grid: tuple
    sites: tuple
    fields: np.ndarray
    needed: list

    def can_finish(self, q, stock):
        i = q[1] * len(self.grid[0]) + q[0]
        if self.fields[-1, i] <= min(stock):
            return True
        for j, (kind, _) in enumerate(self.sites[:-1]):
            d = self.fields[j, i]
            if d <= min(stock):
                after = _arrive(kind, int(d), *stock)
                if after and after[1 if kind == "F" else 0] >= self.needed[j]:
                    return True
        return False

    def signal(self, q):
        x, y = q
        i = y * len(self.grid[0]) + x
        window = tuple(cell(self.grid, x+dx, y+dy) for dy in (-1,0,1) for dx in (-1,0,1))
        return window, tuple(min((self.fields[j,i] for j,(s,_) in enumerate(self.sites) if k == s),
                                 default=None) for k in "FR")


@lru_cache(maxsize=64)
def layout(memory, goal, mask):
    width, height = len(memory[0]), len(memory)
    index = lambda p: p[1] * width + p[0]
    n = width * height
    opened = np.unpackbits(np.frombuffer(mask.to_bytes((n + 7) // 8, "little"), dtype=np.uint8),
                           bitorder="little", count=n)
    edges = lattice(width, height)
    u, v = edges[:2, opened[edges].all(axis=0)]
    graph = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()
    sites = tuple((k, index(p)) for k in "FR" for p in locations(memory, k)) + (("B",index(goal)),)
    fields = dijkstra(graph, directed=False, unweighted=True, indices=[p for _, p in sites])
    roads = bytearray(np.where(opened, ord("."), ord("#")).astype(np.uint8).tobytes())
    for k, i in sites:
        roads[i] = ord(k)
    grid = tuple(roads[i:i+width].decode() for i in range(0, n, width))
    return Layout(grid, sites, fields, required_after_stop(sites, fields))


def feasible_moves(view, goal, mask, options):
    model = layout(view.memory, goal, mask)
    safe = set()
    for move in options:
        q = destination(view.memory, view.position, move)
        stock = _arrive(cell(view.memory, *q), 1, view.fuel, view.health)
        if q == goal or stock and model.can_finish(q, stock):
            safe.add(move)
    return safe


class CompatibleLoops(MissionSimplePaths):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        if readings[kind] is None or (view.fuel, view.health)[kind] - readings[kind] >= 2:
            return parent
        goal = view.position[0] + view.dx, view.position[1] + view.dy
        common = set(options)
        count = 0
        for mask in compatible_masks(view.memory, goal, view.position, readings, prior, old):
            allowed = feasible_moves(view, goal, mask, options)
            if parent in allowed:
                return parent
            if allowed:
                count += 1
                common.intersection_update(allowed)
                if not common:
                    return parent
        # Отступление от родителя требует согласия хотя бы двух различных сетей.
        return next((m for m in options if m in common), parent) if count >= 2 else parent


class WeightedLoops(MissionSimplePaths):
    preserve_possible = False
    loop_samples = staticmethod(compatible_samples)

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        if readings[kind] is None or (view.fuel, view.health)[kind] - readings[kind] >= 2:
            return parent
        goal = view.position[0] + view.dx, view.position[1] + view.dy
        scores = dict.fromkeys(options, 0.)
        preserved = set(options)
        count = 0
        for mask, weight in self.loop_samples(view.memory, goal, view.position, readings, prior, old):
            allowed = feasible_moves(view, goal, mask, options)
            if not allowed:
                continue
            count += 1
            for move in allowed:
                scores[move] += weight
            if self.preserve_possible and parent in allowed:
                preserved.intersection_update(allowed)
        if count < 2:
            return parent
        return max((m for m in options if m in preserved), key=lambda m: (scores[m], m == parent))


class DominantLoops(WeightedLoops):
    """Требует сохранности всех совместимых сетей, в которых исходный ход оставляет финиш."""
    preserve_possible = True


class ObservedLoops(MissionSimplePaths):
    """Вторую команду выбирает одинаково для сетей с одним и тем же будущим наблюдением."""
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        parent = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        if readings[kind] is None or (view.fuel, view.health)[kind] - readings[kind] >= 2:
            return parent
        goal = view.position[0] + view.dx, view.position[1] + view.dy
        branches = {m: defaultdict(Counter) for m in options}
        usable = 0
        for mask, weight in compatible_samples(view.memory, goal, view.position, readings, prior, old):
            model = layout(view.memory, goal, mask)
            allowed = feasible_moves(view, goal, mask, options)
            usable += bool(allowed)
            for m in allowed:
                q = destination(view.memory, view.position, m)
                if q == goal:
                    return m
                stock = _arrive(cell(model.grid, *q), 1, view.fuel, view.health)
                scores = branches[m][model.signal(q)]
                for second in Move:
                    p = destination(model.grid, q, second)
                    if p is None:
                        continue
                    after = _arrive(cell(model.grid, *p), 1, *stock)
                    if after and model.can_finish(p, after):
                        scores[second] += weight
        if usable < 2:
            return parent
        value = {m: sum(max(row.values(), default=0.) for row in groups.values()) for m, groups in branches.items()}
        return max(options, key=lambda m: (value[m], m == parent))
