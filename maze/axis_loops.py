"""Отдельно проверять возможное число проходов вдоль и поперёк видимых улиц."""
from collections import Counter
from functools import lru_cache

import numpy as np

from .compatible_loops import DominantLoops, compatible_interval, compatible_samples, measured_constraints
from .exact_worlds import pack
from .pattern_roads import cells_from_parts, patterns, road_parts, scaffold
from .route_witness import geometry


@lru_cache(maxsize=96)
def axis_samples(memory, goal, origin, readings, prior, old):
    original = compatible_samples(memory, goal, origin, readings, prior, old)
    if len(original) >= 2:
        return original
    width, height = len(memory[0]), len(memory)
    shifts, _ = geometry(width, height)
    constraints = measured_constraints(memory, origin, readings, prior, old)
    board, models, probability = patterns(memory, goal)
    rng = np.random.default_rng(42)
    accepted = Counter()
    for trial in range(16):
        model = models[rng.choice(len(models), p=probability)]
        if model is None:
            continue
        _, u, v, doors, forced, _, _ = scaffold(memory, model)
        tree, _ = road_parts(memory, model, rng)
        base = pack(cells_from_parts(memory, model, board, tree | forced, goal[1]*width + goal[0]))
        prefixes = [[0], [0]]
        for e in rng.permutation(np.flatnonzero(~(tree | forced))):
            mask = sum(1 << int(p) for p in doors[e]) & ~base
            if mask:
                axis = int(u[e] // width == v[e] // width) ^ (trial % 2)
                prefixes[axis].append(prefixes[axis][-1] | mask)
        fixed, varied = prefixes
        # Оба порядка осей представлены поровну. Для каждого слоя первой оси
        # на второй оси отбирается весь интервал точного совпадения датчиков.
        for band in np.array_split(np.arange(len(fixed)), min(4, len(fixed))):
            first = base | fixed[int(rng.choice(band))]
            masks = [first | suffix for suffix in varied]
            lo, end = compatible_interval(masks, shifts, constraints)
            if lo < end:
                sample = masks[int(rng.integers(lo, end))]
                accepted[sample] += len(band) / len(fixed) * (end-lo) / len(masks)
    for mask, weight in original:
        accepted[mask] += weight
    return tuple(sorted(accepted.items()))


class AxisLoops(DominantLoops):
    loop_samples = staticmethod(axis_samples)
