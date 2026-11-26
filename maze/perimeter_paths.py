"""Сравнивать две гипотезы о внешней стене по увиденным клеткам границы."""
from functools import lru_cache

import numpy as np

from .grid import cell
from .passage_flow import passage_graph
from .simple_exit_paths import simple_exit_probabilities
from .specific_paths import SpecificPaths
from .street_trunk import alternatives


@lru_cache(maxsize=64)
def boundary(width, height):
    return tuple((x, y) for y in range(height) for x in range(width)
                 if x == 0 or y == 0 or x == width - 1 or y == height - 1)


@lru_cache(maxsize=256)
def wall_probability(memory, goal):
    edge = boundary(len(memory[0]), len(memory))
    if goal in edge or any(cell(memory, *p) not in "#?" for p in edge):
        return 0.
    seen = sum(cell(memory, *p) == "#" for p in edge)
    # У сплошной стены вероятность n увиденных стен равна 1. При неизвестной
    # доле стен с равномерным априорным распределением она равна 1/(n+1).
    return (seen + 1) / (seen + 2)


@lru_cache(maxsize=64)
def interior_graph(memory, goal):
    graph = passage_graph(memory, goal).tocoo()
    width = len(memory[0])
    interior = np.ones(graph.shape[0], dtype=bool)
    for x, y in boundary(width, len(memory)):
        interior[y*width + x] = False
    restricted = graph.multiply(interior[:, None]).multiply(interior).tocsr()
    restricted.eliminate_zeros()
    return restricted


@lru_cache(maxsize=32)
def perimeter_probabilities(memory, goal, windows):
    ordinary = simple_exit_probabilities(memory, goal, windows)
    # Для маршрута к сервису сохраняем родительскую оценку. Здесь цель
    # задаётся координатами B, даже когда её клетка ещё не была в окне.
    if cell(memory, *goal) in "FR":
        return ordinary
    weight = wall_probability(memory, goal)
    if not weight:
        return ordinary
    enclosed = simple_exit_probabilities(memory, goal, windows, interior_graph)
    if not any(curve[-1] > 0 for curve in enclosed.values()):
        return ordinary
    return {p: (1-weight)*curve + weight*enclosed[p] for p, curve in ordinary.items()}


class PerimeterExits(SpecificPaths):
    probabilities = staticmethod(perimeter_probabilities)


@lru_cache(maxsize=96)
def perimeter_patterns(memory, goal):
    weight = wall_probability(memory, goal)
    patterns = alternatives(memory, goal)
    if not weight:
        return patterns
    edge = boundary(len(memory[0]), len(memory))
    result = []
    for prior, grid in patterns:
        assumed = [list(row) for row in grid]
        for x, y in edge:
            assumed[y][x] = "#"
        result.extend(((prior*(1-weight), grid), (prior*weight, tuple("".join(row) for row in assumed))))
    return result


class PerimeterApproach(PerimeterExits):
    road_patterns = staticmethod(perimeter_patterns)
