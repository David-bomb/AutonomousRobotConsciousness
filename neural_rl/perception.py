"""Кодирование общедоступного наблюдения и исполнение промежуточной цели."""
from functools import lru_cache
from heapq import heappop, heappush
from typing import NamedTuple

import numpy as np

from maze.grid import cell, destination, locations, wave
from maze.interface import Move, Observation

MOVES = tuple(Move)
SIDE = 32
BITS = bytes.maketrans(b"?#.BFR", bytes((1, 2, 4, 4 | 32, 4 | 8, 4 | 16)))


class State(NamedTuple):
    image: np.ndarray
    numbers: np.ndarray
    legal: np.ndarray


def distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@lru_cache(maxsize=8192)
def spatial(memory, position, goal):
    height, width = len(memory), len(memory[0])
    if max(height, width) > SIDE:
        raise ValueError(f"Карта {width}×{height} выходит за поле кодирования {SIDE}×{SIDE}")
    packed = "".join(memory).encode("ascii").translate(BITS)
    image = np.zeros((SIDE, SIDE), dtype=np.uint8)
    image[:height, :width] = np.frombuffer(packed, np.uint8).reshape(height, width)
    image[position[1], position[0]] |= 64
    image[goal[1], goal[0]] |= 32
    return image


@lru_cache(maxsize=8192)
def immediate(window):
    mask = np.array([destination(window, (1, 1), move) is not None for move in MOVES])
    if not mask.any():
        mask[:] = True
    return mask


def encode(view: Observation, hierarchical: bool):
    x, y = view.position
    image = spatial(view.memory, (x, y), (x + view.dx, y + view.dy))
    numbers = [x / SIDE, y / SIDE, view.dx / SIDE, view.dy / SIDE,
               view.fuel / 100, view.health / 100]
    for symbol, reading in (("F", view.fuel_distance), ("R", view.repair_distance)):
        sites = locations(view.memory, symbol)
        point = min(sites, key=lambda p: distance(p, (x, y))) if sites else (x, y)
        numbers.extend(((point[0]-x)/SIDE, (point[1]-y)/SIDE,
                        -1 if reading is None else reading/100))
    numbers.extend(float(view.last_move == m) for m in MOVES)
    numbers.append(-1 if view.moved is None else int(view.moved))
    moves = immediate(view.window)
    numbers.extend(moves)
    mask = (np.array([True, view.fuel_distance is not None,
                      view.repair_distance is not None, True]) if hierarchical else moves)
    return State(image, np.asarray(numbers, np.float32), mask)


def astar(memory, start, target):
    """Кратчайший путь по известным клеткам с оценкой Чебышёва для восьми ходов."""
    queue = [(distance(start, target), 0, start)]
    parent, lengths = {}, {start: 0}
    while queue:
        _, length, point = heappop(queue)
        if length != lengths[point]:
            continue
        if point == target:
            path = []
            while point != start:
                point, move = parent[point]
                path.append(move)
            return tuple(reversed(path))
        for move in MOVES:
            nxt = destination(memory, point, move)
            if nxt is not None and length + 1 < lengths.get(nxt, float("inf")):
                parent[nxt] = point, move
                lengths[nxt] = length + 1
                heappush(queue, (length + 1 + distance(nxt, target), length + 1, nxt))
    return ()


@lru_cache(maxsize=8192)
def route(memory, start, goal, option):
    targets = ((goal,) if option == 0 else locations(memory, "FR"[option-1])
               if option in (1, 2) else ())
    reachable = [(p, length) for p, length, _ in wave(memory, [start]) if length]
    exact = [(length, p) for p, length in reachable if p in targets]
    if exact:
        target = min(exact)[1]
    else:
        frontier = [(p, length) for p, length in reachable
                    if any(cell(memory, p[0]+dx, p[1]+dy) == "?"
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1))]
        candidates = frontier or reachable
        if not candidates:
            return (MOVES[0],)
        def rank(candidate):
            p, length = candidate
            remaining = min((distance(p, t) for t in targets), default=0)
            return remaining, length, p
        target = min(candidates, key=rank)[0]
    return astar(memory, start, target)


def command(view, action, hierarchical, horizon):
    if not hierarchical:
        return (MOVES[action],)
    x, y = view.position
    return route(view.memory, (x, y), (x+view.dx, y+view.dy), action)[:horizon]
