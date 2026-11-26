"""Один закон движения для симулятора и поиска по переданной карте."""
from collections import deque
from collections.abc import Iterable, Iterator
from functools import lru_cache

from .interface import Grid, Move, Point


def cell(grid: Grid, x: int, y: int) -> str:
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        return grid[y][x]
    return "#"


def destination(grid: Grid, point: Point, move: Move) -> Point | None:
    x, y = point
    dx, dy = move.value
    if cell(grid, x + dx, y + dy) in "#?":
        return None
    if dx and dy and (cell(grid, x + dx, y) in "#?" or cell(grid, x, y + dy) in "#?"):
        return None
    return x + dx, y + dy


@lru_cache(maxsize=16)
def _adjacency(grid: Grid) -> dict[Point, tuple[tuple[Move,Point], ...]]:
    # Одна таблица для равных снимков клеток. Пополняется только в раскрытой
    # при обходе области и заново создаётся при изменении снимка.
    return {}


def wave(grid: Grid, starts: Iterable[Point], *, stop_at: frozenset[Point] = frozenset()
         ) -> Iterator[tuple[Point, int, Move | None]]:
    """Выдать клетки и первый ход. При достижении остановки не продолжать через неё."""
    seen = set(starts)
    queue = deque((p, 0, None) for p in sorted(seen))
    try:
        edges = _adjacency(grid)
    except TypeError:  # Нехешируемая карта у внешнего вызывающего кода.
        edges = {}
    while queue:
        point, length, first = queue.popleft()
        yield point, length, first
        if stop_at and length and point in stop_at:
            continue
        neighbors = edges.get(point)
        if neighbors is None:
            neighbors = tuple((move,nxt) for move in Move if (nxt := destination(grid,point,move)) is not None)
            edges[point] = neighbors
        for move, nxt in neighbors:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, length + 1, first or move))


def _locate(grid: Grid, symbol: str) -> tuple[Point, ...]:
    return tuple((x,y) for y,row in enumerate(grid) for x,char in enumerate(row) if char == symbol)


_cached_locations = lru_cache(maxsize=128)(_locate)


def locations(grid: Grid, symbol: str) -> list[Point]:
    try:
        found = _cached_locations(grid,symbol)
    except TypeError:
        found = _locate(grid,symbol)
    return list(found)
