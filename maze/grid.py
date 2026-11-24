"""Один закон движения для симулятора и поиска по переданной карте."""
from collections import deque
from collections.abc import Iterable, Iterator

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


def wave(grid: Grid, starts: Iterable[Point]) -> Iterator[tuple[Point, int, Move | None]]:
    """Выдать клетки в порядке длины пути и первый ход от соответствующего истока."""
    seen = set(starts)
    queue = deque((p, 0, None) for p in sorted(seen))
    while queue:
        point, length, first = queue.popleft()
        yield point, length, first
        for move in Move:
            nxt = destination(grid, point, move)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, length + 1, first or move))


def locations(grid: Grid, symbol: str) -> list[Point]:
    return [(x, y) for y, row in enumerate(grid) for x, char in enumerate(row) if char == symbol]
