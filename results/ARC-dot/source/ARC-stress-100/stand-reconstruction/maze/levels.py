"""Minimal text adapter for validated benchmark grids; not the project's catalogue."""
from dataclasses import dataclass
from .grid import locations


@dataclass(frozen=True)
class Level:
    title: str
    grid: tuple
    start: tuple
    goal: tuple
    fuel: int
    health: int

    @classmethod
    def parse(cls, source):
        title, stocks, *rows = source.splitlines()
        fuel, health = map(int, stocks.split())
        grid = tuple(rows)
        return cls(title, grid, *locations(grid, "A"), *locations(grid, "B"), fuel, health)
