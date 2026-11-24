"""Текстовый уровень: название, начальные ресурсы и прямоугольная карта."""
from dataclasses import dataclass
from pathlib import Path

from .grid import locations
from .interface import Grid, Point

LEVELS = Path(__file__).with_name("levels")


@dataclass(frozen=True)
class Level:
    title: str
    grid: Grid
    start: Point
    goal: Point
    fuel: int
    health: int

    @classmethod
    def parse(cls, source: str):
        title, resources, *rows = source.strip().splitlines()
        fuel, health = map(int, resources.split())
        if not (0 < fuel <= 100 and 0 < health <= 100):
            raise ValueError("Начальные ресурсы должны лежать в диапазоне 1–100")
        if not rows or not rows[0] or any(len(r) != len(rows[0]) or set(r) - set("#.ABFR") for r in rows):
            raise ValueError("Нужна прямоугольная карта из символов #.ABFR")
        grid = tuple(rows)
        starts, goals = locations(grid, "A"), locations(grid, "B")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("На карте нужны один старт A и один финиш B")
        return cls(title, tuple(row.replace("A", ".") for row in rows), starts[0], goals[0], fuel, health)


def catalogue() -> dict[str, Level]:
    return {p.stem: Level.parse(p.read_text(encoding="utf-8")) for p in sorted(LEVELS.glob("*.txt"))}
