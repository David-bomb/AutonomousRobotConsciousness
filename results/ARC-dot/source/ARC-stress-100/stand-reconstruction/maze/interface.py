"""Общие единицы измерения и весь доступный вход контроллера."""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

type Point = tuple[int, int]
type Grid = tuple[str, ...]


class Move(Enum):
    NORTH = (0, -1)
    EAST = (1, 0)
    SOUTH = (0, 1)
    WEST = (-1, 0)
    NE = (1, -1)
    SE = (1, 1)
    SW = (-1, 1)
    NW = (-1, -1)


@dataclass(frozen=True)
class Observation:
    memory: Grid
    window: Grid
    position: Point
    dx: int
    dy: int
    fuel_distance: int | None
    repair_distance: int | None
    fuel: int
    health: int
    last_move: Move | None
    moved: bool | None


class Controller(Protocol):
    def choose(self, view: Observation) -> Move: ...
