"""Фиксированные выборки опыта. Обучение читает только train, val и отпечатки test."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .interface import Grid
from .levels import Level

DATA = Path(__file__).with_name("datasets")
STAGES = (1, 2, 3)


def geometry(grid: Grid) -> str:
    """Отпечаток стен, общий для восьми ориентаций и любых меток A/B/F/R."""
    rows = tuple("".join("#" if c == "#" else "." for c in row) for row in grid)
    variants = []
    for _ in range(4):
        variants.extend(("\n".join(rows), "\n".join(reversed(rows))))
        rows = tuple("".join(col) for col in zip(*rows[::-1]))
    return hashlib.sha256(min(variants).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    name: str
    stage: int
    level: Level


def load_split(split: str) -> list[Case]:
    if split not in ("train", "val"):
        raise ValueError("Допустимы train и val")
    cases = []
    for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        cases.append(Case(item["name"], item["stage"], Level.parse("\n".join(item["lines"]))))
    return cases


def development_sets() -> tuple[list[Case], list[Case]]:
    manifest = json.loads((DATA / "test-manifest.json").read_text(encoding="utf-8"))
    seen = {item["geometry"] for item in manifest.values()}
    train, val = load_split("train"), load_split("val")
    for cases, count in ((train, 30), (val, 12)):
        if [sum(c.stage == s for c in cases) for s in STAGES] != [count] * 3 or len(cases) != count * 3:
            raise ValueError("Нарушен состав выборки")
        for case in cases:
            key = geometry(case.level.grid)
            if key in seen:
                raise ValueError(f"Повтор геометрии: {case.name}")
            seen.add(key)
    return train, val
