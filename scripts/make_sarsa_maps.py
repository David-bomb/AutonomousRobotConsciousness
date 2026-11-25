"""Воспроизводит train и val. Запуск из корня: python -m scripts.make_sarsa_maps."""
import json
import random

from maze.datasets import DATA, STAGES, geometry
from maze.grid import wave
from maze.levels import Level


def draw(stage, rng):
    low, high = ((3, 5), (5, 8), (8, 13))[stage - 1]
    w, h = (rng.randint(low, high) * 2 + 1 for _ in range(2))
    rows = [["#"] * w for _ in range(h)]
    origin = (1, 1)
    rows[1][1] = "."
    stack = [origin]
    while stack:
        x, y = stack[-1]
        options = [(x + dx, y + dy) for dx, dy in ((0, 2), (2, 0), (0, -2), (-2, 0))
                   if 0 < x + dx < w-1 and 0 < y + dy < h-1 and rows[y+dy][x+dx] == "#"]
        if options:
            i, j = rng.choice(options)
            rows[j][i] = rows[(j+y)//2][(i+x)//2] = "."
            stack.append((i, j))
        else:
            stack.pop()
    # Петли и открытые площадки дают как ортогональные, так и диагональные ходы.
    for _ in range(w*h // (6 + stage*2)):
        x, y = rng.randrange(1, w-1), rng.randrange(1, h-1)
        if any(rows[y+dy][x+dx] == "." for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            rows[y][x] = "."
    return rows


def make_case(stage, index, rng):
    while True:
        rows = draw(stage, rng)
        grid = tuple("".join(row) for row in rows)
        roads = [(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == "."]
        start = rng.choice(roads)
        ds = {p: d for p, d, _ in wave(grid, [start])}
        bounds = ((3, 18), (10, 36), (20, 65))[stage-1]
        goals = [p for p, d in ds.items() if bounds[0] <= d <= bounds[1]]
        if not goals:
            continue
        goal = rng.choice(goals)
        length = ds[goal]
        fuel = health = 100
        sites = {}
        if stage > 1:
            dg = {p: d for p, d, _ in wave(grid, [goal])}
            branches = [p for p in roads if ds[p] + dg[p] > length]
            first = [p for p in branches if 2 <= ds[p] <= length // (2 if stage == 2 else 3)]
            if not first:
                continue
            a = rng.choice(first)
            kind = "F" if index % 2 == 0 else "R"
            other = "R" if kind == "F" else "F"
            sites[a] = kind
            stocks = {kind: min(length - 1, ds[a] + rng.randint(5, 12)), other: 100}
            if stage == 3:
                da = {p: d for p, d, _ in wave(grid, [a])}
                second = [p for p in branches if ds[p] > ds[a] and
                          length//3 <= ds[a]+da[p] < length-2 and da[p]+dg[p] < 100]
                if not second:
                    continue
                b = rng.choice(second)
                sites[b] = other
                stocks[other] = min(length-1, ds[a]+da[b]+rng.randint(6, 16))
            extras = [p for p in branches if p not in sites]
            rng.shuffle(extras)
            for symbol in ([other] if stage == 2 else [kind, other]):
                if extras:
                    sites[extras.pop()] = symbol
            fuel, health = stocks["F"], stocks["R"]
        for (x, y), c in {start: "A", goal: "B", **sites}.items():
            rows[y][x] = c
        return [f"Ступень {stage}, карта {index+1}", f"{fuel} {health}", *map("".join, rows)]


def main():
    manifest = json.loads((DATA / "test-manifest.json").read_text(encoding="utf-8"))
    seen = {item["geometry"] for item in manifest.values()}
    for split, seed, count in (("train", 21701, 30), ("val", 83497, 12)):
        rng = random.Random(seed)
        records = []
        for stage in STAGES:
            for index in range(count):
                while True:
                    lines = make_case(stage, index, rng)
                    key = geometry(Level.parse("\n".join(lines)).grid)
                    if key not in seen:
                        break
                seen.add(key)
                records.append(json.dumps(dict(name=f"{split}-{stage}-{index+1:02}",
                                               stage=stage, lines=lines), ensure_ascii=False))
        (DATA / f"{split}.jsonl").write_text("\n".join(records)+"\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
