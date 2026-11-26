"""100 resource mazes, independent generator and full-map reachability proof."""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from random import Random

SEED = 20318577
ADJACENT = ((0, -1), (1, 0), (0, 1), (-1, 0))
STEPS = ADJACENT + ((1, -1), (1, 1), (-1, 1), (-1, -1))
FAMILIES = ("corridors", "branching", "loops", "rooms")


@dataclass(frozen=True)
class Tier:
    name: str
    half_size: tuple
    minimum_direct: int
    first_margin: tuple
    second_margin: tuple


TIERS = (Tier("H1", (9, 13), 28, (5, 8), (7, 12)),
         Tier("H2", (11, 14), 38, (3, 6), (5, 9)),
         Tier("H3", (13, 17), 48, (2, 4), (3, 6)),
         Tier("H4", (15, 18), 58, (0, 2), (1, 4)))


def geometry(grid):
    """Исключает дубли стен, включая повороты, отражения и смену меток."""
    rows = tuple("".join("#" if c == "#" else "." for c in row) for row in grid)
    variants = []
    for _ in range(4):
        variants.extend(("\n".join(rows), "\n".join(reversed(rows))))
        rows = tuple("".join(col) for col in zip(*rows[::-1]))
    return sha256(min(variants).encode()).hexdigest()


def maze_shape(rng, tier, family):
    w, h = (2 * rng.randint(*tier.half_size) + 1 for _ in range(2))
    grid = [["#"] * w for _ in range(h)]
    start = rng.randrange(1, w-1, 2), rng.randrange(1, h-1, 2)
    grid[start[1]][start[0]] = "."
    frontier = [start]
    while frontier:
        j = rng.randrange(len(frontier)) if family == "branching" else len(frontier)-1
        x, y = frontier[j]
        candidates = [(x+2*dx, y+2*dy) for dx, dy in ADJACENT
                      if 0 < x+2*dx < w-1 and 0 < y+2*dy < h-1
                      and grid[y+2*dy][x+2*dx] == "#"]
        if not candidates:
            frontier[j] = frontier[-1]
            frontier.pop()
            continue
        X, Y = rng.choice(candidates)
        grid[Y][X] = grid[(y+Y)//2][(x+X)//2] = "."
        frontier.append((X, Y))
    if family == "loops":
        options = [(x, y) for y in range(1, h-1) for x in range(1, w-1)
                   if grid[y][x] == "#" and ((grid[y-1][x] == grid[y+1][x] == ".") or
                                               (grid[y][x-1] == grid[y][x+1] == "."))]
        for x, y in rng.sample(options, min(len(options), w*h//30)):
            grid[y][x] = "."
    if family == "rooms":
        centers = [(x, y) for x in range(3, w-3, 4) for y in range(3, h-3, 4)]
        for x, y in rng.sample(centers, min(len(centers), w*h//110)):
            radius = rng.choice((1, 2))
            for j in range(y-1, y+2):
                for i in range(x-radius, x+radius+1):
                    grid[j][i] = "."
    return grid


def graph(grid):
    points = {(x, y) for y, row in enumerate(grid) for x, c in enumerate(row) if c != "#"}
    return {p: tuple((x+dx,y+dy) for dx,dy in STEPS
                     if (x+dx,y+dy) in points and
                     (not(dx and dy) or ((x+dx,y) in points and (x,y+dy) in points)))
            for p in sorted(points) for x,y in (p,)}


def lengths(edges, origin):
    found = {origin: 0}
    todo = deque([origin])
    while todo:
        p = todo.popleft()
        for q in edges[p]:
            if q not in found:
                found[q] = found[p] + 1
                todo.append(q)
    return found


def shortest_legal(edges, start, goal, fuel, health, stations, limit):
    dist_to_goal = lengths(edges, goal)
    from_stations = {kind: [lengths(edges, p) for p, k in stations.items() if k == kind]
                     for kind in ("F", "R")}
    nearest = {k: {p: min((d.get(p, float("inf")) for d in w), default=float("inf"))
                   for p in edges} for k,w in from_stations.items()}
    todo = deque([(start, fuel, health, 0)])
    seen = {}
    while todo:
        p, f, h, steps = todo.popleft()
        if p == goal:
            return steps
        d = dist_to_goal.get(p, float("inf"))
        if min(f, h) <= 0 or steps + d > limit:
            continue
        if d <= min(f, h):
            # Do not return yet: another queued state may offer a shorter total.
            limit = min(limit, steps+d)
        if f < min(nearest["F"].get(p, float("inf")), d) or h < min(nearest["R"].get(p, float("inf")), d):
            continue
        prev = seen.setdefault(p, [])
        if any(a >= f and b >= h for a,b in prev):
            continue
        prev.append((f,h))
        for q in edges[p]:
            k = stations.get(q)
            todo.append((q, 100 if k == "F" else f-1, 100 if k == "R" else h-1, steps+1))
    return None


def draft(rng, tier, family):
    grid = maze_shape(rng, tier, family)
    edges = graph(grid)
    for _ in range(22):
        a = rng.choice(tuple(edges))
        ds = lengths(edges, a)
        goals = [p for p, d in ds.items() if tier.minimum_direct <= d <= tier.minimum_direct + 26]
        if not goals:
            continue
        b = rng.choice(goals)
        direct = ds[b]
        dg = lengths(edges, b)
        firsts = [p for p,d in ds.items() if 4 <= d <= min(27, direct//2)
                  and d + dg[p] >= direct + 2]
        rng.shuffle(firsts)
        for first in firsts[:16]:
            da = lengths(edges, first)
            m1 = rng.randint(*tier.first_margin)
            m2 = rng.randint(*tier.second_margin)
            seconds = [p for p,d in da.items()
                       if p not in (a, b, first) and 4 <= d <= 43
                       and ds[first]+d+m2 < min(direct,100)
                       and dg[p]+d <= 99 and dg[p] <= 93
                       and ds[p]+dg[p] >= direct]
            if not seconds:
                continue
            second = rng.choice(seconds)
            small, other = ds[first]+m1, ds[first]+da[second]+m2
            if not (0 < small < direct and 0 < other < direct):
                continue
            first_kind, second_kind = rng.choice((("F", "R"), ("R", "F")))
            stations = {first: first_kind, second: second_kind}
            options = [p for p, x in ds.items() if p not in (a,b,first,second) and
                       x + dg[p] > direct and x >= 4]
            if len(options)<2:
                continue
            stations.update(zip(rng.sample(options,2),("F", "R")))
            fuel, health = (small,other) if first_kind == "F" else (other,small)
            upper = ds[first] + da[second] + dg[second]
            optimum = shortest_legal(edges,a,b,fuel,health,stations,upper)
            if optimum is None:
                raise AssertionError("Constructed certificate failed")
            for (x,y),mark in ({a:"A", b:"B"} | stations).items():
                grid[y][x] = mark
            deg = [len(ns) for ns in edges.values()]
            return dict(grid=["".join(row) for row in grid], fuel=fuel, health=health,
                        direct=direct, optimum=optimum, width=len(grid[0]), height=len(grid),
                        min_start_slack=min(fuel-min(ds[p] for p,k in stations.items() if k=="F"),
                                            health-min(ds[p] for p,k in stations.items() if k=="R")),
                        first_witness_margin=m1, second_witness_margin=m2,
                        dead_ends=deg.count(1), junctions=sum(d>=3 for d in deg),
                        cells=len(edges), edges=sum(deg)//2)
    return None


def generate(excluded=()):
    rng = Random(SEED)
    used = set(excluded)
    for level, tier in enumerate(TIERS):
        families = list(FAMILIES)*6 + [FAMILIES[level]]
        rng.shuffle(families)
        for i,family in enumerate(families,1):
            for _ in range(500):
                data = draft(rng,tier,family)
                if data is None:
                    continue
                fingerprint = geometry(data["grid"])
                if fingerprint in used:
                    continue
                used.add(fingerprint)
                yield dict(name=f"{tier.name}-{i:02}", tier=tier.name, family=family,
                           geometry=fingerprint, **data)
                break
            else:
                raise RuntimeError(f"Failed to generate {tier.name}/{family}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude", type=Path, help="JSON sorted list of previous geometry hashes")
    args = parser.parse_args()
    excluded = json.loads(args.exclude.read_text(encoding="utf-8")) if args.exclude else []
    with args.output.open("x", encoding="utf-8") as f:
        for row in generate(excluded):
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(f"Generated 100 at seed {SEED}: {args.output}")


if __name__ == "__main__":
    main()
