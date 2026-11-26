"""Внешняя оценка двух замороженных траекторий, включая первые необратимые ходы."""
import argparse
from collections import Counter
from heapq import heappop, heappush
from itertools import count
import json

from bench import HERE, MOVES, cases
from maze.environment import Environment, Status
from maze.frontier_automaton import _arrive
from maze.grid import destination, locations, wave


def read(name):
    rows = [json.loads(s) for s in (HERE/f"{name}.jsonl").read_bytes().splitlines()]
    assert len({r["map"] for r in rows}) == len(rows) == 336
    return {r["map"]: r for r in rows}


class PhysicalRoutes:
    """Перебор первых посещаемых сервисов с отсечением по двум остаткам и времени."""
    def __init__(self, level):
        self.goal = level.goal
        self.grid = level.grid
        self.stops = [(p, kind) for kind in "FRB" for p in locations(level.grid, kind)]
        self.lengths = {p: {q: d for q, d, _ in wave(level.grid, [p])} for p, _ in self.stops}

    def allows(self, p, f, h, remaining):
        serial = count()
        heap = [(0, next(serial), p, f, h)]
        seen = {}
        while heap:
            used, _, point, fuel, health = heappop(heap)
            if point == self.goal:
                return used <= remaining
            if min(fuel, health) <= 0 or used >= remaining:
                continue
            stocks = seen.setdefault(point, [])
            if any(a >= fuel and b >= health for a,b in stocks):
                continue
            stocks.append((fuel, health))
            for site, kind in self.stops:
                d = self.lengths[site].get(point)
                if not d or used+d > remaining:
                    continue
                after = _arrive(kind, d, fuel, health)
                if after:
                    heappush(heap, (used+d, next(serial), site, *after))
        return False


def replay(level, row):
    env = Environment(level)
    track = []
    for char in row["actions"]:
        v = env.view
        move = MOVES[int(char)]
        track.append((v.position, v.fuel, v.health, v.fuel_distance, v.repair_distance,
                      move, env.refuels, env.repairs))
        env.step(move)
    assert (env.status.name,env.steps,env.refuels,env.repairs) == tuple(
        row[k] for k in ("status","steps","refuels","repairs"))
    return track, env


def transition(routes, before, move, left):
    p, f, h = before[:3]
    q = destination(routes.grid, p, move)
    if q is None:
        return False
    kind = routes.grid[q[1]][q[0]]
    after = _arrive(kind, 1, f, h)
    return bool(after and routes.allows(q, *after, left))


def first_loss(routes, track, env):
    lo, hi = 0, len(track)-1
    while lo < hi:
        i = (lo+hi)//2
        if routes.allows(*track[i+1][:3], env.limit-i-1):
            lo = i+1
        else:
            hi = i
    p, f, h, df, dr, move, refuels, repairs = track[lo]
    after = track[lo+1][3:5] if lo+1 < len(track) else (env.view.fuel_distance, env.view.repair_distance)
    viable = [m.name for m in MOVES if transition(routes, track[lo], m, env.limit-lo-1)]
    assert viable and move.name not in viable
    return dict(step=lo+1, pos=p, fuel=f, health=h, dF=df, dR=dr, move=move.name, viable=viable,
                delta=[b-a if a is not None and b is not None else None for a,b in zip((df,dr),after)],
                refuels=refuels,repairs=repairs)


def compare(parent_name, child_name):
    a,b = read(parent_name), read(child_name)
    output = HERE/f"{parent_name}--{child_name}.jsonl"
    partial = output.with_suffix(".partial")
    existing = output if output.exists() else partial
    saved = {}
    if existing.exists():
        for line in existing.read_bytes().splitlines():
            r = json.loads(line)
            name = r["map"]
            if name in saved or any(r[key] != [rows[name]["status"],rows[name]["steps"]]
                                    for key,rows in (("parent",a),("child",b))):
                raise ValueError(f"Несовместимый журнал {existing}, карта {name}")
            saved[name] = r
        if len(saved) == len(a):
            if existing == partial:
                partial.replace(output)
            return
        if existing == output:
            if partial.exists():
                raise ValueError(f"Найдены два незаконченных сравнения {output.name}")
            output.rename(partial)
    counts = {}
    with partial.open("a", encoding="utf8") as out:
        for split, name, tier, family, level in cases():
            if name in saved:
                counts.setdefault(split, Counter())[saved[name]["verdict"]] += 1
                continue
            old, new = a[name], b[name]
            verdict = ("сохранён" if old["status"]=="SUCCESS" else "приобретён") if new["status"]=="SUCCESS" else (
                "потерян" if old["status"]=="SUCCESS" else "оба отказа")
            counts.setdefault(split, Counter())[verdict] += 1
            r = dict(map=name, split=split, tier=tier, family=family, verdict=verdict,
                     parent=[old["status"],old["steps"]], child=[new["status"],new["steps"]])
            diff = next((i for i,(x,y) in enumerate(zip(old["actions"],new["actions"])) if x!=y),None)
            if diff is not None or new["status"]!="SUCCESS":
                routes = PhysicalRoutes(level)
                child, env = replay(level, new)
                if diff is not None:
                    s = child[diff]
                    parent_move = MOVES[int(old["actions"][diff])]
                    r["first"] = dict(step=diff+1, pos=s[0], fuel=s[1],health=s[2],dF=s[3],dR=s[4],
                        parent=parent_move.name, child=s[5].name,
                        parent_viable=transition(routes,s,parent_move,env.limit-diff-1),
                        child_viable=transition(routes,s,s[5],env.limit-diff-1))
                if env.status != Status.SUCCESS:
                    r["child_loss"] = first_loss(routes,child,env)
                if old["status"]!="SUCCESS":
                    if diff is None:
                        r["parent_loss"] = r["child_loss"]
                    else:
                        parent, parent_env = replay(level, old)
                        r["parent_loss"] = first_loss(routes,parent,parent_env)
            else:
                assert old["actions"] == new["actions"]
            out.write(json.dumps(r,ensure_ascii=False)+"\n")
            out.flush()
    partial.replace(output)
    print(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent")
    parser.add_argument("child")
    args = parser.parse_args()
    compare(args.parent,args.child)


if __name__ == "__main__":
    main()
