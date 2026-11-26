"""Внешний вопрос: мог ли другой подъезд по уже раскрытым дорогам спасти тот же ход?"""
from collections import Counter, deque
from copy import deepcopy
from heapq import heappop, heappush
from itertools import count
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, read, transition
from maze.environment import Environment, Status
from maze.frontier_automaton import _arrive
from maze.grid import cell, destination


def visible_legs(memory, source, critical, goal):
    queue = deque([(source, "")])
    seen = {source}
    ends = {}
    while queue:
        p, path = queue.popleft()
        for i, move in enumerate(MOVES):
            q = destination(memory, p, move)
            if q is None or q in seen or q == goal:
                continue
            seen.add(q)
            trail = path + str(i)
            kind = cell(memory, *q)
            if q == critical or kind in "FR":
                ends[q] = trail
            if kind not in "FR":
                queue.append((q, trail))
    return ends


def alternative(level, loss, memory, move, routes, limit, origin=None):
    if origin is None:
        origin = Environment(level)
    critical = tuple(loss["pos"])
    serial = count()
    queue = [(origin.steps, next(serial), origin.position, origin.fuel, origin.health, "")]
    seen, legs = {}, {}
    while queue:
        length, _, p, f, h, trail = heappop(queue)
        if p == critical and transition(routes, (p, f, h), move, limit-length-1):
            witness = deepcopy(origin)
            for c in trail:
                assert witness.status == Status.RUNNING
                witness.step(MOVES[int(c)])
            assert (witness.position, witness.fuel, witness.health, witness.steps) == (p, f, h, length)
            return dict(actions=trail, steps=length, fuel=f, health=h,
                        by_original_step=length < loss["step"])
        prior = seen.setdefault(p, [])
        if any(a >= f and b >= h for a, b in prior):
            continue
        prior.append((f, h))
        if p not in legs:
            legs[p] = visible_legs(memory, p, critical, level.goal)
        for q, segment in legs[p].items():
            end = length + len(segment)
            if end >= limit:
                continue
            stocks = _arrive(cell(memory, *q), len(segment), f, h)
            if stocks:
                heappush(queue, (end, next(serial), q, *stocks, trail+segment))
    return None


def main():
    runs = read("street-trunk")
    pairs = [json.loads(b) for b in (HERE/"marked-streets--street-trunk.jsonl").read_bytes().splitlines()]
    failures = {r["map"]:r["child_loss"] for r in pairs if "child_loss" in r}
    totals = Counter()
    with (HERE/"preparation-street-trunk.jsonl").open("x", encoding="utf8") as out:
        for split, name, tier, family, level in cases():
            if name not in failures:
                continue
            env = Environment(level)
            loss = failures[name]
            actions = runs[name]["actions"]
            for char in actions[:loss["step"]-1]:
                env.step(MOVES[int(char)])
            move = MOVES[int(actions[env.steps])]
            routes = PhysicalRoutes(level)
            assert not transition(routes, (env.position, env.fuel, env.health), move, env.limit-env.steps-1)
            witness = alternative(level, loss, env.view.memory, move, routes, env.limit)
            totals["states"] += 1
            totals["witness"] += bool(witness)
            totals["in_time"] += bool(witness and witness["by_original_step"])
            out.write(json.dumps(dict(map=name, split=split, family=family, tier=tier,
                                      critical=loss, alternative=witness))+"\n")
            out.flush()
    print(totals)


if __name__ == "__main__":
    main()
