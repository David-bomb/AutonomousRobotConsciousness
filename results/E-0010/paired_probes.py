"""Проверить, можно ли раньше получить различимое наблюдение, сохранив финиш на обеих картах пары."""
from collections import deque
from dataclasses import replace
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes
from indistinguishable import finish
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.grid import cell, destination
from scripts.audit_automaton import Reachability


def signal(env, p):
    x, y = p
    window = tuple(cell(env.level.grid, x+dx, y+dy) for dy in (-1,0,1) for dx in (-1,0,1))
    return window, env._lengths["F"].get(p), env._lengths["R"].get(p)


def common_probe(pair, routes, state, remaining):
    grid = pair[0].level.grid
    queue = deque([(*state, "")])
    reached = {}
    while queue:
        p, f, h, path = queue.popleft()
        stocks = reached.setdefault(p, [])
        if any(old_f >= f and old_h >= h for old_f, old_h in stocks):
            continue
        stocks.append((f,h))
        for i, move in enumerate(MOVES):
            q = destination(grid, p, move)
            if q is None:
                continue
            assert q == destination(pair[1].level.grid, p, move)
            stock = _arrive(cell(grid,*q),1,f,h)
            left = remaining - len(path) - 1
            if left < 0 or not stock or not all(r.allows(q,*stock,left) for r in routes):
                continue
            trial = path + str(i)
            if q == pair[0].level.goal or signal(pair[0], q) != signal(pair[1], q):
                return trial
            queue.append((q, *stock, trial))
    return None


def main():
    records = {r["map"]:r for r in map(json.loads,(HERE/"indistinguishable.jsonl").read_bytes().splitlines())
               if r["certificate"]}
    results = []
    for split, name, _, _, base in cases():
        if name not in records:
            continue
        row = records[name]
        prefix = row["prefix"]
        levels = [base, replace(base,grid=tuple(row["certificate"]["grid"]))]
        pair = [Environment(level) for level in levels]
        routes = [PhysicalRoutes(level) for level in levels]
        origin = pair[0]
        states = [(origin.position,origin.fuel,origin.health)]
        for c in prefix:
            origin.step(MOVES[int(c)])
            states.append((origin.position,origin.fuel,origin.health))
        found = None
        for t in range(len(states)-1, -1, -1):
            path = common_probe(pair,routes,states[t],origin.limit-t)
            if path is None:
                continue
            assert t < len(prefix)
            pair = [Environment(level) for level in levels]
            for c in prefix[:t] + path:
                assert pair[0].view == pair[1].view
                for env in pair:
                    env.step(MOVES[int(c)])
            assert pair[0].view != pair[1].view or pair[0].position == base.goal
            suffix = [finish(e,Reachability(e.level)) if e.position != base.goal else "" for e in pair]
            found = dict(after=t, shared=path, copied=prefix[:t], suffix=suffix)
            break
        results.append(dict(map=name,split=split,prefix=prefix,probe=found))
    (HERE/"paired-probes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results),encoding="utf8")
    print([(r["map"], r["probe"]["after"], r["probe"]["shared"]) for r in results if r["probe"]],flush=True)


if __name__ == "__main__":
    main()
