"""Найти на независимой достройке лабиринта одинаковую историю восприятия с несовместимыми первыми ходами."""
from dataclasses import replace
from bisect import bisect_left
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import PhysicalRoutes, read, transition
from maze.compatible_loops import compatible_masks
from maze.environment import Environment, Status
from maze.grid import locations, wave
from scripts.audit_automaton import Reachability

def completion(base, view, mask):
    w, h = len(view.memory[0]), len(view.memory)
    markers = {p: k for k in "FR" for p in locations(view.memory, k)}
    markers[base.goal] = "B"
    grid = tuple("".join(markers.get((x,y), ".") if 0 < x < w-1 and 0 < y < h-1
                         and mask & (1 << (y*w+x)) else "#" for x in range(w)) for y in range(h))
    connected = {p for p, _, _ in wave(grid, [base.start])}
    if not markers.keys() <= connected:
        return None
    grid = tuple("".join(c if (x,y) in connected else "#" for x,c in enumerate(row)) for y,row in enumerate(grid))
    if any(c != "?" and c != grid[y][x] for y,row in enumerate(view.memory) for x,c in enumerate(row)):
        return None
    return replace(base, grid=grid)


def safe_commands(env, oracle):
    return sorted(m.name for m, q in oracle.edges[env.position]
                  if oracle.possible(q, *oracle.after(q, env.fuel, env.health), env.limit - env.steps - 1))


def finish(env, oracle):
    remaining = env.limit - env.steps
    length = bisect_left(range(remaining + 1), True,
                         key=lambda time: oracle.possible(env.position, env.fuel, env.health, time))
    assert 0 < length <= remaining
    moves = []
    for left in range(length - 1, -1, -1):
        move = next(m for m, q in oracle.edges[env.position]
                    if oracle.possible(q, *oracle.after(q, env.fuel, env.health), left))
        moves.append(move)
        env.step(move)
    assert env.status == Status.SUCCESS
    return "".join(str(MOVES.index(m)) for m in moves)


def verify():
    records = {r["map"]: r for r in map(json.loads,(HERE/"indistinguishable.jsonl").read_bytes().splitlines())
               if r["certificate"]}
    count = 0
    for _, name, _, _, base in cases():
        if name not in records:
            continue
        row, cert = records[name], records[name]["certificate"]
        grids = (base.grid, tuple(cert["grid"]))
        assert grids[0] != grids[1]
        pair = [Environment(replace(base,grid=g)) for g in grids]
        for c in row["prefix"]:
            assert pair[0].view == pair[1].view
            for env in pair:
                env.step(MOVES[int(c)])
        assert pair[0].view == pair[1].view
        for env, key, route in zip(pair, ("original","alternative"), ("finish_actual","finish_alternative")):
            oracle = PhysicalRoutes(env.level)
            state = env.position, env.fuel, env.health
            safe = sorted(m.name for m in MOVES if transition(oracle,state,m,env.limit-env.steps-1))
            assert safe == cert[key] and MOVES[int(cert[route][0])].name in safe
            for c in cert[route]:
                env.step(MOVES[int(c)])
            assert env.status == Status.SUCCESS
        assert set(cert["original"]).isdisjoint(cert["alternative"])
        count += 1
    assert count == len(records)
    print("Независимо проверены пары",count,"и финишные маршруты",2*count,flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="воспроизвести сохранённые пары и проверить все команды")
    if parser.parse_args().verify:
        verify()
        return
    controls = {r["map"]:r for r in map(json.loads,(HERE/"service-ambiguity-controls.jsonl").read_bytes().splitlines())}
    runs = read("street-trunk")
    rows = []
    for split, name, _, _, base in cases():
        if name not in controls:
            continue
        clear_episode_caches()
        prefix = runs[name]["actions"][:controls[name]["step"] - 1]
        env = Environment(base)
        history = [env.view]
        for c in prefix:
            history.append(env.step(MOVES[int(c)]))
        v = env.view
        old = (history[-2].fuel_distance,history[-2].repair_distance) if v.moved else (None,None)
        prior = history[-2].position if v.moved else None
        masks = compatible_masks(v.memory, base.goal, v.position, (v.fuel_distance,v.repair_distance), prior, old)
        saved = None
        for mask in masks:
            level = completion(base,v,mask)
            if level is None:
                continue
            alt = Environment(level)
            if alt.view != history[0] or any(alt.step(MOVES[int(c)]) != expected for c, expected in zip(prefix, history[1:])):
                continue
            oracle = Reachability(level)
            choices = safe_commands(alt, oracle)
            if not choices or set(choices).intersection(controls[name]["safe"]):
                continue
            actual = Reachability(base)
            original = safe_commands(env, actual)
            assert original and original == sorted(controls[name]["safe"])
            saved = dict(grid=level.grid, original=original, alternative=choices,
                         finish_actual=finish(env,actual), finish_alternative=finish(alt,oracle))
            break
        rows.append(dict(map=name,split=split,prefix=prefix, samples=len(masks), certificate=saved))
    (HERE/"indistinguishable.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf8")
    print("Проверено",len(rows),"полных историй, пар с несовместимыми ходами",sum(bool(r["certificate"]) for r in rows), flush=True)


if __name__ == "__main__":
    main()
