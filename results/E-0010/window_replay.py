"""Внешняя проверка: довести до первого нового окна, вернуть управление и доехать до конца."""
from argparse import ArgumentParser
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from heapq import heappop, heappush
from itertools import count
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import read
from maze.environment import Environment, Status
from maze.frontier_automaton import _arrive, _frontiers
from maze.grid import cell, destination, locations
from maze.street_trunk import StreetTrunk


def trails_to_first_windows(view, goal, limit):
    memory = view.memory
    exits = set(_frontiers(memory)) | {goal}
    terminals = exits | {p for kind in "FR" for p in locations(memory, kind)}
    legs = {}
    def from_stop(origin):
        if origin in legs:
            return legs[origin]
        paths = {}
        pending = deque([(origin, "")])
        seen = {origin}
        while pending:
            p, prefix = pending.popleft()
            for i, move in enumerate(MOVES):
                q = destination(memory, p, move)
                if q is None or q in seen:
                    continue
                seen.add(q)
                trail = prefix + str(i)
                if q in terminals:
                    paths[q] = trail
                else:
                    pending.append((q, trail))
        legs[origin] = paths
        return paths

    serial = count()
    queue = [(0, next(serial), view.position, view.fuel, view.health, "")]
    seen = {}
    result = {}
    while queue:
        length, _, p, f, h, prefix = heappop(queue)
        for q, trail in from_stop(p).items():
            d = len(trail)
            after = _arrive("B" if q == goal else cell(memory, *q), d, f, h)
            if length + d > limit or after is None:
                continue
            signature = length + d, *after
            saved = seen.setdefault(q, {})
            if any(t <= signature[0] and fuel >= after[0] and health >= after[1] for t, fuel, health in saved):
                continue
            for old in list(saved):
                if signature[0] <= old[0] and after[0] >= old[1] and after[1] >= old[2]:
                    del saved[old]
            saved[signature] = prefix + trail
            if q in exits:
                result[q] = saved
            else:
                heappush(queue, (signature[0], next(serial), q, *after, prefix + trail))
    return result


def trial(job):
    case, run, event = job
    split, name, _, _, level = case
    clear_episode_caches()
    env, parent = Environment(level), StreetTrunk()
    for c in run["actions"][:event["step"] - 1]:
        assert parent.choose(env.view) == MOVES[int(c)]
        env.step(MOVES[int(c)])
    trails = trails_to_first_windows(env.view, level.goal, env.limit - env.steps)
    listed = {(tuple(e["point"]), a["length"], a["fuel"], a["health"]): a["viable"]
              for e in event["exits"] for a in e["arrivals"]}
    assert {(p, *key) for p, saved in trails.items() for key in saved} == listed.keys()
    outcomes = []
    for (p, length, f, h), viable in listed.items():
        if not viable:
            continue
        forced = trails[p][length, f, h]
        sim, robot = deepcopy(env), deepcopy(parent)
        for c in forced:
            assert sim.view.memory == env.view.memory
            robot.choose(sim.view)
            sim.step(MOVES[int(c)])
        assert (sim.position, sim.fuel, sim.health) == (p, f, h)
        assert sim.position == level.goal or sim.view.memory != env.view.memory
        suffix = []
        while sim.status == Status.RUNNING:
            action = robot.choose(sim.view)
            suffix.append(str(MOVES.index(action)))
            sim.step(action)
        outcomes.append(dict(point=p, fuel=f, health=h, forced=forced, status=sim.status.name,
                             steps=sim.steps, suffix="".join(suffix)))
    return dict(map=name, split=split, step=event["step"], outcomes=outcomes)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    events = {r["map"]: r for line in (HERE / "next-observation.jsonl").read_bytes().splitlines()
              if (r := json.loads(line))}
    path = HERE / "window-replay.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
    done = {r["map"] for r in rows}
    assert len(done) == len(rows) and done <= events.keys()
    runs = read("street-trunk")
    jobs = [(case, runs[case[1]], events[case[1]]) for case in cases() if case[1] in events and case[1] not in done]
    with ProcessPoolExecutor(args.workers) as pool, path.open("a", encoding="utf8") as out:
        for row in pool.map(trial, jobs):
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
    assert len(rows) == len(events) == 24
    print(Counter((r["split"], any(o["status"] == "SUCCESS" for o in r["outcomes"])) for r in rows))


if __name__ == "__main__":
    main()
