"""Проверка двух гипотез по накопленным наблюдениям и точному внешнему поиску."""
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Использование: check_scouting.py <стенд E-0006> <новый JSON>")
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from diagnose_unsolved import ROOT, checked_drive, shortest_route
from maze.environment import Environment
from maze.frontier_automaton import _arrive, _paths
from maze.grid import cell, destination, locations
from maze.interface import Move
from maze.levels import Level
from scripts.audit_automaton import Reachability


def known_service_pair(view, f, h):
    roads = _paths(view.memory, view.position)
    sites = [(p, k) for k in "FR" for p in locations(view.memory, k)]
    for p, kind in sites:
        if p not in roads:
            continue
        # Возврат на текущую станцию уже учтён при расчёте f, h.
        after = (f, h) if p == view.position else _arrive(kind, roads[p][0], f, h)
        if not after:
            continue
        onward = _paths(view.memory, p)
        for q, other in sites:
            if kind != other and q in onward and _arrive(other, onward[q][0], *after):
                return [p, q]
    return None


def probe_options(level, record):
    env, oracle = Environment(level), Reachability(level)
    past = []
    for name in record["prefix"]:
        past.append((env.view, Move[name]))
        env.step(Move[name])
    assert list(env.view.memory) == record["critical"]["memory"]
    critical_memory = env.view.memory
    possible = None
    for i in range(len(past)-1, -1, -1):
        v, original = past[i]
        for move in Move:
            p = destination(v.memory, v.position, move)
            if not p or move == original or p == level.goal:
                continue
            x, y = p
            # Новые клетки, которые штатный агент так и не увидел к решающему ходу.
            count = sum(cell(critical_memory, x+dx, y+dy) == "?"
                        for dx in (-1,0,1) for dy in (-1,0,1))
            if not count:
                continue
            out = _arrive(cell(v.memory, *p), 1, v.fuel, v.health)
            back = out and _arrive(cell(v.memory, *v.position), 1, *out)
            if not back:
                continue
            pair = known_service_pair(v, *back)
            if possible and not pair:
                continue
            if not oracle.possible(v.position, *back, env.limit-i-2):
                continue
            sample = dict(step=i+1, position=v.position, probe=move.name, new_cells=count,
                          fuel_after=back[0], health_after=back[1], known_service_pair=pair)
            possible = possible or sample
            if pair:
                return dict(possible=possible, with_known_service_pair=sample)
    return dict(possible=possible, with_known_service_pair=None)


def check_ambiguous_start(level):
    grid = [list(line) for line in level.grid]
    view = Environment(level).view
    edits = [(8,7,"#","."), (3,11,".","#")]
    for x,y,old,new in edits:
        assert view.memory[y][x] == "?" and grid[y][x] == old
        grid[y][x] = new
    alternative = Level(level.title, tuple("".join(line) for line in grid), level.start,
                        level.goal, level.fuel, level.health)
    assert Environment(alternative).view == view
    evidence = {}
    for key, scenario in (("original", level),("alternative", alternative)):
        oracle = Reachability(scenario)
        limit = Environment(scenario).limit
        viable = [m.name for m,q in oracle.edges[scenario.start]
                  if oracle.possible(q, *oracle.after(q, view.fuel, view.health), limit-1)]
        route = shortest_route(oracle, scenario.start, view.fuel, view.health, limit)
        proof = checked_drive(scenario, route)
        evidence[key] = dict(viable=viable, moves=[m.name for m in route], **proof)
    assert evidence["original"]["viable"] == ["WEST"] and evidence["alternative"]["viable"] == ["EAST"]
    return dict(map=level.title, edits=edits, start=view.position, window=view.window,
                fuel=view.fuel, health=view.health, dx=view.dx, dy=view.dy,
                fuel_distance=view.fuel_distance, repair_distance=view.repair_distance, **evidence)


def main(output):
    records = json.loads((ROOT/"replay/stress/unsolved-reserve.json").read_text(encoding="utf-8"))
    layouts = [json.loads(s) for s in (ROOT/"source/ARC-stress-100/maps.jsonl").read_bytes().splitlines()]
    levels = {m["name"]:Level.parse("\n".join([m["name"], f'{m["fuel"]} {m["health"]}', *m["grid"]]))
              for m in layouts}
    data = dict(probes={r["map"]:probe_options(levels[r["map"]],r) for r in records},
                same_observation=check_ambiguous_start(levels["H4-25"]))
    with output.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.write("\n")


if __name__ == "__main__":
    main(Path(sys.argv[2]))
