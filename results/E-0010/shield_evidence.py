"""Сопоставить доказуемые исключения ходов и совместимые геометрии с реальными поездками."""
from collections import Counter
from dataclasses import replace
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import PhysicalRoutes, read, transition
from maze.compatible_loops import layout
from maze.environment import Environment
from maze.grid import destination
from maze.route_shield import completion_status


def main():
    losses = {r["map"]:r for r in map(json.loads,(HERE/"second-window-last-waypoints.jsonl").read_bytes().splitlines())}
    runs = read("second-window")
    results = []
    for split, name, _, _, level in cases():
        if name not in losses:
            continue
        clear_episode_caches()
        loss = losses[name]["loss"]
        env, history = Environment(level), []
        for char in runs[name]["actions"][:loss["step"]-1]:
            v = env.view
            history.append((v.position,(v.fuel_distance,v.repair_distance)))
            env.step(MOVES[int(char)])
        v = env.view
        old = history[-1] if v.moved else (None,(None,None))
        gauges = v.fuel_distance, v.repair_distance
        states = {}
        for move in MOVES:
            if destination(v.memory,v.position,move) is None:
                continue
            status, mask, searched = completion_status(v, level.goal, gauges, *old, move)
            entry = dict(status=status, searched=searched)
            states[move.name] = entry
            if status == "impossible":
                assert move.name not in loss["viable"], (name,move)
            if mask is None:
                continue
            model = layout(v.memory, level.goal, mask)
            current = replace(level, grid=model.grid)
            comparison = Environment(current)
            width = len(v.memory[0])
            assert all(c == "?" or (c=="#")==(model.grid[y][x]=="#")
                       for y,row in enumerate(v.memory) for x,c in enumerate(row))
            def matches(point, measured):
                return all(comparison._lengths[k].get(point) == d for k,d in zip("FR",measured))
            assert matches(v.position,gauges) and (old[0] is None or matches(*old)), (name,move,"датчики")
            routes = PhysicalRoutes(current)
            remaining = v.fuel+v.health+100*len(model.sites)*len(model.sites)+8*len(model.grid)*width
            assert transition(routes,(v.position,v.fuel,v.health),move,remaining), (name,move,"достижимость")
            entry["same_history"] = all(matches(pos, measured) for pos,measured in history)
            entry["in_time"] = transition(routes,(v.position,v.fuel,v.health),move,env.limit-env.steps-1)
            entry["mask"] = hex(mask)
        results.append(dict(map=name, split=split, step=loss["step"], parent=loss["move"], safe=loss["viable"],
                            start=not(loss["refuels"] and loss["repairs"]), alternatives=states))
    (HERE/"shield-evidence.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results),encoding="utf8")
    print(Counter(r["alternatives"][r["parent"]]["status"] for r in results),flush=True)


if __name__ == "__main__":
    main()
