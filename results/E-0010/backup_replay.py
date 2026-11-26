"""Внешняя проверка основного и запасного окон после первого различия в действиях."""
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import PhysicalRoutes, read
from maze.environment import Environment
from maze.first_window import first_events, window_arrivals
from maze.frontier_automaton import _frontiers
from maze.frontier_reserve import _lengths
from maze.grid import cell
from maze.overlapping_windows import opening_cells
from maze.second_window import SecondWindow, closed_probability
from maze.specific_paths import SpecificPaths


def revealed(env, actions, routes, backup):
    sim = deepcopy(env)
    first = None
    next_window = None
    for c in actions:
        previous = sim.view.memory
        sim.step(MOVES[int(c)])
        if first is None:
            if sim.view.memory != previous or sim.position == routes.goal:
                first = dict(point=sim.position, steps=sim.steps-env.steps, stocks=[sim.fuel, sim.health],
                             viable=routes.allows(sim.position, sim.fuel, sim.health, sim.limit-sim.steps))
        else:
            if next_window is None and sim.view.memory != previous:
                next_window = dict(point=sim.position, step=sim.steps)
            if sim.position == backup:
                first["backup_visit"] = dict(step=sim.steps, viable=routes.allows(sim.position, sim.fuel, sim.health,
                                                                                 sim.limit-sim.steps))
                break
    if first is not None:
        first["next_window"] = next_window
    return first


def main():
    parent, child = "specific-paths", "second-window"
    differences = {r["map"]: r for r in map(json.loads, (HERE/f"{parent}--{child}.jsonl").read_bytes().splitlines())
                   if "first" in r}
    runs = [read(parent), read(child)]
    records = []
    for split, name, _, _, level in cases():
        if name not in differences:
            continue
        clear_episode_caches()
        row = differences[name]
        env, routes = Environment(level), PhysicalRoutes(level)
        start = row["first"]["step"]-1
        for c in runs[0][name]["actions"][:start]:
            env.step(MOVES[int(c)])
        view = env.view
        points = tuple(sorted(p for p, _, _ in first_events(view.memory, view.position, frozenset(_frontiers(view.memory)))))
        arrivals = list(window_arrivals(view))
        model = SecondWindow()
        curves = model.probabilities(view.memory, level.goal, points)
        plans = []
        for i, bot in enumerate((SpecificPaths(), model)):
            arrival = bot.pick(view, level.goal, arrivals)
            assert arrival.first == MOVES[int(runs[i][name]["actions"][start])], name
            a = arrival.point
            stock = min(arrival.fuel, arrival.health)
            lengths = _lengths(view.memory, a)
            alt = max((p for p in points if p != a and lengths.get(p, 101) < stock),
                      key=lambda p: curves[p][stock-lengths[p]], default=None)
            opens = opening_cells(view.memory, a)
            fallback = None
            if alt is not None:
                d = lengths[alt]
                fallback = dict(point=alt, length=d, share=float(curves[alt][stock-d]),
                                shared=len(opens & opening_cells(view.memory, alt)),
                                viable_without_refill=routes.allows(alt, arrival.fuel-d, arrival.health-d,
                                                                    env.limit-start-arrival.length-d))
            plans.append(dict(point=a, length=arrival.length, stocks=[arrival.fuel, arrival.health],
                              share=float(curves[a][stock]), p_closed=float(closed_probability(view.memory, a, level.goal)),
                              closed=all(cell(level.grid,*p) == "#" for p in opens), backup=fallback,
                              actual=revealed(env, runs[i][name]["actions"][start:], routes, alt)))
        records.append(dict(map=name, split=split, step=start+1, verdict=row["verdict"], plans=plans))
    (HERE/"backup-replay.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf8")
    print(len(records), "пар разных поездок")


if __name__ == "__main__":
    main()
