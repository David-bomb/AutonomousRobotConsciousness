"""Проверить первую достижимую остановку после каждой ещё спасающей команды в критическом состоянии."""
from collections import Counter, deque
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import PhysicalRoutes, read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.frontier_reserve import _lengths, _optimistic
from maze.grid import cell, destination, wave


def waypoints(level, view, remaining, routes):
    stops = dict(routes.stops)
    terminals = frozenset(stops)
    options = {}
    for move in MOVES:
        q = destination(level.grid, view.position, move)
        if q is None:
            continue
        if q in terminals:
            legs = [(q, 0, None)]
        else:
            legs = wave(level.grid, [q], stop_at=terminals)
        available = []
        for point, suffix, _ in legs:
            if point not in terminals or suffix + 1 > remaining:
                continue
            kind = stops[point]
            after = _arrive(kind, suffix + 1, view.fuel, view.health)
            if after and routes.allows(point, *after, remaining - suffix - 1):
                available.append(dict(point=point, kind=kind, length=suffix + 1, stocks=after))
        options[move.name] = available
    return options


def main():
    parser = ArgumentParser(description=__doc__)
    sources = {"simple-exit-paths": ("exit-current", "simple"),
               "specific-paths": ("station-useful-paths", "specific"),
               "second-window": ("specific-paths", "second-window"),
               "ranked-standby": ("standby-turn","ranked-standby")}
    parser.add_argument("--agent", choices=sources, default="simple-exit-paths")
    source = parser.parse_args().agent
    parent, label = sources[source]
    runs = read(source)
    comparison = HERE / f"{parent}--{source}.jsonl"
    losses = {r["map"]: r["child_loss"] for r in map(json.loads, comparison.read_bytes().splitlines())
              if "child_loss" in r}
    assert set(losses) == {k for k, r in runs.items() if r["status"] != "SUCCESS"}
    rows = []
    for split, name, _, _, level in cases():
        if name not in losses:
            continue
        clear_episode_caches()
        loss = losses[name]
        env = Environment(level)
        machine = controller(source) if source=="ranked-standby" else None
        history = deque(maxlen=4)
        previous = env.view
        last_service_memory = None
        for action in runs[name]["actions"][:loss["step"] - 1]:
            previous = env.view
            if machine:
                choice = machine.choose(previous)
                assert choice==MOVES[int(action)]
                history.append((previous.position,previous.fuel_distance,previous.repair_distance,
                                previous.fuel,previous.health,choice.name,machine.activity))
            env.step(MOVES[int(action)])
            if cell(env.view.memory, *env.position) in "FR":
                last_service_memory = env.view.memory
        view = env.view
        options = waypoints(level, view, env.limit - env.steps, PhysicalRoutes(level))
        assert sorted(m for m, stops in options.items() if stops) == sorted(loss["viable"]), (name, options, loss)
        known = _lengths(view.memory, view.position)
        for destinations in options.values():
            for stop in destinations:
                stop["known_length"] = known.get(stop["point"])
        optimistic = _lengths(_optimistic(view.memory), view.position)
        rows.append(dict(map=name, split=split, loss=loss, waypoints=options,
                         goal_lower=optimistic[level.goal], goal_known=known.get(level.goal),
                         current=cell(view.memory,*view.position), new_window=previous.memory != view.memory,
                         new_since_service=last_service_memory is not None and last_service_memory != view.memory,
                         previous=[previous.fuel_distance, previous.repair_distance]))
        if machine:
            assert machine.choose(view).name==loss["move"]
            slopes = {}
            for m in MOVES:
                q = destination(level.grid,view.position,m)
                if q is not None:
                    slopes[m.name] = [env._lengths[k][q]-env._lengths[k][view.position] if q in env._lengths[k] else None
                                      for k in "FR"]
            rows[-1].update(mode=machine._fallback.mode.name, activity=machine.activity,
                            history=list(history), deltas=slopes)
    (HERE / f"{label}-last-waypoints.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
    for phase in range(3):
        counts = Counter()
        for row in rows:
            loss = row["loss"]
            if int(loss["refuels"] > 0) + int(loss["repairs"] > 0) != phase:
                continue
            types = {s["kind"] for options in row["waypoints"].values() for s in options}
            counts["".join(sorted(types))] += 1
        print(phase, counts, flush=True)


if __name__ == "__main__":
    main()
