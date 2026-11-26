"""Поздние отказы: провести до физически пригодной станции и вернуть управление исходному автомату."""
from collections import Counter
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import read
from first_stop_replay import shortest_legs
from maze.environment import Environment, Status


def main():
    runs = read("simple-exit-paths")
    points = {r["map"]: r for r in map(json.loads, (HERE / "simple-last-waypoints.jsonl").read_bytes().splitlines())
              if r["loss"]["refuels"] and r["loss"]["repairs"]}
    rows = []
    for split, name, _, _, level in cases():
        if name not in points:
            continue
        r = points[name]
        viable = {tuple(s["point"]) for stops in r["waypoints"].values() for s in stops if s["kind"] != "B"}
        if not viable:
            continue
        clear_episode_caches()
        env, bot = Environment(level), controller("simple-exit-paths")
        for c in runs[name]["actions"][:r["loss"]["step"] - 1]:
            assert bot.choose(env.view) == MOVES[int(c)]
            env.step(MOVES[int(c)])
        paths = shortest_legs(level, env.position, True)
        trials = []
        for site in sorted(viable):
            commands = paths[site]
            sim, agent = deepcopy(env), deepcopy(bot)
            for c in commands:
                agent.choose(sim.view)
                sim.step(MOVES[int(c)])
            assert sim.position == site and sim.status == Status.RUNNING
            reopened = sim.view.memory != env.view.memory
            suffix = []
            while sim.status == Status.RUNNING:
                move = agent.choose(sim.view)
                suffix.append(str(MOVES.index(move)))
                sim.step(move)
            trials.append(dict(site=site, forced=commands, new_memory=reopened, status=sim.status.name,
                               steps=sim.steps, suffix="".join(suffix)))
        rows.append(dict(map=name, split=split, critical=r["loss"]["step"], trials=trials))
    assert len(rows) == 15
    (HERE / "simple-return-replay.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    print("Карты", len(rows), "финиши", sum(any(t["status"]=="SUCCESS" for t in r["trials"]) for r in rows),
          "поездки", Counter((t["new_memory"], t["status"]) for r in rows for t in r["trials"]),flush=True)


if __name__ == "__main__":
    main()
