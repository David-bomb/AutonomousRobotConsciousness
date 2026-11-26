"""На фиксированных ранних наблюдениях проверить совместимость предполагаемых дорог независимым обходом."""
from collections import Counter
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import read
from maze.compatible_loops import compatible_masks, feasible_moves
from maze.axis_loops import axis_samples
from maze.room_lengths import room_samples
from maze.exchanged_roads import exchanged_samples
from maze.environment import Environment
from maze.grid import locations, wave
from maze.interface import Move


def check_readings(mask, view, prior, old):
    w, h = len(view.memory[0]), len(view.memory)
    grid = tuple("".join("." if mask & (1 << (y*w+x)) else "#" for x in range(w)) for y in range(h))
    for point, gauges in ((view.position, (view.fuel_distance, view.repair_distance)), (prior, old)):
        if point is None:
            continue
        lengths = {p: d for p, d, _ in wave(grid, [point])}
        for kind, measured in zip("FR", gauges):
            if measured is not None:
                assert min((lengths.get(p, float("inf")) for p in locations(view.memory, kind))) == measured


def main():
    parser = ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--axis", action="store_const", const="axis", dest="model")
    selection.add_argument("--rooms", action="store_const", const="room", dest="model")
    selection.add_argument("--exchange", action="store_const", const="exchange", dest="model")
    label = parser.parse_args().model or "loop"
    controls = {r["map"]: r for r in map(json.loads, (HERE / "service-ambiguity-controls.jsonl").read_bytes().splitlines())}
    observations = {r["map"]: r["event"] for r in map(json.loads, (HERE / "first-ambiguity.jsonl").read_bytes().splitlines())}
    runs, rows = read("street-trunk"), []
    for split, name, _, _, level in cases():
        if name not in controls:
            continue
        clear_episode_caches()
        env = Environment(level)
        previous = None
        for c in runs[name]["actions"][:controls[name]["step"] - 1]:
            previous = env.view
            env.step(MOVES[int(c)])
        v = env.view
        readings = v.fuel_distance, v.repair_distance
        old = (previous.fuel_distance, previous.repair_distance) if v.moved else (None, None)
        prior = previous.position if v.moved else None
        goal = v.position[0] + v.dx, v.position[1] + v.dy
        options = [Move[m] for m in observations[name]["possible"]]
        counts = Counter()
        args = v.memory, goal, v.position, readings, prior, old
        sampler = {"room": room_samples, "axis": axis_samples}.get(label)
        if label == "exchange":
            trials = exchanged_samples(*args)
            worlds = sorted({mask for trial in trials for mask in trial})
        else:
            worlds = [mask for mask, _ in sampler(*args)] if sampler else compatible_masks(*args)
        for mask in worlds:
            check_readings(mask, v, prior, old)
            counts[tuple(sorted(m.name for m in feasible_moves(v, goal, mask, options)))] += 1
        rows.append(dict(map=name, split=split, samples=len(worlds), supports=[[k, n] for k, n in counts.items()],
                         safe=controls[name]["safe"], parent=controls[name]["choices"]["mission-order-reserve"]))
        if label == "exchange":
            rows[-1]["independent_draws"] = len(trials)
    path = HERE / ("room-loop-evidence.jsonl" if label == "room" else f"{label}-evidence.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    print(len(rows), "наблюдений,", sum(r["samples"] for r in rows), "проверенных сетей,",
          sum(r["samples"] >= 2 for r in rows), "наблюдений с двумя или более сетями",flush=True)


if __name__ == "__main__":
    main()
