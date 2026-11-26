"""Сопоставить выбор окна с самостоятельным продолжением после открытия клеток."""
from collections import Counter
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases
from compare import read
from maze.deduced_roads import deductions
from maze.environment import Environment
from maze.exit_flow import CorridorFlow, ExitCurrent, ExitFlow
from maze.first_window import WallWindow, window_arrivals
from maze.frontier_reserve import _lengths
from maze.grid import cell, destination, locations
from maze.sampled_margin import SampledMargin
from maze.simple_exit_paths import SimpleExitPaths


def next_landmark(grid, point, commands):
    for c in commands:
        if cell(grid, *point) in "FRB":
            break
        point = destination(grid, point, MOVES[int(c)]) or point
    kind = cell(grid, *point)
    assert kind in "FRB"
    return kind


def records(name):
    return {r["map"]: r for line in (HERE / name).read_bytes().splitlines() if (r := json.loads(line))}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--new-rules", action="store_true", help="проверить выбор проезжего выхода и удалённой станции")
    args = parser.parse_args()
    models = [("wall-window", WallWindow()), ("exit-flow", ExitFlow()), ("corridor-flow", CorridorFlow()),
              ("exit-current", ExitCurrent()), ("sampled-margin", SampledMargin()), ("simple-exit-paths", SimpleExitPaths())]
    if args.new_rules:
        from maze.gateway_paths import GatewayPaths
        from maze.service_exit_paths import ServiceExitPaths
        from maze.feasible_windows import FeasibleWindows
        models.extend([("gateway-paths", GatewayPaths()), ("service-exit-paths", ServiceExitPaths()),
                       ("feasible-windows", FeasibleWindows())])
    predictions = records("window-ranking.jsonl")
    replay = records("window-replay.jsonl")
    events = records("next-observation.jsonl")
    runs = read("street-trunk")
    assert predictions.keys() == events.keys() == replay.keys() and len(replay) == 24
    rows = []
    for split, name, _, _, level in cases():
        if name not in replay:
            continue
        env = Environment(level)
        for c in runs[name]["actions"][:events[name]["step"] - 1]:
            env.step(MOVES[int(c)])
        options = list(window_arrivals(env.view))
        winners = {(tuple(o["point"]), len(o["forced"]), o["fuel"], o["health"]):
                   next_landmark(level.grid, tuple(o["point"]), o["suffix"])
                   for o in replay[name]["outcomes"] if o["status"] == "SUCCESS"}
        choices = {rule: (tuple(r["choice"][0]), *r["choice"][1:]) for rule, r in predictions[name]["predictions"].items()}
        goal = env.position[0] + env.view.dx, env.position[1] + env.view.dy
        view = env.view
        inferred = deductions(view.memory, view.position, goal, (view.fuel_distance, view.repair_distance),
                              None, (None, None), tuple(tuple(locations(view.memory, k)) for k in "FR"))
        added = [(x, y, c) for y, row in enumerate(inferred) for x, c in enumerate(row) if c != view.memory[y][x]]
        assert all(cell(level.grid, x, y) == c or ((x, y) == goal and c == ".") for x, y, c in added)
        for rule, bot in models:
            a = bot.pick(env.view, goal, options)
            choices[rule] = a.point, a.length, a.fuel, a.health
        # Лишь внешний контроль получает истинную длину до финиша. Проверяем,
        # хватило бы одного точного ранжирования окон с прежним продолжением.
        goal_field = _lengths(level.grid, goal)
        direct = [a for a in options if goal_field[a.point] <= min(a.fuel, a.health)]
        reference = {}
        for key, order in (("shortest", lambda a: (goal_field[a.point], a.length)),
                           ("spare", lambda a: (goal_field[a.point] - min(a.fuel, a.health), a.length))):
            if direct:
                a = min(direct, key=lambda a: (*order(a), a.point, -a.fuel, -a.health))
                choice = a.point, a.length, a.fuel, a.health
                reference[key] = dict(choice=choice, direct_length=goal_field[a.point], finished=choice in winners)
        rows.append(dict(map=name, split=split, choices=choices, winners=sorted(winners), oracle=reference,
                         deduced_cells=added,
                         first_landmarks=[winners[k] for k in sorted(winners)],
                         finishes={k: choice in winners for k, choice in choices.items()}))
    output = HERE / ("window-continuations-resumed.json" if args.new_rules else "window-continuations.json")
    with output.open("w", encoding="utf8") as stream:
        json.dump(rows, stream, indent=2)
    print(Counter(rule for r in rows for rule, success in r["finishes"].items() if success), flush=True)
    print("Внешняя точная длина до финиша", Counter(rule for r in rows for rule, trial in r["oracle"].items()
                                                   if trial["finished"]), flush=True)


if __name__ == "__main__":
    main()
