"""Сравнить парные прохождения E-0003 с прибавкой 0, 2 и 3."""
import argparse
import json
from pathlib import Path

from maze.automaton import Automaton, ResourceAutomaton
from maze.datasets import development_sets, geometry
from maze.environment import Environment, Status
from scripts.audit_automaton import independent_cases


def episode(case, controller, offset, keep_trace):
    env = Environment(case.level)
    robot = controller(service_offset=offset)
    trace = []
    while env.status == Status.RUNNING:
        view = env.view
        move = robot.choose(view)
        if keep_trace:
            trace.append(dict(step=env.steps + 1, position=view.position, mode=robot.mode.name,
                              move=move.name, fuel=view.fuel, health=view.health,
                              fuel_distance=view.fuel_distance, repair_distance=view.repair_distance))
        env.step(move)
    result = dict(outcome=env.status.name, steps=env.steps, refuels=env.refuels, repairs=env.repairs)
    if keep_trace:
        result["trace"] = trace
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train, val = development_sets()
    excluded = [geometry(c.level.grid) for c in independent_cases(train, val)]
    control = independent_cases(train, val, 25112027, excluded)
    splits = [("val", [c for c in val if c.stage == 3]), ("confirm", control)]
    rows = []
    for split, cases in splits:
        for case in cases:
            name = case.name.replace("fresh-", "confirm-")
            row = dict(name=name, split=split, geometry=geometry(case.level.grid))
            for label, controller in (("pilot", Automaton), ("resources", ResourceAutomaton)):
                row[label] = {offset: episode(case, controller, offset, name == "val-3-10")
                              for offset in (0, 2, 3)}
            rows.append(row)
        print(f"{split}: {len(cases)} карт, шесть прохождений каждой", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
