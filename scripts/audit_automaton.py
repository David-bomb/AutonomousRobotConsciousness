"""Внешняя диагностика E-0002. Контроллер получает только Observation."""
import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
import random
import sys

from maze.datasets import Case, DATA, development_sets, geometry
from maze.environment import Environment, Status
from maze.grid import destination, locations, wave
from maze.interface import Move
from maze.levels import Level
from maze.pilot_automaton import PilotAutomaton
from maze.runner import run
from scripts.make_sarsa_maps import make_case


class Reachability:
    """Точный поиск успешного продолжения с полной картой только для оценки."""
    def __init__(self, level):
        self.goal = level.goal
        self.grid = level.grid
        self.distance = {p: d for p, d, _ in wave(level.grid, [level.goal])}
        self.services = [{p: d for p, d, _ in wave(level.grid, locations(level.grid, kind))}
                         for kind in "FR"]
        self.edges = {p: [(move, q) for move in Move if (q := destination(level.grid, p, move))]
                      for p in self.distance}

    def after(self, point, f, h):
        tile = self.grid[point[1]][point[0]]
        return (100 if tile == "F" else f-1, 100 if tile == "R" else h-1)

    def possible(self, p, fuel, health, remaining):
        queue = deque([(p, fuel, health, remaining)])
        seen = {}
        while queue:
            p, f, h, time = queue.popleft()
            if p == self.goal:
                return True
            d = self.distance.get(p, float("inf"))
            if min(f, h) <= 0 or d > time:
                continue
            if d <= min(f, h):
                return True
            if any(min(d, paths.get(p, float("inf"))) > stock
                   for paths, stock in zip(self.services, (f, h))):
                continue
            stocks = seen.setdefault(p, [])
            # BFS извлекает более ранние прибытия первыми.
            if any(of >= f and oh >= h for of, oh in stocks):
                continue
            stocks.append((f, h))
            for _, q in self.edges[p]:
                nf, nh = self.after(q, f, h)
                queue.append((q, nf, nh, time-1))
        return False


def diagnose(case, factory=PilotAutomaton):
    env = Environment(case.level)
    bot = factory()
    oracle = Reachability(case.level)
    critical = None
    decisions = []
    while env.status == Status.RUNNING:
        before = env.view
        move = bot.choose(before)
        after = env.step(move)
        decisions.append(dict(step=env.steps, pos=before.position, mode=bot.mode.name,
                              move=move.name, fuel=before.fuel, health=before.health,
                              dF=before.fuel_distance, dR=before.repair_distance))
        if critical is None and not oracle.possible(env.position, env.fuel, env.health, env.limit-env.steps):
            alternatives = []
            for m, q in oracle.edges[before.position]:
                f, h = oracle.after(q, before.fuel, before.health)
                if oracle.possible(q, f, h, env.limit-env.steps):
                    alternatives.append(m.name)
            if not alternatives:
                raise ValueError("До выбранного хода не найден успешный маршрут")
            critical = dict(**decisions[-1], after=after.position, after_dF=after.fuel_distance,
                            after_dR=after.repair_distance, alternatives=alternatives)
    return dict(name=case.name, outcome=env.status.name, steps=env.steps, critical=critical, trace=decisions)


def independent_cases(train, val, seed=25112026, exclude=()):
    manifest = json.loads((DATA / "test-manifest.json").read_text(encoding="utf-8"))
    occupied = {v["geometry"] for v in manifest.values()} | {geometry(c.level.grid) for c in train+val}
    occupied.update(exclude)
    rng = random.Random(seed)
    cases = []
    while len(cases) < 100:
        i = len(cases)
        level = Level.parse("\n".join(make_case(3, i, rng)))
        key = geometry(level.grid)
        if key not in occupied:
            occupied.add(key)
            cases.append(Case(f"fresh-3-{i+1:03}", 3, level))
    return cases


def measure(case, factory):
    env = Environment(case.level)
    result = run(env, factory())
    return dict(outcome=result.name, steps=env.steps)


def first_revision():
    snapshot = Path(__file__).resolve().parents[1] / "results/E-0002/revision-1/automaton.py"
    spec = importlib.util.spec_from_file_location("maze._audit_first_revision", snapshot)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Automaton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "train", "evaluate", "confirm"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train, val = development_sets()
    hard = [c for c in val if c.stage == 3]
    if args.command == "diagnose":
        data = [row for c in hard if (row := diagnose(c))["critical"] is not None]
    else:
        from maze.automaton import ResourceAutomaton as Automaton
        if args.command == "confirm":
            excluded = [geometry(c.level.grid) for c in independent_cases(train, val)]
            cases = hard + independent_cases(train, val, 25112027, excluded)
            revision = first_revision()
            data = [dict(name=c.name if c in hard else c.name.replace("fresh-", "confirm-"),
                         geometry=geometry(c.level.grid), pilot=measure(c, PilotAutomaton),
                         first=measure(c, revision), current=measure(c, Automaton)) for c in cases]
        else:
            cases = train if args.command == "train" else hard + independent_cases(train, val)
            factory = Automaton if args.command == "train" else first_revision()
            data = [dict(name=c.name, geometry=geometry(c.level.grid),
                         pilot=measure(c, PilotAutomaton), current=measure(c, factory)) for c in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
