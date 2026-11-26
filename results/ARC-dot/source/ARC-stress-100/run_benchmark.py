"""Парный прогон фиксированных контроллеров на замороженном наборе."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys


def factories(partial_run=False):
    from maze.pilot_automaton import PilotAutomaton
    from maze.resource_automaton import Automaton as ResourceAutomaton
    from maze.frontier_automaton import FrontierAutomaton
    from maze.frontier_reserve import ReserveFrontier
    from maze.frontier_guarded import (GuardedBase, GuardedReserve, GuardedProbe,
                                       GuardedEvidence, GuardedExcursion, GuardedExcursionOnly)
    available = {"pilot": PilotAutomaton,
                 "pilot+2": partial(PilotAutomaton, service_offset=2),
                 "pilot+3": partial(PilotAutomaton, service_offset=3),
                 "resources": ResourceAutomaton,
                 "resources+2": partial(ResourceAutomaton, service_offset=2),
                 "resources+3": partial(ResourceAutomaton, service_offset=3),
                 "frontier": FrontierAutomaton, "frontier-reserve": ReserveFrontier,
                 "A": GuardedBase, "B": GuardedReserve, "C (frontier-auto)": GuardedProbe,
                 "D": GuardedEvidence, "E": GuardedExcursion, "F": GuardedExcursionOnly}
    if not partial_run:
        from maze.sarsa import Sarsa
        from scripts.audit_automaton import first_revision
        data = Path("results/E-0001/run-02/policy.json").read_text(encoding="utf-8")
        available["SARSA (frozen)"] = partial(Sarsa, json.loads(data)["values"])
        available["resources (E-0002 r1)"] = first_revision()
    return available


def decision_state(controller):
    state = []
    for k, v in sorted(vars(controller).items()):
        if k == "_fallback":
            v = decision_state(v)
        elif isinstance(v, dict):
            v = tuple(v.items())
        state.append((k,v))
    return tuple(state)


def group(variant, tier, maps, partial_run, accelerate=True):
    from maze.environment import Environment, Status
    from maze.levels import Level
    agent = factories(partial_run)[variant]
    episodes = []
    for case in maps:
        level = Level.parse("\n".join([case["name"], f'{case["fuel"]} {case["health"]}', *case["grid"]]))
        environment = Environment(level)
        controller = agent()
        invalid = skipped = 0
        use_cycles = accelerate and variant not in ("SARSA (frozen)", "resources (E-0002 r1)")
        seen = {}
        last_memory = None
        while environment.status == Status.RUNNING:
            if use_cycles:
                if environment.view.memory != last_memory:
                    seen.clear()
                    last_memory = environment.view.memory
                state = (environment.view, decision_state(controller))
                if state in seen:
                    step, fuels, repairs, bad = seen[state]
                    cycle = environment.steps-step
                    count = (environment.limit-environment.steps)//cycle
                    if count:
                        skip = count*cycle
                        environment.steps += skip
                        environment.refuels += count*(environment.refuels-fuels)
                        environment.repairs += count*(environment.repairs-repairs)
                        invalid += count*(invalid-bad)
                        skipped += skip
                        if environment.steps == environment.limit:
                            environment.status = Status.LIMIT
                            break
                    use_cycles = False
                else:
                    seen[state] = (environment.steps, environment.refuels, environment.repairs, invalid)
            environment.step(controller.choose(environment.view))
            invalid += not environment.moved
        episodes.append(dict(map=case["name"], tier=tier, family=case["family"], variant=variant,
                             status=environment.status.name, steps=environment.steps,
                             optimum=case["optimum"], refuels=environment.refuels,
                             repairs=environment.repairs, invalid=invalid, skipped_cycle_steps=skipped))
    return episodes


def run(args):
    args.output.mkdir(parents=True)
    sys.path.insert(0, str(args.repo.resolve()))
    import os
    os.chdir(args.repo)
    source = args.maps.read_bytes()
    maps = [json.loads(line) for line in source.splitlines()]
    if len(maps) != 100 or len({m["geometry"] for m in maps}) != 100:
        raise ValueError("Invalid test set")
    agents = tuple(factories(args.partial_run))
    tiers = tuple(sorted({m["tier"] for m in maps}))
    if any(sum(m["tier"] == tier for m in maps) != 25 for tier in tiers):
        raise ValueError("Unbalanced tiers")
    with (args.output / "episodes.csv").open("x", newline="", encoding="utf-8") as f:
        writer = None
        with ProcessPoolExecutor(args.workers) as executor:
            jobs = {executor.submit(group, name, tier, [m for m in maps if m["tier"] == tier],
                                    args.partial_run, not args.no_acceleration): (name,tier)
                    for name in agents for tier in tiers}
            for job in as_completed(jobs):
                data = job.result()
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=data[0])
                    writer.writeheader()
                writer.writerows(data)
                f.flush()
                print(*jobs[job], dict(Counter(x["status"] for x in data)), flush=True)
    (args.output/"run-info.json").write_text(json.dumps(
        dict(map_sha256=sha256(source).hexdigest(), agents=agents, tiers=tiers,
             python=sys.version.split()[0], reconstructed=args.partial_run),
        ensure_ascii=False,indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("maps", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--partial-run", action="store_true")
    p.add_argument("--no-acceleration", action="store_true")
    a = p.parse_args()
    a.repo, a.maps, a.output = [v.resolve() for v in (a.repo,a.maps,a.output)]
    run(a)


if __name__ == "__main__":
    main()
