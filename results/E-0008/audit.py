"""Повтор траекторий и два диагностических контроля из protocol.md."""
import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARC = HERE.parent / "ARC-dot"
E7 = HERE.parent / "E-0007"
PACK = ARC / "source/ARC-stress-100"
STAND = ROOT / "tmp/dot-auto-20261125"
sys.path.insert(0, str(STAND))
sys.path.insert(0, str(PACK))
from maze.environment import Environment, Status
from maze.frontier_reserve import ReserveFrontier
from maze.levels import Level
from scripts.audit_automaton import Reachability
from scripts.evaluate_frontier import Transition, first_irreversible


def run(job):
    case, previous, condition = job
    f, h = case["fuel"], case["health"]
    if condition == "extra":
        f, h = min(100, f + 6), min(100, h + 6)
    level = Level.parse("\n".join([case["name"], f"{f} {h}", *case["grid"]]))
    env, bot = Environment(level), ReserveFrontier()
    history = []
    while env.status == Status.RUNNING:
        view = env.view
        if condition == "complete":
            view = replace(view, memory=level.grid)
        action = bot.choose(view)
        after = env.step(action)
        assert after.moved
        if condition == "trace":
            history.append(Transition(view, action, after, bot.activity))
    out = dict(map=case["name"], tier=case["tier"], family=case["family"], condition=condition,
               baseline=previous["status"], outcome=env.status.name, steps=env.steps,
               refuels=env.refuels, repairs=env.repairs, optimum=case["optimum"])
    if condition == "trace":
        assert out["outcome"] == previous["status"]
        assert all(out[k] == int(previous[k]) for k in ("steps", "refuels", "repairs"))
        oracle = Reachability(level)
        event = first_irreversible(oracle, env.limit, history)
        assert event["viable"] and event["move"] not in event["viable"]
        i = event["step"] - 1
        before, after = history[i].before, history[i].after
        assert oracle.possible(before.position, before.fuel, before.health, env.limit-i)
        assert not oracle.possible(after.position, after.fuel, after.health, env.limit-i-1)
        event["post_distance"] = (after.fuel_distance, after.repair_distance)
        event["with_reset_clock"] = oracle.possible(after.position, after.fuel, after.health, env.limit)
        event["services_before"] = sum(t.after.memory[t.after.position[1]][t.after.position[0]] in "FR"
                                       for t in history[:i])
        out["critical"] = event
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=("trace", "complete", "extra"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    provenance = json.loads((ARC/"replay/provenance.json").read_bytes())
    for file, value in (provenance["base_sources_lf"] | provenance["controllers_lf"]["E6"]).items():
        data = (STAND/file).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == value
    assert hashlib.sha256((PACK/"maps.jsonl").read_bytes()).hexdigest() == provenance["stress_maps"]
    frozen = json.loads((E7/"frozen.json").read_bytes())
    assert hashlib.sha256((E7/"fresh.jsonl").read_bytes().replace(b"\r\n",b"\n")).hexdigest() == frozen["maps"]
    jobs = []
    for maps, runs in [(PACK/"maps.jsonl", E7/"development.csv"), (E7/"fresh.jsonl", E7/"confirmation.csv")]:
        with runs.open(encoding="utf-8", newline="") as stream:
            baseline = {r["map"]: r for r in csv.DictReader(stream) if r["variant"] == "reserve"}
        for line in maps.read_bytes().splitlines():
            case = json.loads(line)
            previous = baseline[case["name"]]
            if args.condition != "trace" or previous["status"] != "SUCCESS":
                jobs.append((case, previous, args.condition))
    assert len(jobs) == (90 if args.condition == "trace" else 200)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream, ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(run, jobs):
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
