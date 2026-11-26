"""Парное испытание правил на отдельных физических эпизодах."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARC = HERE.parent / "ARC-dot"
PACK = ARC / "source/ARC-stress-100"
BASE = "reserve"
VARIANTS = [BASE] + ["".join(group) for n in range(1,5) for group in combinations("GPST",n)]
SEED = 26041719


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def setup(stand):
    sys.path.insert(0,str(stand))
    sys.path.insert(0,str(PACK))


def run_case(item, variants):
    from maze.environment import Environment, Status
    from maze.frontier_reserve import ReserveFrontier
    from maze.levels import Level
    from controller import ExperimentalReserve
    from run_benchmark import decision_state
    from run_lockstep import Trial, fork_environment
    level = Level.parse("\n".join([item["name"],f'{item["fuel"]} {item["health"]}',*item["grid"]]))
    first = Environment(level)
    trials = {name:Trial(fork_environment(first), ReserveFrontier() if name == BASE else ExperimentalReserve(name))
              for name in variants}
    rows = []
    while trials:
        for name,t in tuple(trials.items()):
            env = t.env
            if t.accelerate:
                if env.view.memory != t.memory:
                    t.seen.clear()
                    t.memory = env.view.memory
                key = env.view, decision_state(t.bot)
                if key in t.seen:
                    start,f,r,bad = t.seen[key]
                    period = env.steps-start
                    cycles = (env.limit-env.steps)//period
                    t.skipped = cycles*period
                    env.steps += t.skipped
                    env.refuels += cycles*(env.refuels-f)
                    env.repairs += cycles*(env.repairs-r)
                    t.invalid += cycles*(t.invalid-bad)
                    t.accelerate = False
                    if env.steps == env.limit:
                        env.status = Status.LIMIT
                else:
                    t.seen[key] = env.steps,env.refuels,env.repairs,t.invalid
            if env.status == Status.RUNNING:
                env.step(t.bot.choose(env.view))
                t.invalid += not env.moved
            if env.status != Status.RUNNING:
                rows.append(dict(map=item["name"],tier=item["tier"],family=item["family"],variant=name,
                                 status=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                                 invalid=t.invalid,skipped=t.skipped))
                del trials[name]
    return rows


def maps_from(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def generate(path):
    import make_hard_maps
    used = set(json.loads((PACK/"excluded.json").read_text(encoding="utf-8")))
    used.update(r["geometry"] for r in maps_from(PACK/"maps.jsonl"))
    used.update(json.loads((ARC/"replay/e6-fresh/diagnosis.json").read_text(encoding="utf-8"))["geometries"].values())
    make_hard_maps.SEED = SEED
    rows = list(make_hard_maps.generate(used))
    assert len({r["geometry"] for r in rows} - used) == len(rows) == 100
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        for row in rows:
            row["name"] = "fresh-"+row["name"]
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")


def validation_maps():
    from maze.datasets import load_split
    rows = []
    for case in load_split("val"):
        level = case.level
        grid = list(level.grid)
        x,y = level.start
        grid[y] = grid[y][:x]+"A"+grid[y][x+1:]
        rows.append(dict(name=case.name,tier=str(case.stage),family="val",grid=grid,
                         fuel=level.fuel,health=level.health))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=["development","generate","fresh","val"])
    parser.add_argument("output",type=Path)
    parser.add_argument("--stand",type=Path,default=HERE.parents[1]/"tmp/dot-auto-20261125")
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--frozen",type=Path,default=HERE/"frozen.json")
    parser.add_argument("--maps",type=Path,default=HERE/"fresh.jsonl")
    args = parser.parse_args()
    stand = args.stand.resolve()
    setup(stand)
    if args.split == "generate":
        generate(args.output)
        return
    provenance = json.loads((ARC/"replay/provenance.json").read_text(encoding="utf-8"))
    for name,h in (provenance["base_sources_lf"] | provenance["controllers_lf"]["E6"]).items():
        assert digest(stand/name) == h
    assert hashlib.sha256((PACK/"maps.jsonl").read_bytes()).hexdigest() == provenance["stress_maps"]
    if args.split == "development":
        cases, variants = maps_from(PACK/"maps.jsonl"), VARIANTS
    else:
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
        assert frozen["controller"] == digest(HERE/"controller.py") and frozen["runner"] == digest(Path(__file__))
        variants = [BASE,frozen["variant"]]
        cases = validation_maps() if args.split == "val" else maps_from(args.maps)
        if args.split == "fresh":
            assert digest(args.maps) == frozen["maps"]
    with args.output.open("x",encoding="utf-8",newline="") as stream:
        writer = None
        with ProcessPoolExecutor(args.workers,initializer=setup,initargs=(stand,)) as pool:
            tasks = [pool.submit(run_case,item,variants) for item in cases]
            for i,job in enumerate(as_completed(tasks),1):
                rows = job.result()
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                if i%10 == 0 or i == len(cases):
                    print(f"{args.split} {i}/{len(cases)}",flush=True)


if __name__ == "__main__":
    main()
