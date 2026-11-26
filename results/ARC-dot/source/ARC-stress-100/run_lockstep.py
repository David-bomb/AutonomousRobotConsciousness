"""Lockstep version of the same benchmark, sharing pure-function caches per map."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from run_benchmark import decision_state, factories


@dataclass
class Trial:
    env: object
    bot: object
    seen: dict = field(default_factory=dict)
    memory: object = None
    invalid: int = 0
    skipped: int = 0
    accelerate: bool = True


def one_map(case, names, partial):
    from maze.environment import Environment, Status
    from maze.levels import Level
    level = Level.parse("\n".join([case["name"], f'{case["fuel"]} {case["health"]}', *case["grid"]]))
    agents = factories(partial)
    first = Environment(level)
    trials = {name: Trial(first if index == 0 else fork_environment(first), agents[name](),
                          accelerate=name not in ("SARSA (frozen)", "resources (E-0002 r1)"))
              for index,name in enumerate(names)}
    done = []
    while trials:
        for name,t in tuple(trials.items()):
            e = t.env
            if t.accelerate:
                if t.memory != e.view.memory:
                    t.memory = e.view.memory
                    t.seen.clear()
                key = (e.view,decision_state(t.bot))
                old = t.seen.get(key)
                if old is None:
                    t.seen[key] = (e.steps,e.refuels,e.repairs,t.invalid)
                else:
                    st,fu,re,iv=old
                    cycle=e.steps-st
                    count=(e.limit-e.steps)//cycle
                    skipped=count*cycle
                    e.steps += skipped
                    t.skipped += skipped
                    e.refuels += count*(e.refuels-fu)
                    e.repairs += count*(e.repairs-re)
                    t.invalid += count*(t.invalid-iv)
                    t.accelerate=False
                    if e.steps == e.limit:
                        e.status=Status.LIMIT
            if e.status == Status.RUNNING:
                e.step(t.bot.choose(e.view))
                t.invalid += not e.moved
            if e.status != Status.RUNNING:
                done.append(dict(map=case["name"], tier=case["tier"], family=case["family"],
                                 variant=name,status=e.status.name,steps=e.steps,
                                 optimum=case["optimum"],refuels=e.refuels,repairs=e.repairs,
                                 invalid=t.invalid,skipped_cycle_steps=t.skipped))
                del trials[name]
    return done


def fork_environment(first):
    from copy import copy
    new=copy(first)
    # These sensor-distance tables and the service list are read-only; each
    # controller still gets a fresh position, health, and episode memory.
    new.reset()
    return new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("maps",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--partial-run",action="store_true")
    p.add_argument("--reuse",type=Path,help="Existing per-map results on these exact maps and variants")
    a=p.parse_args()
    a.repo,a.maps,a.output=[x.resolve() for x in (a.repo,a.maps,a.output)]
    if a.reuse is not None:a.reuse=a.reuse.resolve()
    source=a.maps.read_bytes()
    maps=[json.loads(r) for r in source.splitlines()]
    assert len(maps)==100 and len({c["geometry"] for c in maps})==100
    sys.path.insert(0,str(a.repo))
    os.chdir(a.repo)
    names=tuple(factories(a.partial_run))
    prior = list(csv.DictReader(a.reuse.open(encoding="utf-8"))) if a.reuse else []
    filtered=[r for r in prior if r.get("variant") in names]
    has={(r["map"],r["variant"]) for r in filtered}
    a.output.mkdir(parents=True)
    with (a.output/"episodes.csv").open("x",encoding="utf-8",newline="") as f:
        w=None
        if filtered:
            w=csv.DictWriter(f,fieldnames=list(filtered[0]))
            w.writeheader()
            w.writerows(filtered)
            f.flush()
        with ProcessPoolExecutor(a.workers) as pool:
            tasks=[pool.submit(one_map,c,tuple(n for n in names if (c["name"],n) not in has),a.partial_run)
                   for c in maps if any((c["name"],n) not in has for n in names)]
            for job in as_completed(tasks):
                rows=job.result()
                if w is None:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]))
                    w.writeheader()
                w.writerows(rows)
                f.flush()
                print(rows[0]["map"],len(rows),flush=True)
    (a.output/"run-info.json").write_text(json.dumps(
        dict(map_sha256=sha256(source).hexdigest(),agents=names,
             tiers=sorted({r["tier"] for r in maps}),
             python=sys.version.split()[0],reconstructed=a.partial_run),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
