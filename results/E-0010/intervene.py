"""Внешняя одноходовая замена в первом расхождении двух физических поездок."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json

from bench import AGENTS, HERE, MOVES, cases
from compare import read
from maze.environment import Environment, Status


def replay_switch(job):
    case, label, original, winner = job
    split, name, tier, family, level = case
    factory = AGENTS[label]
    if isinstance(factory, str):
        module, symbol = factory.split(".")
        factory = getattr(importlib.import_module("maze."+module), symbol)
    agent = factory()
    i = next(i for i,(a,b) in enumerate(zip(original["actions"],winner["actions"])) if a != b)
    env = Environment(level)
    track = []
    while env.status == Status.RUNNING:
        move = agent.choose(env.view)
        if env.steps <= i:
            assert move == MOVES[int(original["actions"][env.steps])]
        if env.steps == i:
            mode = agent.activity
            move = MOVES[int(winner["actions"][i])]
        track.append(str(MOVES.index(move)))
        env.step(move)
    return dict(map=name,split=split,tier=tier,family=family,controller=label,
                step=i+1,mode=mode,original=original["status"],status=env.status.name,
                steps=env.steps,refuels=env.refuels,repairs=env.repairs,actions="".join(track))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent", choices=AGENTS)
    p.add_argument("child", choices=AGENTS)
    p.add_argument("--workers", type=int, default=2)
    args=p.parse_args()
    a,b = read(args.parent),read(args.child)
    jobs=[]
    for case in cases():
        name=case[1]
        if (a[name]["status"]=="SUCCESS") == (b[name]["status"]=="SUCCESS"):
            continue
        if a[name]["status"]=="SUCCESS":
            jobs.append((case,args.child,b[name],a[name]))
        else:
            jobs.append((case,args.parent,a[name],b[name]))
    path=HERE/f"switch-{args.parent}--{args.child}.jsonl"
    with path.open("x",encoding="utf8") as out, ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(replay_switch,jobs):
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(len(jobs), "замен в первом расхождении", flush=True)


if __name__=="__main__":
    main()
