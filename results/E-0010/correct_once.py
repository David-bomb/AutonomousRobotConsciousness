"""Что даёт одна внешняя замена в первом необратимом решении выбранного автомата."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib
import json

from bench import AGENTS,HERE,MOVES,cases
from compare import read
from maze.environment import Environment,Status
from maze.interface import Move


def trial(job):
    case,agent,row,loss=job
    split,name,tier,family,level=case
    cls=AGENTS[agent]
    if isinstance(cls,str):
        module,symbol=cls.split(".")
        cls=getattr(importlib.import_module("maze."+module),symbol)
    bot=cls()
    env=Environment(level)
    for c in row["actions"][:loss["step"]-1]:
        chosen=bot.choose(env.view)
        assert chosen==MOVES[int(c)]
        env.step(chosen)
    before=env.view
    assert bot.choose(before).name==loss["move"]
    outcomes=[]
    for direction in loss["viable"]:
        copy_env,copy_bot=deepcopy(env),deepcopy(bot)
        move=Move[direction]
        actions=[str(MOVES.index(move))]
        copy_env.step(move)
        returned=False
        while copy_env.status==Status.RUNNING:
            v=copy_env.view
            move=copy_bot.choose(v)
            if v.position==before.position and v.memory==before.memory and move.name==loss["move"]:
                returned=True
            actions.append(str(MOVES.index(move)))
            copy_env.step(move)
        outcomes.append(dict(direction=direction,status=copy_env.status.name,steps=copy_env.steps,
                             returned_to_choice=returned,suffix="".join(actions)))
    return dict(map=name,split=split,tier=tier,family=family,critical=loss,mode=bot.activity,alternatives=outcomes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent",choices=AGENTS)
    p.add_argument("child",choices=AGENTS)
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    runs=read(args.child)
    pairs=[json.loads(s) for s in (HERE/f"{args.parent}--{args.child}.jsonl").read_bytes().splitlines()]
    assert len(pairs)==len(runs)
    loss={r["map"]:r["child_loss"] for r in pairs if "child_loss" in r}
    jobs=[(case,args.child,runs[case[1]],loss[case[1]]) for case in cases() if case[1] in loss]
    with (HERE/f"one-correction-{args.child}.jsonl").open("x",encoding="utf8") as out, ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(trial,jobs):
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(len(jobs),"неудачных эпизодов с проверкой всех разовых замен")


if __name__=="__main__":
    main()
