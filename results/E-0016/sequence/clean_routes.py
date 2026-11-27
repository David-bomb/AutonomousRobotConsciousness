"""Удаление только физически избыточных учебных кругов с одинаковой картой памяти."""
from collections import defaultdict
import argparse
from maze.environment import Environment,Status
from ..data import read,rows,level,save,digest
from ..policy import MOVES
from .bank import HERE


def clean(row,actions):
    env=Environment(level(row))
    observations=[]
    groups=defaultdict(list)
    for t,c in enumerate(actions):
        v=env.view
        key=v.position,v.memory
        groups[key].append(t)
        observations.append((key,v.fuel,v.health))
        env.step(MOVES[int(c)])
    outcome=env.status
    env.reset()
    selected=[]
    i=0
    while i<len(actions):
        key,f,h=observations[i]
        assert (env.position,env.view.memory)==key and env.fuel>=f and env.health>=h
        j=next(j for j in reversed(groups[key]) if j>=i and observations[j][1]<=f and observations[j][2]<=h)
        selected.append(actions[j])
        env.step(MOVES[int(actions[j])])
        i=j+1
    assert env.status==outcome or (outcome!=Status.SUCCESS and env.status==Status.RUNNING)
    return "".join(selected)


def main(n):
    bank={r["name"]:r for r in rows(HERE/"train.jsonl")}
    path=HERE/f"exploration-{n}.json"
    episodes=[]
    for r in read(path)["episodes"]:
        actions=clean(bank[r["map"]],r["actions"])
        episodes.append(dict(map=r["map"],actions=actions,outcome=r["outcome"],source_steps=len(r["actions"])))
    stats=dict(before=sum(r["source_steps"] for r in episodes),after=sum(len(r["actions"]) for r in episodes),
               shortened=sum(len(r["actions"])<r["source_steps"] for r in episodes))
    save(HERE/f"clean-{n}.json",dict(source_sha256=digest(path),episodes=episodes,summary=stats))
    print(stats)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("count",type=int)
    main(p.parse_args().count)
