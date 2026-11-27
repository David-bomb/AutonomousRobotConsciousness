"""Наблюдающий учитель на train с отдельным физическим исправлением тупиков и потери запаса."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import random
import time
import numpy as np
from maze.environment import Environment,Status
from maze.frontier_reserve import ReserveFrontier
from ..data import digest,level,rows,save
from ..policy import MOVES
from ..teacher import Teacher
from .bank import HERE


def episode(row):
    rng=random.Random(int(row["geometry"][:12],16))
    env=Environment(level(row))
    partial=ReserveFrontier()
    full=Teacher(env.level)
    actions=[]
    corrections=[]
    stalled=0
    rescue=False
    while env.status==Status.RUNNING and env.steps<520:
        v=env.view
        preferred=MOVES.index(partial.choose(v))
        best,viable=full.labels(env)
        assert best.any()
        rescue=rescue or stalled>=16
        if rescue or not viable[preferred]:
            a=int(rng.choice(np.flatnonzero(best)))
            corrections.append(env.steps)
        else:
            a=preferred
        actions.append(str(a))
        env.step(MOVES[a])
        if env.view.memory!=v.memory:
            stalled=0
            rescue=False
        else:
            stalled+=1
    full.remaining.cache_clear()
    return dict(map=row["name"],geometry=row["geometry"],outcome=env.status.name,steps=env.steps,refuels=env.refuels,
                repairs=env.repairs,actions="".join(actions),observer_steps=corrections)


def main(n):
    assert n%100==0
    path=HERE/f"exploration-{n}.json"
    if path.exists():
        raise FileExistsError(path)
    bank=rows(HERE/"train.jsonl")[:n]
    t=time.monotonic()
    with ProcessPoolExecutor(3) as workers:
        result=[]
        for trial in workers.map(episode,bank,chunksize=1):
            result.append(trial)
            if len(result)%50==0:
                print(len(result),dict(Counter(r["outcome"] for r in result)),round(time.monotonic()-t,1),flush=True)
    save(path,dict(train_sha256=digest(HERE/"train.jsonl"),episodes=result,cap=520,stalled=16,seconds=time.monotonic()-t))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("count",type=int)
    main(p.parse_args().count)
