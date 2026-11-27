"""Демонстрации только на независимом train. Сохраняются физические команды и исход."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

from maze.environment import Environment,Status
from maze.frontier_reserve import ReserveFrontier
from maze.interface import Move
from ..data import OUT, ROOT, digest, level, rows, save

HERE=Path(__file__).resolve().parent
MOVES=tuple(Move)


def play(row):
    env=Environment(level(row))
    teacher=ReserveFrontier()
    actions=[]
    while env.status==Status.RUNNING and env.steps<260:
        action=teacher.choose(env.view)
        env.step(action)
        actions.append(str(MOVES.index(action)))
    return dict(map=row["name"],geometry=row["geometry"],outcome=env.status.name,steps=env.steps,
                refuels=env.refuels,repairs=env.repairs,actions="".join(actions))


def main(count):
    p=HERE/f"partial-{count}.json"
    if p.exists():
        raise FileExistsError(p)
    bank=rows(OUT/"data/train.jsonl")
    # Равные представительства H1–H4 даже в первой малой проверке.
    selected=[r for t in ("H1","H2","H3","H4") for r in [r for r in bank if r["tier"]==t][:count//4]]
    assert len(selected)==count
    t=time.monotonic()
    with ProcessPoolExecutor(3) as executor:
        result=[]
        for row in executor.map(play,selected,chunksize=1):
            result.append(row)
            if len(result)%25==0:
                print(len(result),dict(Counter(r["outcome"] for r in result)),round(time.monotonic()-t,1),flush=True)
    save(p,dict(train_sha256=digest(OUT/"data/train.jsonl"),teacher_sha256=digest(ROOT/"maze/frontier_reserve.py"),
                cap=260,seconds=time.monotonic()-t,episodes=result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("count",type=int)
    main(parser.parse_args().count)
