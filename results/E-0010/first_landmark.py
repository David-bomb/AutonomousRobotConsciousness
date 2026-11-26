"""Какая остановка может быть первой между решением и выполнимым финишем."""
from collections import Counter,deque
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.grid import destination


def landmarks(env,routes):
    p=env.position
    seen={p}
    queue=deque([(p,0)])
    possible=[]
    while queue:
        point,used=queue.popleft()
        d=used+1
        if d>min(env.fuel,env.health,env.limit-env.steps):
            continue
        for move in MOVES:
            q=destination(routes.grid,point,move)
            if q is None or q in seen:
                continue
            seen.add(q)
            kind=routes.grid[q[1]][q[0]]
            if kind not in "FRB":
                queue.append((q,d))
                continue
            after=_arrive(kind,d,env.fuel,env.health)
            if after and routes.allows(q,*after,env.limit-env.steps-d):
                possible.append(dict(kind=kind,point=q,length=d,nearest=(
                    d==env._lengths[kind][p] if kind in "FR" else None)))
    return possible


def main():
    runs=read("streets")
    failures={r["map"]:r for b in (HERE/"streets-loss-modes.jsonl").read_bytes().splitlines()
              if (r:=json.loads(b))}
    counts=Counter()
    with (HERE/"first-landmarks.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in failures:
                continue
            event=failures[name]["critical"]
            env=Environment(level)
            for c in runs[name]["actions"][:event["step"]-1]:
                env.step(MOVES[int(c)])
            viable=landmarks(env,PhysicalRoutes(level))
            assert viable, name
            summary="или".join(sorted(set(k["kind"] for k in viable)))
            counts[summary]+=1
            row=dict(map=name,split=split,family=family,mode=failures[name]["mode"],critical=event,stops=viable)
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(counts)


if __name__=="__main__":
    main()
