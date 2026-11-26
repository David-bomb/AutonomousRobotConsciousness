"""Внешняя проверка выполнимости намерения ехать всё время к ближайшему сервису."""
from collections import Counter,deque
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.grid import destination
from maze.pilot_automaton import Mode
from maze.resource_automaton import service_order


def downhill(env,routes,kind):
    field=env._lengths[kind]
    queue=deque([(env.position,env.fuel,env.health,0)])
    seen={}
    while queue:
        p,f,h,used=queue.popleft()
        if p==routes.goal:
            return True
        if field[p]==0:
            if routes.allows(p,f,h,env.limit-used):
                return True
            continue
        stocks=seen.setdefault(p,[])
        if any(a>=f and b>=h for a,b in stocks):
            continue
        stocks.append((f,h))
        for m in MOVES:
            q=destination(routes.grid,p,m)
            if q is None or field[q]!=field[p]-1:
                continue
            after=_arrive(routes.grid[q[1]][q[0]],1,f,h)
            if after:
                queue.append((q,*after,used+1))
    return False


def main():
    runs=read("range-roads")
    counts=Counter()
    with (HERE/"initial-target.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            env=Environment(level)
            routes=PhysicalRoutes(level)
            target=service_order(env.view,Mode.GOAL,2)
            mode=("F" if target.mode==Mode.FUEL else "R") if target else None
            possible={k:downhill(env,routes,k) for k in "FR" if env.position in env._lengths[k]}
            row=dict(map=name,split=split,tier=tier,family=family,target=mode,
                     downhill=possible,status=runs[name]["status"])
            if split!="val":
                counts[(row["status"]=="SUCCESS",possible.get(mode),any(possible.values()))]+=1
            out.write(json.dumps(row)+"\n")
    print(counts)


if __name__=="__main__":
    main()
