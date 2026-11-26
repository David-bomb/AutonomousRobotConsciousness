"""Какие первые новые окна достижимы с сохранением финиша в последней поездке."""
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.grid import cell, locations, wave
from maze.sight_automaton import window_unknown


def evaluate(env,actions,routes):
    start=env.view
    memory=start.memory
    origin=start.position
    goal=env.level.goal
    stops={p for kind in "FR" for p in locations(memory,kind)}
    exits={p for p,_,_ in wave(memory,[origin]) if window_unknown(memory,p)}
    assert origin not in exits
    terminals=frozenset({goal}|stops|exits)
    @lru_cache(None)
    def legs(p):
        return [(q,d) for q,d,_ in wave(memory,[p],stop_at=terminals) if d and q in terminals]

    # Для одной позиции после более раннего прибытия с обоими запасами
    # не меньше можно выполнить любую ту же оставшуюся поездку.
    exits_reached={}
    supplies={}
    serial=count()
    pending=[(0,next(serial),origin,start.fuel,start.health)]
    while pending:
        used,_,p,f,h=heappop(pending)
        inventories=supplies.setdefault(p,[])
        if any(a>=f and b>=h for a,b in inventories):
            continue
        inventories.append((f,h))
        for q,d in legs(p):
            total=used+d
            if total>env.limit-env.steps:
                continue
            after=_arrive(cell(memory,*q),d,f,h)
            if after is None:
                continue
            if q==goal or q in exits:
                arrivals=exits_reached.setdefault(q,[])
                if any(t<=total and a>=after[0] and b>=after[1] for t,a,b in arrivals):
                    continue
                arrivals[:]=[(t,a,b) for t,a,b in arrivals if not (total<=t and after[0]>=a and after[1]>=b)]
                arrivals.append((total,*after))
            else:
                heappush(pending,(total,next(serial),q,*after))
    results=[]
    for p,arrivals in sorted(exits_reached.items()):
        alternatives=[dict(length=t,fuel=f,health=h,viable=routes.allows(p,f,h,env.limit-env.steps-t))
                      for t,f,h in arrivals]
        results.append(dict(point=p,goal=p==goal,can_finish=any(a["viable"] for a in alternatives),
                            arrivals=alternatives))
    assert any(r["can_finish"] for r in results)
    actual=None
    for c in actions:
        env.step(MOVES[int(c)])
        if env.view.memory!=memory or env.position==goal:
            alt=next((r for r in results if r["point"]==env.position),None)
            if alt is None:
                assert min(env.fuel,env.health)==0
            actual=dict(step=env.steps,point=env.position,fuel=env.fuel,health=env.health,
                        can_finish=routes.allows(env.position,env.fuel,env.health,env.limit-env.steps),
                        possible_at_same_exit=bool(alt and alt["can_finish"]))
            break
    return dict(start=dict(point=origin,fuel=start.fuel,health=start.health),actual=actual,exits=results)


def main():
    events={r["map"]:r for l in (HERE/"return-boundary.jsonl").read_bytes().splitlines() if (r:=json.loads(l))}
    histories=read("street-trunk")
    path=HERE/"next-observation.jsonl"
    saved={r["map"] for l in path.read_bytes().splitlines() if (r:=json.loads(l))} if path.exists() else set()
    with path.open("a",encoding="utf8") as output:
        for split,name,_,_,level in cases():
            if name not in events or name in saved:
                continue
            event=events[name]["departure"]
            step=event["step"]
            env=Environment(level)
            actions=histories[name]["actions"]
            for c in actions[:step-1]:
                env.step(MOVES[int(c)])
            assert list(env.position)==event["pos"] and (env.fuel,env.health)==(event["fuel"],event["health"])
            result=evaluate(env,actions[step-1:],PhysicalRoutes(level))
            output.write(json.dumps(dict(map=name,split=split,step=step,**result),ensure_ascii=False)+"\n")
            output.flush()
    print("Выезды",len(events),flush=True)


if __name__=="__main__":
    main()
