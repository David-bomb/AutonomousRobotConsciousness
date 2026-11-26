"""Какие первые необратимые ходы уже исключаются видимой роботу геометрией."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.frontier_reserve import _optimistic,_relaxed_route,_service_floors
from maze.grid import cell,destination,locations
from maze.pilot_automaton import Mode
from maze.resource_automaton import Service,possible_directions


def main():
    runs=read("streets")
    events={r["map"]:r for b in (HERE/"streets-loss-modes.jsonl").read_bytes().splitlines()
            if (r:=json.loads(b))}
    counts=Counter()
    with (HERE/"relaxed-loss.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in events:
                continue
            event=events[name]["critical"]
            env=Environment(level)
            old={}
            for c in runs[name]["actions"][:event["step"]-1]:
                old={Mode.FUEL:env.view.fuel_distance,Mode.REPAIR:env.view.repair_distance}
                env.step(MOVES[int(c)])
            v=env.view
            goal=v.position[0]+v.dx,v.position[1]+v.dy
            projected=_optimistic(v.memory)
            stops=tuple((p,k) for k in "FR" for p in locations(v.memory,k))+((goal,"B"),)
            descending={}
            for mode,k,stock,d in ((Mode.FUEL,"F",v.fuel,v.fuel_distance),(Mode.REPAIR,"R",v.health,v.repair_distance)):
                if d:
                    service=Service(mode,stock,d,tuple(p for p,kind in stops if kind==k))
                    descending[k]=d,possible_directions(v,service,old.get(mode))
            options=[[],[]]
            for m in MOVES:
                q=destination(v.memory,v.position,m)
                if q is None:
                    continue
                after=_arrive("B" if q==goal else cell(v.memory,*q),1,v.fuel,v.health)
                if after is None:
                    continue
                floors=_service_floors(v,old,q)
                for extended in (False,True):
                    if extended:
                        for kind,(d,moves) in descending.items():
                            if m not in moves:
                                floors[kind]=max(floors.get(kind,0),d)
                    if _relaxed_route(projected,q,goal,*after,stops,floors):
                        options[int(extended)].append(m.name)
            assert all(set(event["viable"])<=set(opts) for opts in options)
            row=dict(map=name,split=split,mode=events[name]["mode"],critical=event,relaxed=options[0],extended=options[1])
            counts[row["mode"],event["move"] in options[0],event["move"] in options[1]]+=1
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(counts)


if __name__=="__main__":
    main()
