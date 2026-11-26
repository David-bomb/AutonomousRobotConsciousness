"""Какие физически допустимые первые станции уже были достижимы по раскрытым дорогам."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read
from maze.environment import Environment
from maze.frontier_automaton import _arrive
from maze.grid import cell
from preparation import visible_legs


def main():
    events={r["map"]:r for b in (HERE/"target-or-direction.jsonl").read_bytes().splitlines() if (r:=json.loads(b))}
    runs=read("street-trunk")
    counts=Counter()
    with (HERE/"known-first-stops.jsonl").open("x",encoding="utf8") as out:
        for split,name,_,_,level in cases():
            if name not in events:
                continue
            env=Environment(level)
            event=events[name]
            for c in runs[name]["actions"][:event["critical"]["step"]-1]:
                env.step(MOVES[int(c)])
            routes=PhysicalRoutes(level)
            legs=visible_legs(env.view.memory,env.position,env.position,level.goal)
            choices=[]
            for p,trail in legs.items():
                k=cell(env.view.memory,*p)
                if k not in "FR":
                    continue
                d=len(trail)
                after=_arrive(k,d,env.fuel,env.health)
                if after:
                    choices.append(dict(kind=k,point=p,length=d,actions=trail,
                                        viable=routes.allows(p,*after,env.limit-env.steps-d)))
            counts["records"]+=1
            counts["has_stop"]+=bool(choices)
            counts["has_viable"]+=any(c["viable"] for c in choices)
            counts["has_viable_selected"]+=any(c["viable"] and c["kind"]==event["selected"] for c in choices)
            out.write(json.dumps(dict(map=name,split=split,critical=event["critical"],selected=event["selected"],
                                      choices=choices))+"\n")
            out.flush()
    print(counts)


if __name__=="__main__":
    main()
