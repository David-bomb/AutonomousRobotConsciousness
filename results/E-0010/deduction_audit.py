"""Проверяет каждую геометрическую дедукцию на стартах и в критических состояниях."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.deduced_roads import deductions
from maze.environment import Environment
from maze.grid import locations


def main():
    runs=read("streets")
    critical={r["map"]:r["critical"]["step"] for b in (HERE/"streets-loss-modes.jsonl").read_bytes().splitlines()
              if (r:=json.loads(b))}
    counts=Counter()
    with (HERE/"deduction-audit.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            env=Environment(level)
            for step in sorted({1,critical.get(name,1)}):
                prior,old=None,(None,None)
                for c in runs[name]["actions"][env.steps:step-1]:
                    prior=env.position
                    old=(env.view.fuel_distance,env.view.repair_distance)
                    env.step(MOVES[int(c)])
                v=env.view
                sites=tuple(tuple(locations(v.memory,k)) for k in "FR")
                new=deductions(v.memory,v.position,level.goal,(v.fuel_distance,v.repair_distance),prior,old,sites)
                roads=walls=0
                for row,actual,pred in zip(v.memory,level.grid,new):
                    for s,a,b in zip(row,actual,pred):
                        if s==b:
                            continue
                        assert s=="?" and (a=="#")== (b=="#"), (name,step,s,a,b)
                        roads+=b=="."
                        walls+=b=="#"
                counts["states"]+=1
                counts["changed"]+=bool(roads or walls)
                counts["roads"]+=roads
                counts["walls"]+=walls
                out.write(json.dumps(dict(map=name,split=split,step=step,roads=roads,walls=walls))+"\n")
    print(counts)


if __name__=="__main__":
    main()
