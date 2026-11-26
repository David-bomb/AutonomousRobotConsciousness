"""Проверить допустимость фактических и условных следующих показаний на реальных дорогах."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.grid import destination,locations
from maze.implied_turn import support


def main():
    runs=read("street-trunk")
    critical={r["map"]:r["critical"]["step"] for b in (HERE/"target-or-direction.jsonl").read_bytes().splitlines()
              if (r:=json.loads(b))}
    stats=Counter()
    with (HERE/"implied-audit.jsonl").open("x",encoding="utf8") as out:
        for split,name,_,_,level in cases():
            env=Environment(level)
            for step in sorted({1,critical.get(name,1)}):
                for c in runs[name]["actions"][env.steps:step-1]:
                    env.step(MOVES[int(c)])
                v=env.view
                readings=v.fuel_distance,v.repair_distance
                sites=tuple(tuple(locations(v.memory,k)) for k in "FR")
                options=[]
                for move in MOVES:
                    q=destination(v.memory,v.position,move)
                    if q is None:
                        continue
                    actual=tuple(env._lengths[k].get(q) for k in "FR")
                    observed=support(v.memory,v.position,q,level.goal,readings,actual,sites)
                    assert observed>0,(name,step,move.name,actual)
                    if all(d is not None and d>0 for d in readings):
                        both=tuple(d-1 for d in readings)
                        prediction=support(v.memory,v.position,q,level.goal,readings,both,sites)
                        assert prediction>0 or actual!=both
                        stats[actual==both,prediction>0]+=1
                    else:
                        prediction=None
                    options.append(dict(move=move.name,actual=actual,factual_weight=observed,
                                        weight_for_both=prediction))
                out.write(json.dumps(dict(map=name,split=split,step=step,readings=readings,options=options))+"\n")
                out.flush()
    print(stats)


if __name__=="__main__":
    main()
