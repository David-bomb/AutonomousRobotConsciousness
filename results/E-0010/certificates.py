"""Сверка логических границ с физическими направлениями в стартовых и критических состояниях."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.grid import destination,locations
from maze.proven_approach import certified


def check(env,old,prior):
    v=env.view
    checks={}
    for k,d in (("F",v.fuel_distance),("R",v.repair_distance)):
        if not d:
            continue
        sites=tuple(locations(v.memory,k))
        sure,possible=certified(v.memory,v.position,d,old.get(k),prior,sites)
        actual={m for m in MOVES if (q:=destination(env.level.grid,v.position,m)) is not None
                and env._lengths[k][q]==d-1}
        assert sure<=actual<=possible,(v.position,k,d,sure,actual,possible)
        checks[k]={name:sorted(m.name for m in group) for name,group in
                   (("certain",sure),("actual",actual),("possible",possible))}
    return checks


def main():
    events={r["map"]:r["critical"] for b in (HERE/"streets-loss-modes.jsonl").read_bytes().splitlines()
            if (r:=json.loads(b))}
    runs=read("streets")
    tally=Counter()
    with (HERE/"certificates.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            env=Environment(level)
            for step in sorted({1,events.get(name,{"step":1})["step"]}):
                old={}
                prior=None
                for c in runs[name]["actions"][env.steps:step-1]:
                    prior=env.position
                    old={"F":env.view.fuel_distance,"R":env.view.repair_distance}
                    env.step(MOVES[int(c)])
                checks=check(env,old,prior)
                for r in checks.values():
                    tally[bool(r["certain"])]+=1
                out.write(json.dumps(dict(map=name,split=split,step=step,checks=checks))+"\n")
    print(tally)


if __name__=="__main__":
    main()
