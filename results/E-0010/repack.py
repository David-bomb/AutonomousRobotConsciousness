"""Можно ли перед самой ошибкой пополниться и вернуться в ту же клетку."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read
from maze.environment import Environment
from preparation import alternative


def main():
    original = read("street-trunk")
    failures = {r["map"]:r for b in (HERE/"preparation-street-trunk.jsonl").read_bytes().splitlines()
                if (r:=json.loads(b))}
    totals=Counter()
    with (HERE/"repack-street-trunk.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in failures:
                continue
            loss=failures[name]["critical"]
            env=Environment(level)
            trail=original[name]["actions"]
            for c in trail[:loss["step"]-1]:
                env.step(MOVES[int(c)])
            move=MOVES[int(trail[env.steps])]
            witness=alternative(level,loss,env.view.memory,move,PhysicalRoutes(level),env.limit,env)
            if witness:
                assert len(witness["actions"]) == witness["steps"]-env.steps > 0
            phase=int(bool(env.refuels))+int(bool(env.repairs))
            totals[phase,bool(witness)]+=1
            out.write(json.dumps(dict(map=name,split=split,critical=loss,detour=witness))+"\n")
            out.flush()
    print(totals)


if __name__=="__main__":
    main()
