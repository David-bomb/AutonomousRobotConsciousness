"""Определить режим планирования в фактически потерянных состояниях streets."""
import json
from collections import Counter

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment
from maze.excursion_automaton import excursion, shelters
from maze.frontier_automaton import _paths
from maze.grid import locations


def main():
    rows=read("streets")
    pairs=[json.loads(b) for b in (HERE/"range-roads--streets.jsonl").read_bytes().splitlines()]
    losses={r["map"]:r["child_loss"] for r in pairs if "child_loss" in r}
    totals=Counter()
    with (HERE/"streets-loss-modes.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in losses:
                continue
            loss=losses[name]
            env=Environment(level)
            track=rows[name]["actions"]
            for c in track[:loss["step"]-1]:
                env.step(MOVES[int(c)])
            v=env.view
            goal=(v.position[0]+v.dx,v.position[1]+v.dy)
            stops=tuple((p,k) for k in "FR" for p in locations(v.memory,k))
            plan=excursion(v,goal,stops,True)
            mode="до открытия базы" if plan is None else plan[0]
            if plan:
                assert plan[1].name==loss["move"]
            known=_paths(v.memory,v.position)
            bases=[p for p in shelters(v.memory,stops) if p in known]
            known_d=[min((known[p][0] for p,t in stops if t==k and p in known),default=None) for k in "FR"]
            totals[(mode,env.refuels>0,env.repairs>0)]+=1
            row=dict(map=name,split=split,family=family,mode=mode,known_sites=known_d,
                     reachable_base_cells=len(bases),known_goal=known.get(goal,(None,))[0],
                     time_left=env.limit-env.steps,critical=loss)
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(totals)


if __name__=="__main__":
    main()
