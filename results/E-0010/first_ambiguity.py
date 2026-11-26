"""Первый неоднозначный выбор без запаса на пробу, включая успешные родительские поездки."""
from collections import Counter
import argparse
import json

from bench import HERE,MOVES,cases
from compare import PhysicalRoutes,read,transition
from maze.environment import Environment
from maze.grid import locations
from maze.pattern_roads import ManyStreets
from maze.pilot_automaton import Mode
from maze.resource_automaton import possible_directions,service_order
from maze.street_trunk import StreetTrunk,StreetTrunkReserve


def trial(job):
    case,run=job
    split,name,tier,family,level=case
    bot,env=StreetTrunk(),Environment(level)
    event=None
    for c in run["actions"]:
        if env.refuels and env.repairs:
            break
        v=env.view
        demand=service_order(v,bot._fallback.mode,bot.RESERVE)
        old=tuple(bot._fallback.last_lengths.get(m) if v.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        chosen=bot.choose(v)
        assert chosen==MOVES[int(c)]
        if demand and demand.mode.value in bot.activity and 0<=demand.stock-demand.distance<2:
            kind=int(demand.mode==Mode.REPAIR)
            options=possible_directions(v,demand,old[kind])
            if len(options)>1:
                routes=PhysicalRoutes(level)
                state=(v.position,v.fuel,v.health)
                safe=[m for m in MOVES if transition(routes,state,m,env.limit-env.steps-1)]
                if not safe:
                    break
                prior=None
                if v.moved:
                    dx,dy=v.last_move.value
                    prior=v.position[0]-dx,v.position[1]-dy
                readings=v.fuel_distance,v.repair_distance
                sites=tuple(tuple(locations(v.memory,k)) for k in "FR")
                choices={"actual":chosen}
                for model,app in (("trunk",bot),("reserve",StreetTrunkReserve()),("many",ManyStreets())):
                    score=app.direction_votes(v,readings,old,prior,sites)[kind]
                    choices[model]=max(options,key=lambda m:(score.get(m,0.),m==chosen))
                event=dict(step=env.steps+1,selected="FR"[kind],pos=v.position,slack=demand.stock-demand.distance,
                           safe=[m.name for m in safe],possible=[m.name for m in options],
                           choice={k:m.name for k,m in choices.items()})
                break
        env.step(chosen)
    return dict(map=name,split=split,tier=tier,family=family,parent_status=run["status"],event=event)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    runs=read("street-trunk")
    totals=Counter()
    # Один процесс избегает одновременного хранения массивов разных эпизодов
    # в нескольких копиях в ходе этой внешней проверки.
    path=HERE/"first-ambiguity.jsonl"
    saved=[json.loads(line) for line in path.read_bytes().splitlines()] if args.resume else []
    existing={r["map"]:r for r in saved}
    assert len(existing)==len(saved) and existing.keys()<=runs.keys()
    assert all(r["parent_status"]==runs[k]["status"] for k,r in existing.items())
    with path.open("a" if args.resume else "x",encoding="utf8") as out:
        for case in cases():
            row=existing.get(case[1])
            if row is None:
                row=trial((case,runs[case[1]]))
                out.write(json.dumps(row,ensure_ascii=False)+"\n")
                out.flush()
            if row["event"]:
                for model,command in row["event"]["choice"].items():
                    totals[model,command in row["event"]["safe"]]+=1
    print(totals)


if __name__=="__main__":
    main()
