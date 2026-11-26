"""Сравнить оценки направлений в одних и тех же критических наблюдениях."""
from collections import Counter
import argparse
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.grid import locations
from maze.implied_turn import ImpliedTurn
from maze.pattern_roads import MarkedStreets,ManyStreets
from maze.pilot_automaton import Mode,separation
from maze.proven_approach import certified,smallest_detour
from maze.resource_automaton import Service,possible_directions
from maze.room_roads import room_votes
from maze.street_trunk import StreetTrunk
from maze.translated_paths import translated_votes
from maze.uniform_streets import UniformStreets
from maze.window_bounds import WindowBounds


def compare_detours():
    coverage = json.loads((HERE/"coverage-62-11b594fc31c2.json").read_text(encoding="utf8"))
    unresolved = {name for names in coverage["unsolved"].values() for name in names}
    rows = []
    for line in (HERE/"direction-models.jsonl").read_bytes().splitlines():
        event = json.loads(line)
        state = event["critical"]
        fuel = event["selected"] == "F"
        slack = state["fuel"] - state["dF"] if fuel else state["health"] - state["dR"]
        proposal = smallest_detour(event["possible"], event["guaranteed"], event["possible_with_exit"]) if 0 <= slack < 2 else None
        base = event["opinions"]["trunk"]["choice"]
        choice = proposal or base
        rows.append(dict(map=event["map"],split=event["split"],step=state["step"],service=event["selected"],
                         slack=slack,lengths=event["possible"],confirmed=event["guaranteed"],
                         compatible=event["possible_with_exit"],base=base,command=choice,overridden=choice!=base,
                         signal=proposal is not None,baseViable=base in state["viable"],viable=choice in state["viable"],
                         down=choice in event["viable_downhill"],baseDown=base in event["viable_downhill"],
                         inHard=event["map"] in unresolved,hasSafeDown=bool(event["viable_downhill"])))
    path = HERE/"detour-decisions.jsonl"
    if path.exists():
        assert rows == [json.loads(line) for line in path.read_bytes().splitlines()]
    else:
        path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf8")
    print("Проверены оценки направлений", len(rows), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--rooms",action="store_true",help="сравнить помещения с ранее записанными оценками")
    mode.add_argument("--detour",action="store_true",help="сверить выбор по нижней оценке пути в сохранённых наблюдениях")
    args=parser.parse_args()
    if args.detour:
        return compare_detours()
    events={r["map"]:r for b in (HERE/"target-or-direction.jsonl").read_bytes().splitlines()
            if (r:=json.loads(b))["selected"]}
    runs=read("street-trunk")
    models={"marked":MarkedStreets(),"many":ManyStreets(),"trunk":StreetTrunk(),"implied":ImpliedTurn(),
            "uniform":UniformStreets(),"window":WindowBounds()}
    totals=Counter()
    originals=({r["map"]:r for line in (HERE/"direction-models.jsonl").read_bytes().splitlines()
                if (r:=json.loads(line))} if args.rooms else {})
    path=HERE/("room-critical-directions.jsonl" if args.rooms else "direction-models.jsonl")
    saved={r["map"]:r for line in path.read_bytes().splitlines() if (r:=json.loads(line))} if (
        args.rooms and path.exists()) else {}
    with path.open("a" if args.rooms else "x",encoding="utf8") as out:
        for split,name,_,family,level in cases():
            if name not in events:
                continue
            event=events[name]
            critical=event["critical"]
            if name in saved:
                assert saved[name]["critical"]==critical
                continue
            env=Environment(level)
            old=(None,None)
            prior=None
            for c in runs[name]["actions"][:critical["step"]-1]:
                prior=env.position
                old=(env.view.fuel_distance,env.view.repair_distance)
                env.step(MOVES[int(c)])
            v=env.view
            sites=tuple(tuple(locations(v.memory,k)) for k in "FR")
            actual=v.fuel_distance,v.repair_distance
            i="FR".index(event["selected"])
            demand=Service((Mode.FUEL,Mode.REPAIR)[i],(v.fuel,v.health)[i],actual[i],
                           tuple(p for p in sites[i] if separation(p,v.position)<=actual[i]))
            options=possible_directions(v,demand,old[i])
            assert set(event["downhill"])<={m.name for m in options}
            default=MOVES[int(runs[name]["actions"][env.steps])]
            def summarize(row):
                best=max(row.get(m,0.) for m in options)
                top=[m for m in options if row.get(m,0.)==best]
                return dict(top=[m.name for m in top],choice=(default if default in top else top[0]).name)
            if args.rooms:
                baseline=originals[name]
                assert baseline["critical"]==critical and set(baseline["possible"])=={m.name for m in options}
                opinions={}
                for label,exact in (("room",True),("weighted",False)):
                    scores=room_votes(v,actual,old,prior,sites,exact)
                    prediction=summarize(scores[i]) if scores is not None else baseline["opinions"]["trunk"].copy()
                    opinions[label]=dict(prediction,fallback=scores is None,
                                         support=float(sum(scores[i].values())) if scores is not None else 0.)
                row=dict(map=name,split=split,family=family,selected=event["selected"],critical=critical,
                         viable_downhill=event["viable_downhill"],opinions=opinions,
                         earlier_hit=any(o["choice"] in event["viable_downhill"] for o in baseline["opinions"].values()))
                out.write(json.dumps(row,ensure_ascii=False)+"\n")
                out.flush()
                continue
            guaranteed,possible=certified(v.memory,v.position,actual[i],old[i],prior,sites[i])
            votes={k:summarize(model.direction_votes(v,actual,old,prior,sites)[i]) for k,model in models.items()}
            terrain,matched,pred=translated_votes(v.memory,v.position,level.goal,actual,old,prior,sites)
            votes["translated"]=summarize(pred[i]) if matched else votes["trunk"]
            for model,value in votes.items():
                totals[model]+=value["choice"] in event["viable_downhill"]
            out.write(json.dumps(dict(map=name,split=split,family=family,critical=critical,selected=event["selected"],
                                      downhill=event["downhill"],viable_downhill=event["viable_downhill"],
                                      possible={m.name:n for m,n in options.items()},terrain=terrain,
                                      guaranteed=sorted(m.name for m in guaranteed),
                                      possible_with_exit=sorted(m.name for m in possible),
                                      exact_placements=matched,opinions=votes))+"\n")
            out.flush()
    print(len(events),totals)


if __name__=="__main__":
    main()
