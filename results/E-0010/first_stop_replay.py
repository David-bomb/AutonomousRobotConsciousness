"""Внешний повтор: вести до выбранной станции, затем вернуть управление прежнему роботу."""
from collections import deque
import argparse
from copy import deepcopy
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment,Status
from maze.grid import destination
from maze.street_trunk import StreetTrunk


def shortest_legs(level, origin, stop_at_every_marker):
    queue=deque([(origin,"")])
    seen={origin}
    paths={}
    while queue:
        p,trail=queue.popleft()
        for i,m in enumerate(MOVES):
            q=destination(level.grid,p,m)
            if q is None or q in seen:
                continue
            seen.add(q)
            next_trail=trail+str(i)
            if level.grid[q[1]][q[0]] in "FRB":
                paths[q]=next_trail
                if stop_at_every_marker:
                    continue
            queue.append((q,next_trail))
    return paths


def trial(job):
    case,run,event=job
    split,name,_,_,level=case
    env=Environment(level)
    robot=StreetTrunk()
    critical=event["critical"]
    for c in run["actions"][:critical["step"]-1]:
        assert robot.choose(env.view)==MOVES[int(c)]
        env.step(MOVES[int(c)])
    paths=shortest_legs(level,env.position,True)
    # Эта часть использует ретроспективный сертификат первого сервиса.
    trials={paths[tuple(s["point"])]:["достижимая первая станция",s["kind"]] for s in event["first"] if s["kind"]!="B"}
    guided=None
    if event["selected"]:
        all_paths=shortest_legs(level,env.position,False)
        chosen=min((path for p,path in all_paths.items() if level.grid[p[1]][p[0]]==event["selected"]),
                   key=lambda path:(len(path),path))
        assert len(chosen)==critical["d"+event["selected"]]
        guided=chosen
        trials.setdefault(chosen,["ближайшая выбранного вида",event["selected"]])
    outcomes=[]
    for trail,(purpose,kind) in trials.items():
        sim,bot=deepcopy(env),deepcopy(robot)
        first_change=None
        for j,c in enumerate(trail):
            if sim.status!=Status.RUNNING:
                break
            suggested=bot.choose(sim.view)
            if suggested!=MOVES[int(c)] and first_change is None:
                first_change=j+1
            sim.step(MOVES[int(c)])
        at_handoff=sim.status.name,sim.steps,sim.position,sim.fuel,sim.health
        suffix=[]
        while sim.status==Status.RUNNING:
            move=bot.choose(sim.view)
            suffix.append(str(MOVES.index(move)))
            sim.step(move)
        outcomes.append(dict(rule=purpose,kind=kind,forced=trail,nearest_selected=trail==guided,
                             first_changed=first_change,handoff=at_handoff,status=sim.status.name,
                             steps=sim.steps,suffix="".join(suffix)))
    return dict(map=name,split=split,critical=critical,selected=event["selected"],
                first_kinds=sorted({s["kind"] for s in event["first"]}),outcomes=outcomes)


def write_summary(rows):
    corrections={r["map"]:r for line in (HERE/"one-correction-street-trunk.jsonl").read_bytes().splitlines()
                 if (r:=json.loads(line))}
    assert len(corrections)==len(rows)==len({r["map"] for r in rows})==87
    groups={}
    paired={"only_one":[],"only_station":[],"both":[],"neither":[]}
    for row in rows:
        key=row["map"]
        critical=row["critical"]
        assert corrections[key]["critical"]==critical
        phase=int(critical["refuels"]>0)+int(critical["repairs"]>0)
        trials=row["outcomes"]
        guided=[t for t in trials if t["nearest_selected"]]
        assert len(guided)==int(row["selected"] is not None)
        eligible=[t for t in trials if t["rule"]=="достижимая первая станция"]
        one=any(t["status"]=="SUCCESS" for t in corrections[key]["alternatives"])
        station=any(t["status"]=="SUCCESS" for t in eligible)
        paired[("neither","only_station","only_one","both")[2*one+station]].append(key)
        for group in ("all",f"phase_{phase}",row["split"]):
            stats=groups.setdefault(group,dict(maps=0,one=0,has_selected=0,selected_finishes=0,goal_during_guidance=0,
                                              has_station=0,station_finishes=0,direct_only=0))
            stats["maps"]+=1
            stats["one"]+=one
            stats["has_selected"]+=bool(guided)
            stats["selected_finishes"]+=any(t["status"]=="SUCCESS" for t in guided)
            stats["goal_during_guidance"]+=any(t["handoff"][0]=="SUCCESS" for t in guided)
            stats["has_station"]+=bool(eligible)
            stats["station_finishes"]+=station
            stats["direct_only"]+=row["first_kinds"]==["B"]
    assert groups["all"]["one"]==28
    (HERE/"first-stop-summary.json").write_text(json.dumps(dict(groups=groups,paired=paired),indent=2)+"\n",
                                              encoding="utf8")
    print(groups,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    runs=read("street-trunk")
    events={r["map"]:r for b in (HERE/"target-or-direction.jsonl").read_bytes().splitlines() if (r:=json.loads(b))}
    assert len(events)==87
    path=HERE/"first-stop-replay.jsonl"
    saved=[json.loads(line) for line in path.read_bytes().splitlines()] if args.resume else []
    done={r["map"] for r in saved}
    assert len(done)==len(saved) and done<=events.keys()
    assert all(r["critical"]==events[r["map"]]["critical"] for r in saved)
    jobs=[(case,runs[case[1]],events[case[1]]) for case in cases() if case[1] in events and case[1] not in done]
    with path.open("a" if args.resume else "x",encoding="utf8") as out:
        for row in map(trial,jobs):
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
            saved.append(row)
    write_summary(saved)
    print(len(jobs)+len(done),"эпизодов")


if __name__=="__main__":
    main()
