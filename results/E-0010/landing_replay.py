"""Где фактически кончается заезд к известной мастерской и повторяется ли поездка."""
import argparse
import json
from math import inf

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, first_loss, replay
from maze.environment import Environment
from maze.grid import cell
from maze.renewable_routes import renewable_routes


def repeating_tail(actions,states):
    end=len(actions)
    for period in range(1,end//2+1):
        if states[end]!=states[end-period] or actions[end-period:]!=actions[end-2*period:end-period]:
            continue
        begin=end-period
        while begin>0 and actions[begin-1]==actions[begin-1+period] and states[begin-1]==states[begin-1+period]:
            begin-=1
        repeats=(end-begin)//period
        if repeats>=2:
            return dict(start=begin,period=period,repeats=repeats,unchanged_memory=states[end][-1]==states[begin][-1])
    return None


def analyze(level,row,critical,base_actions):
    env=Environment(level)
    routes=PhysicalRoutes(level)
    states=[]
    memories={}
    first_station=None
    next_discovery=None
    known_base={}
    first_diff=next((i+1 for i,(a,b) in enumerate(zip(base_actions,row["actions"])) if a!=b),None)
    reference=min(first_diff or critical["step"],critical["step"])
    for t in range(row["steps"]+1):
        v=env.view
        memory_id=memories.setdefault(v.memory,len(memories))
        states.append((v.position,v.fuel,v.health,v.last_move,v.moved,memory_id))
        if t==reference-1:
            if reference==critical["step"] and (
                    list(v.position),v.fuel,v.health)!=(critical["pos"],critical["fuel"],critical["health"]):
                raise ValueError("Поездки разошлись до исходного контрольного состояния")
            stops=tuple(((x,y),k) for y,line in enumerate(v.memory) for x,k in enumerate(line) if k in "FR")
            known_base=renewable_routes(v.memory,stops)
        kind=cell(level.grid,*v.position)
        if t>=reference and kind in "FRB" and first_station is None:
            first_station=dict(step=t,point=v.position,kind=kind,fuel=v.fuel,health=v.health,
                               known_renewable=v.position in known_base,
                               possible_finish=routes.allows(v.position,v.fuel,v.health,env.limit-t),
                               unknown=sum(line.count("?") for line in v.memory))
        elif first_station and next_discovery is None and memory_id!=states[first_station["step"]][-1]:
            unseen=sum(line.count("?") for line in v.memory)
            if unseen<first_station["unknown"]:
                next_discovery=dict(step=t,point=v.position,new_cells=first_station["unknown"]-unseen)
        if t<row["steps"]:
            env.step(MOVES[int(row["actions"][t])])
    assert (env.status.name,env.refuels,env.repairs)==(row["status"],row["refuels"],row["repairs"])
    result=dict(first_difference=first_diff,reference_step=reference,first_station=first_station,
                next_discovery=next_discovery,states=len(states),status=row["status"],steps=row["steps"],
                refuels=env.refuels,repairs=env.repairs,repeating_tail=repeating_tail(row["actions"],states))
    if row["status"]!="SUCCESS":
        track,terminal=replay(level,row)
        loss=first_loss(routes,track,terminal)
        p,f,h=states[loss["step"]][:3]
        result.update(first_loss=loss,finish_without_deadline=routes.allows(p,f,h,inf))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup",action="store_true")
    args=parser.parse_args()
    name="fresh-H1-11"
    case=next(c for c in cases() if c[1]==name)
    record=next(r for s in (HERE/"known-first-stops.jsonl").read_bytes().splitlines() if (r:=json.loads(s))["map"]==name)
    results={"map":name,"before":record["critical"],"known_first":record["choices"],"runs":{}}
    baseline=next(r for line in (HERE/"street-trunk-reserve.jsonl").read_bytes().splitlines()
                  if (r:=json.loads(line))["map"]==name)
    agents=(("first-landing","after-landing","outward-after-landing") if args.followup else
            ("street-trunk-reserve","known-landing","boundary-landing","first-landing"))
    for agent in agents:
        row=next(r for line in (HERE/f"{agent}.jsonl").read_bytes().splitlines() if (r:=json.loads(line))["map"]==name)
        results["runs"][agent]=analyze(case[4],row,record["critical"],baseline["actions"])
    path=HERE/("landing-followup.json" if args.followup else "landing-first-stop.json")
    path.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(name,{k:(r["status"],r["steps"],r["first_station"]) for k,r in results["runs"].items()},flush=True)


if __name__=="__main__":
    main()
