"""Compare frontier commitment and actual reachability at already registered training forks."""
import torch
from departure_policy import Departure
from renewable_train import HERE,Environment,data,digest,moves,save
from service_window_policy import load as baseline
from maze.grid import destination,wave


def main():
    torch.set_num_threads(1)
    out=HERE/"departure-search"
    loss=HERE/"late-errors/summary.json";parent=HERE/"service-window/fixed/model.pt"
    foundation=HERE/"protected-search/fixed/model.pt"
    pack=torch.load(foundation,weights_only=True,map_location="cpu")
    model=Departure(dict(pack,departure_samples=36,departure=torch.zeros(12,dtype=torch.float64)))
    pilot=baseline(parent.read_bytes())
    groups=data.read(loss)["episodes"]
    maps={r["name"]:r for f in ("return-search/bank.jsonl","handover-search/train.jsonl") for r in data.rows(HERE/f)}
    save(out/"probe-protocol.json",dict(forks=digest(loss),source=digest(HERE/"departure_probe.py"),
            model=digest(parent),sampled_model=digest(foundation),native=digest(HERE/"departure.dll"),
            rule="Replay the predetermined baseline training trips up to the first irreversible late fork or matched viable comparison. Count viable choices for default and six probability-based commitment estimates. For descriptive training-only diagnosis, count whether viable actions return to the preceding cell, shorten each real service distance, or shorten true exit distance."))
    records=[]
    for event in groups:
        if event["step"] is None:continue
        env=Environment(data.level(maps[event["map"]]));old=(None,None);cache={}
        for _ in range(event["step"]):
            v=env.view;action=pilot.choose(v,cache);old=v.fuel_distance,v.repair_distance
            env.step(action)
        v=env.view;parent_move=moves.index(pilot.choose(v,cache))
        assert event["fork"]["chosen"]==parent_move
        q,fields=model.late_parts(v,old)
        assert q.argmax()==parent_move
        valid=event["fork"]["viable"];goal=env.level.goal
        bd={cell:distance for cell,distance,_ in wave(env.level.grid,[goal])}
        ways={}
        for j in valid+[parent_move]:
            q0=destination(v.memory,v.position,moves[j]);assert q0 is not None
            mx,my=moves[j].value
            ways[str(j)]=dict(back=v.last_move is not None and tuple(-k for k in v.last_move.value)==(mx,my),
                 novel=sum(v.memory[y][x]=="?" for y in range(max(0,q0[1]-1),min(len(v.memory),q0[1]+2))
                           for x in range(max(0,q0[0]-1),min(len(v.memory[0]),q0[0]+2))),
                 deltas=[bd[q0]-bd[v.position],env._lengths["F"][q0]-v.fuel_distance,env._lengths["R"][q0]-v.repair_distance])
        options=[int((q+12*fields[:,j]).argmax()) for j in range(6)]
        records.append(dict(map=event["map"],tier=event["tier"],win=event["win"],valid=valid,chosen=parent_move,
                            survey=options,action_data=ways,reading=list(old)))
    result={}
    for label,rs in (("first_loss",[r for r in records if not r["win"]]),("matched",[r for r in records if r["win"]])):
        result[label]=dict(n=len(rs),baseline=sum(r["chosen"] in r["valid"] for r in rs),
            candidates=[sum(r["survey"][j] in r["valid"] for r in rs) for j in range(6)],
            viable_back=sum(any(r["action_data"][str(j)]["back"] for j in r["valid"]) for r in rs),
            viable_exit_descent=sum(any(r["action_data"][str(j)]["deltas"][0]<0 for j in r["valid"]) for r in rs))
    save(out/"probe.json",dict(summary=result,records=records));print(result,flush=True)


if __name__=="__main__":main()
