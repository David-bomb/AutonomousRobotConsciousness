"""Физическое продолжение 32 учебных циклов после добавления границ с одним возвратом."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import digest,level,read,rows,save
from ...policy import MOVES
from ...teacher import Teacher
from ..model import SequenceNet,encode
from .features import HERE
from .itineraries import measures


def init():
    global net,prior
    torch.set_num_threads(1)
    net=SequenceNet().eval()
    net.load_state_dict(torch.load(HERE.parent/"runs/mission600/rl.pt",weights_only=True,map_location="cpu"))
    prior=net.wall_prior.numpy().tobytes()


@torch.inference_mode()
def trial(job):
    row,prefix=job
    base=level(row)
    t=Teacher(base)
    cache={}
    trials=[]
    for mode in ("one","joint","optimistic"):
        env=Environment(base)
        for c in prefix:env.step(MOVES[int(c)])
        commands={}
        roles=Counter()
        while env.status==Status.RUNNING:
            v=env.view
            if v not in commands:
                if v not in cache:
                    x,ok=encode(v,prior)
                    raw=net(torch.from_numpy(x[None]),torch.from_numpy(ok[None]))[0].numpy()
                    route=x[:,255:295].reshape(8,5,8)
                    lengths=np.where((route[:,:,6]>.5)&ok[:,None],route[:,:,4]*75,10000).min(1)
                    frontier=x[:,415:427].reshape(8,3,4)[:,:,1]*50
                    feasible=x[:,439:451].reshape(8,3,4)[:,:,1]>.5
                    frontier=np.where(feasible&ok[:,None],frontier,10000)
                    onward=measures(v,prior)
                    s,group,d,offset=min((v.fuel-v.fuel_distance,0,v.fuel_distance,0),(v.health-v.repair_distance,1,v.repair_distance,2))
                    down=np.array([bool(ok[j] and min(dist[destination(v.memory,v.position,m)] for dist in t.dist[offset:offset+2])<d)
                                   for j,m in enumerate(MOVES)])
                    cache[v]=(raw,lengths,frontier,onward,s,d,down,ok)
                raw,lengths,front,onward,s,d,down,ok=cache[v]
                alternatives=[("known",lengths),("both",front[:,2])]
                if mode!="one":
                    for c in (1,0) if mode=="optimistic" else (1,):
                        for terrain in (2,1,0):
                            col=onward[:,terrain,c,0]
                            alternatives.append((f"future-{c}",np.where(col<300,col,10000)))
                alternatives.append(("one",front[:,1]))
                role="network"
                a=int(raw.argmax())
                for tag,ds in alternatives:
                    if ds.min()<10000:
                        a=int(np.argmax(np.where(ds==ds.min(),raw,-1e8)))
                        role=tag
                        break
                else:
                    if s<=4 and d>0 and down.any():
                        a=int(np.argmax(np.where(down,raw,-1e8)))
                        role="true_sensor"
                commands[v]=a,role
            a,role=commands[v]
            roles[role]+=1
            env.step(MOVES[a])
        trials.append(dict(mode=mode,outcome=env.status.name,steps=env.steps,roles=dict(roles)))
    return dict(map=row["name"],prefix=len(prefix),trials=trials)


def main():
    root=HERE.parent
    source=root/"survey-teacher-100.json"
    cycle=root/"cycle-audit.json"
    lookup={r["map"]:r["first"] for r in read(cycle)["episodes"] if r["period"]>0}
    bank={r["name"]:r for r in rows(root/"train.jsonl")}
    jobs=[]
    for item in read(source)["records"]:
        name=item["map"]
        if name not in lookup:continue
        actions=next(r for r in item["episodes"] if r["mode"]=="true_sensor")["actions"]
        jobs.append((bank[name],actions[:lookup[name]]))
    with ProcessPoolExecutor(2,initializer=init) as workers:
        records=list(workers.map(trial,jobs))
    summary={name:dict(Counter(t["outcome"] for r in records for t in r["trials"] if t["mode"]==name))
             for name in ("one","joint","optimistic")}
    save(HERE/"rescue.json",dict(source=digest(source),cycles=digest(cycle),n=len(records),summary=summary,records=records))
    print(summary,flush=True)


if __name__=="__main__":
    main()
