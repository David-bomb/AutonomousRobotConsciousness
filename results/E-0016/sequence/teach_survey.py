"""Сравнение учителей только на train. Их правила никогда не загружаются политикой."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ..data import digest,level,rows,save
from ..policy import MOVES
from ..teacher import Teacher
from .bank import HERE
from .model import SequenceNet, encode


def initialize():
    global net,prior
    torch.set_num_threads(1)
    net=SequenceNet().eval()
    net.load_state_dict(torch.load(HERE/"runs/mission600/rl.pt",weights_only=True,map_location="cpu"))
    prior=net.wall_prior.numpy().tobytes()


@torch.inference_mode()
def run(row):
    start=level(row)
    expert=Teacher(start)
    master={}
    logs=[]
    for mode in ("survey","sensor","true_sensor"):
        env=Environment(start)
        history={}
        commands=[]
        roles=Counter()
        repeated=False
        while env.status==Status.RUNNING:
            v=env.view
            if v in history:
                a,role=history[v]
                repeated=True
            else:
                if v not in master:
                    x,ok=encode(v,prior)
                    scores,grad=net.heads(torch.from_numpy(x[None]),torch.from_numpy(ok[None]))
                    raw=scores[0].numpy()
                    pred=grad[0].softmax(-1).numpy()
                    paths=x[:,255:295].reshape(8,5,8)
                    known=np.where((paths[:,:,6]>.5)&ok[:,None],paths[:,:,4],10.).min(1)
                    safe=ok&(x[:,447]>.5)
                    front=x[:,415:427].reshape(8,3,4)
                    # Сравнение физических длин, затем оставшейся геометрической дальности.
                    survey=np.where(safe,front[:,2,1],10.)
                    master[v]=(raw,pred,known,survey,front[:,2,2],ok)
                raw,pred,known,survey,end,ok=master[v]
                role="network"
                a=int(raw.argmax())
                if known.min()<10:
                    a=int(np.argmax(np.where(known==known.min(),raw,-1e8)))
                    role="confirmed"
                elif survey.min()<10:
                    low=survey==survey.min()
                    remain=np.where(low,end,10.)
                    a=int(np.argmax(np.where(remain==remain.min(),raw,-1e8)))
                    role="survey"
                else:
                    slack,which,d,field=min((v.fuel-v.fuel_distance,0,v.fuel_distance,0),
                                           (v.health-v.repair_distance,1,v.repair_distance,2))
                    if mode!="survey" and d>0 and slack<=4:
                        role="sensor"
                        k=1-which if v.health<v.fuel else which
                        if mode=="true_sensor":
                            down=np.array([bool(ok[i] and min(ds[destination(v.memory,v.position,m)]
                                        for ds in expert.dist[field:field+2]) <d) for i,m in enumerate(MOVES)])
                            if down.any():
                                a=int(np.argmax(np.where(down,raw,-1e8)))
                        else:
                            a=int(np.argmax(np.where(ok,pred[:,k,0]+.005*np.clip(raw,-10,10),-100)))
                history[v]=(a,role)
            roles[role]+=1
            env.step(MOVES[a])
            commands.append(str(a))
        logs.append(dict(mode=mode,outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                         repeated=repeated,roles=dict(roles),actions="".join(commands)))
    return dict(map=row["name"],episodes=logs)


def main(n):
    assert n%100==0 and 0<n<=600
    bank=rows(HERE/"train.jsonl")[1000:1000+n]
    with ProcessPoolExecutor(2,initializer=initialize) as pool:
        out=[]
        for r in pool.map(run,bank):
            out.append(r)
            if len(out)%20==0:
                print("teacher",len(out),flush=True)
    summary={}
    for mode in ("survey","sensor","true_sensor"):
        trials=[t for r in out for t in r["episodes"] if t["mode"]==mode]
        summary[mode]=dict(outcomes=dict(Counter(t["outcome"] for t in trials)),
                           both=sum(bool(t["refuels"] and t["repairs"]) for t in trials))
    save(HERE/f"survey-teacher-{n}.json",dict(source_sha256=digest(Path(__file__)),train_range=[1000,1000+n],
         train_sha256=digest(HERE/"train.jsonl"),summary=summary,records=out))
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("n",type=int)
    main(p.parse_args().n)
