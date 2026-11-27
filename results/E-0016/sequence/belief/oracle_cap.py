"""Независимые полные учебные старты для измерения ошибки предсказания и выбора границы."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ...data import level,rows,save,read,digest
from ...policy import MOVES
from ...teacher import Teacher
from .features import HERE
from .joint_model import Joint


def init():
    global net,weights
    torch.set_num_threads(1)
    pack=torch.load(HERE.parent/"runs/joint400/rl.pt",weights_only=True,map_location="cpu")
    net=Joint(pack)
    weights=pack["linear"].numpy()


def trial(row):
    lvl=level(row)
    expert=Teacher(lvl)
    cache={}
    results=[]
    for name in ("ordinary","true_downhill","prevent_loss"):
        env=Environment(lvl)
        actions={}
        corrected=0
        while env.status==Status.RUNNING:
            v=env.view
            if v not in actions:
                if v not in cache:
                    x,ok=net.directions(v)
                    s,group,d,offset=min((v.fuel-v.fuel_distance,0,v.fuel_distance,0),
                                         (v.health-v.repair_distance,1,v.repair_distance,2))
                    closer=np.array([bool(ok[j] and min(ds[destination(v.memory,v.position,m)] for ds in expert.dist[offset:offset+2])<d)
                                    for j,m in enumerate(MOVES)])
                    cache[v]=(x,ok,closer,s,d)
                x,ok,closer,s,d=cache[v]
                value=x@weights
                a=int(np.where(ok,value,-1e12).argmax())
                base=a
                if name=="true_downhill" and not x[:,1:3].any() and s<=4 and d>0 and closer.any():
                    # Сохраняем возможность целевого выхода к границе после двух сервисов.
                    if not x[:,3:5].any():a=int(np.where(closer,value,-1e12).argmax())
                if name=="prevent_loss":
                    _,viable=expert.labels(env)
                    if viable.any() and not viable[a]: a=int(np.where(viable,value,-1e12).argmax())
                actions[v]=a,int(a!=base)
            a,changed=actions[v]
            env.step(MOVES[a]);corrected+=changed
        results.append(dict(mode=name,outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                            changed_steps=corrected))
    expert.remaining.cache_clear()
    return dict(map=row["name"],trials=results)


def main():
    source=HERE.parent/"train.jsonl"
    with ProcessPoolExecutor(2,initializer=init) as pool:
        result=list(pool.map(trial,rows(source)[800:900]))
    summary={}
    for mode in ("ordinary","true_downhill","prevent_loss"):
        trials=[t for r in result for t in r["trials"] if t["mode"]==mode]
        summary[mode]=dict(outcomes=dict(Counter(t["outcome"] for t in trials)),
             both=sum(bool(t["refuels"] and t["repairs"]) for t in trials))
    save(HERE/"oracle-cap.json",dict(source=digest(source),range=[800,900],summary=summary,records=result))
    print(summary,flush=True)


if __name__=="__main__":
    main()
