"""Отбор вклада пространственного прогноза по терминальным физическим поездкам train."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import time
import numpy as np
import torch
from maze.environment import Environment,Status
from ...data import OUT,rows,read,level,save,digest
from ...dev import evaluate
from ...policy import MOVES
from ..belief.population import subset
from .model import HERE
from .transfer_model import FullModel,load


def package():
    base=HERE.parent/"runs/joint400/rl.pt"
    record=read(HERE/"field-3000/results.json")
    field=HERE/f"field-3000/s{record['selected']}.pt"
    return dict(foundation=torch.load(base,weights_only=True,map_location="cpu"),
                field=torch.load(field,weights_only=True,map_location="cpu"))


def init():
    global net
    torch.set_num_threads(1)
    net=FullModel(package())


def trial(job):
    row,weights=job
    env=Environment(level(row))
    cache={}
    weights=np.array(weights,np.float64).T
    results=[]
    for index in range(weights.shape[1]):
        env.reset()
        while env.status==Status.RUNNING:
            v=env.view
            if v not in cache:
                x,ok=net.directions(v)
                cache[v]=np.where(ok[:,None],x@weights,-1e12).argmax(0)
            env.step(MOVES[cache[v][index]])
        results.append(dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs))
    return dict(map=row["name"],tier=row["tier"],episodes=results)


def stage(pool,bank,settings,folder):
    folder.mkdir(exist_ok=True)
    pending=[pool.submit(trial,(row,settings)) for row in bank if not (folder/(row["name"]+".json")).exists()]
    for i,fut in enumerate(as_completed(pending),1):
        r=fut.result()
        save(folder/(r["map"]+".json"),r)
        if i%12==0:print(folder.name,i,len(pending),flush=True)
    records=[read(folder/(r["name"]+".json")) for r in bank]
    rank=[]
    for j,w in enumerate(settings):
        wins=Counter(r["tier"] for r in records if r["episodes"][j]["outcome"]=="SUCCESS")
        rank.append(dict(weights=w,wins=sum(wins.values()),minimum=min(wins[t] for t in ("H1","H2","H3","H4"))))
    return dict(rank=rank,records=records)


def main():
    extra=rows(HERE/"extra.jsonl")
    selection=rows(HERE/"selection.jsonl")
    root=HERE.parent/"runs/field3200"
    root.mkdir(exist_ok=True,parents=True)
    zero=[1,0,0,0,0,0,0,0]
    candidates=[zero]
    for scale,old in ((3,0),(6,-.5),(12,-1)):
        for mix in ((1,1,0,1),(1,2,1,0),(1,2,1,1),(0,1,0,2),(1,0,0,2)):
            candidates.append([1,old,*[scale*t for t in mix],6,2])
    cfg=dict(screen=40,selection=100,weights=candidates,field=read(HERE/"field-3000/results.json")["selected"],
             foundation=digest(HERE.parent/"runs/joint400/rl.pt"),selection_bank=digest(HERE/"selection.jsonl"),
             sources={n:digest(HERE/n) for n in ("transfer.py","transfer_model.py","field_model.py","transfer_adapter.py")})
    if (root/"config.json").exists():assert read(root/"config.json")==cfg
    else:save(root/"config.json",cfg)
    cache=HERE.parent/"cache/physical-field";cache.mkdir(exist_ok=True)
    with ProcessPoolExecutor(3,initializer=init) as pool:
        screen=stage(pool,subset(extra[2900:3000],10),candidates,cache/"screen")
        save(root/"screen.json",screen)
        ranked=sorted(screen["rank"],key=lambda r:(-r["wins"],-r["minimum"]))
        choices=[r["weights"] for r in ranked[:4]]
        if zero not in choices:choices.append(zero)
        result=stage(pool,selection[:100],choices,cache/"selection")
        save(root/"selection.json",result)
    best=max(result["rank"],key=lambda r:(r["wins"],r["minimum"]))
    pack=package()
    pack["linear"]=torch.tensor(best["weights"],dtype=torch.float64)
    path=root/"rl.pt"
    if not path.exists():torch.save(pack,path)
    sig=digest(path)
    save(root/"locked-before-dev.json",dict(sha256=sig,selection=best,dev_sha256=digest(OUT/"data/dev.jsonl")))
    t=time.monotonic();summary,episodes=evaluate(load(path.read_bytes()),rows(OUT/"data/dev.jsonl"))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,seconds=time.monotonic()-t))
    print(best,summary,flush=True)


if __name__=="__main__":
    main()
