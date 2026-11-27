"""Контрфактические первые действия с обычным продолжением замороженной field3200."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import median
import numpy as np
import torch
from maze.environment import Environment, Status
from ...data import digest, level, read, rows, save
from ...policy import MOVES
from ...teacher import Teacher
from ..branches import fork
from .gates import describe
from .local_model import local_image
from .model import HERE, image
from .transfer_model import FullModel

WEIGHTS=HERE.parent/"runs/field3200/rl.pt"


def init():
    global model, weights
    torch.set_num_threads(1)
    pack=torch.load(WEIGHTS,weights_only=True,map_location="cpu")
    model=FullModel(pack)
    weights=pack["linear"].numpy()


def finish(env):
    return dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs)


def trial(args):
    row, destination=args
    env=Environment(level(row))
    teacher=Teacher(env.level)
    rng=np.random.default_rng(int(row["geometry"][:12],16))
    choices={}

    def inspect(view):
        if view not in choices:
            xs,ok=model.directions(view)
            scores=xs@weights
            action=int(np.where(ok,scores,-1e12).argmax())
            choices[view]=(scores,ok,action)
        return choices[view]

    count=Counter()
    selected={}
    loss=None
    distinct=set()
    while env.status==Status.RUNNING:
        v=env.view
        scores,ok,a=inspect(v)
        if v not in distinct:
            distinct.add(v)
            stage="both" if env.refuels and env.repairs else "one" if env.refuels or env.repairs else "before"
            if loss is None:
                _,viable=teacher.labels(env)
                if viable.any() and not viable[a]:
                    loss=dict(step=env.steps+1,stage=stage,degree=int(ok.sum()))
                    selected["loss"]=fork(env)
            if ok.sum()>1 and (loss is None or stage=="both"):
                count[stage]+=1
                if rng.integers(count[stage])==0:
                    selected[stage]=fork(env)
        env.step(MOVES[a])
    ordinary=finish(env)
    snaps=[]
    if "loss" in selected:
        snaps.append(("loss",selected["loss"]))
    for key in ("both","one","before"):
        if key in selected and all(selected[key].steps!=s.steps for _,s in snaps):
            snaps.append((key,selected[key]))
        if len(snaps)>=3:break
    records=[]
    labels=[]
    for reason,s in snaps:
        v=s.view
        scores,ok,a=inspect(v)
        best,viable=teacher.labels(s)
        endings=[]
        wins=np.zeros(8,bool)
        for i in np.flatnonzero(ok):
            branch=fork(s)
            branch.step(MOVES[i])
            while branch.status==Status.RUNNING:
                pick=inspect(branch.view)[2]
                branch.step(MOVES[pick])
            item=finish(branch)
            endings.append(dict(action=int(i),**item))
            wins[i]=branch.status==Status.SUCCESS
            if i==a:assert item==ordinary
        patch,meta=local_image(v)
        stock=np.array([v.fuel,v.health],np.int16)
        gates,_,_=describe(image(v)[0],meta,stock)
        last=MOVES.index(v.last_move) if v.last_move else 8
        labels.append((gates.astype(np.float16),patch,meta,stock,last,scores.astype(np.float32),ok,viable,best,wins))
        records.append(dict(reason=reason,step=s.steps,refuels=s.refuels,repairs=s.repairs,original=a,
                            alternatives=endings))
    features=destination.with_suffix(".npz")
    with features.with_suffix(".pending").open("wb") as f:
        names=("gates","patch","meta","stocks","last","scores","legal","safe","fast","wins")
        np.savez_compressed(f,**{k:np.stack(col) for k,col in zip(names,zip(*labels))})
    features.with_suffix(".pending").replace(features)
    save(destination,dict(map=row["name"],tier=row["tier"],ordinary=ordinary,first_loss=loss,
                          samples=records,data_sha256=digest(features)))
    teacher.remaining.cache_clear()


def main():
    fit=rows(HERE/"large.jsonl")[200:400]
    held=rows(HERE/"reward-probe.jsonl")
    parent=HERE.parent/"cache/terminal-choices"
    parent.mkdir(exist_ok=True)
    cfg=dict(weights=digest(WEIGHTS),source=digest(HERE/"terminal_choices.py"),fit=[digest(HERE/"large.jsonl"),200,400],
             held=digest(HERE/"reward-probe.jsonl"),policy="greedy from each physical first action")
    path=parent/"config.json"
    if path.exists():assert read(path)==cfg
    else:save(path,cfg)
    groups=[]
    with ProcessPoolExecutor(2,initializer=init) as pool:
        for name,bank in (("fit",fit),("held",held)):
            folder=parent/name
            folder.mkdir(exist_ok=True)
            jobs=[(r,folder/(r["name"]+".json")) for r in bank]
            pending=[pool.submit(trial,item) for item in jobs if not item[1].exists()]
            for j,job in enumerate(as_completed(pending),1):
                job.result()
                if j%20==0:print(name,j,len(pending),flush=True)
            data=[read(p) for _,p in jobs]
            sample_count=mixed=alternatives=rescued=lost=0
            for row in data:
                for case in row["samples"]:
                    outcome=[a["outcome"]=="SUCCESS" for a in case["alternatives"]]
                    sample_count+=1
                    mixed+=int(any(outcome) and not all(outcome))
                    alternatives+=len(outcome)
                    orig=row["ordinary"]["outcome"]=="SUCCESS"
                    rescued+=int(not orig and any(outcome))
                    lost+=int(orig and not all(outcome))
            loss=[r["first_loss"] for r in data if r["first_loss"] is not None]
            summary=dict(maps=len(data),outcomes=dict(Counter(r["ordinary"]["outcome"] for r in data)),
                both=sum(bool(r["ordinary"]["refuels"] and r["ordinary"]["repairs"]) for r in data),
                loss_stages=dict(Counter(r["stage"] for r in loss)),median_loss=median(r["step"] for r in loss) if loss else None,
                tiers={t:sum(r["ordinary"]["outcome"]=="SUCCESS" for r in data if r["tier"]==t) for t in ("H1","H2","H3","H4")},
                samples=sample_count, mixed=mixed, alternatives=alternatives, rescued=rescued, lost=lost)
            groups.append(dict(name=name,summary=summary,episodes=data))
            save(HERE/f"terminal-choices-{name}.json",dict(config=cfg,**groups[-1]))
            print(name,summary,flush=True)


if __name__=="__main__":
    main()
