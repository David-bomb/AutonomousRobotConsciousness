"""Физические префиксы на новом train и полные попарные поездки с одной поздней сетью."""
import os
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import sys
import torch
from maze.environment import Environment,Status
from .adapter import MOVES
from .common import HERE,DEV,digest,level,read,rows,write
from .early_bank import FOLDER,LATE,LATE_SHA
from .early_policy import complete,load
from .release import differences

CTRL=None


def bundle(tag):
    f=FOLDER/(tag+"-joint400.pt")
    assert digest(LATE)==LATE_SHA
    source=HERE/"weights"/(tag+".pt")
    info=read(source.with_suffix(".json"));assert digest(source)==info["sha256"]
    if not f.exists():
        pack=dict(format="prev-two-early-joint-v1",early=torch.load(source,map_location="cpu",weights_only=True),
                  late=torch.load(LATE,map_location="cpu",weights_only=True))
        torch.save(pack,f)
        write(f.with_suffix(".json"),dict(sha256=digest(f),early=info["sha256"],late=LATE_SHA,
              adapter=digest(HERE/"early_policy.py"),factory="results.E-0016.prev-two.early_policy.load"))
    meta=read(f.with_suffix(".json"))
    assert meta["sha256"]==digest(f) and meta["early"]==info["sha256"] and meta["late"]==LATE_SHA
    assert meta["adapter"]==digest(HERE/"early_policy.py")
    return f


def init(path,ablate):
    global CTRL
    torch.set_num_threads(1)
    CTRL=load(path.read_bytes(),ablate)


def play(job):
    row,full=job
    env=Environment(level(row));CTRL.reset();cache={};actions=[];switch=None;both=None
    while env.status==Status.RUNNING:
        if switch is None and complete(env.view.memory):
            switch=env.steps
            if not full:break
        v=env.view;pair=v.fuel_distance,v.repair_distance
        move=CTRL.choose(v,cache);assert CTRL.previous==pair
        env.step(move);assert env.moved
        actions.append(str(MOVES.index(move)))
        if both is None and env.refuels and env.repairs:both=env.steps
    return dict(map=row["name"],geometry=row["geometry"],tier=row["tier"],handover=switch,both_at=both,
                steps=env.steps,outcome=env.status.name,actions="".join(actions))


def evaluate(tag,split,ablate=False,workers=2):
    f=bundle(tag)
    name=f"{split}-{tag}"+("-ablated" if ablate else "")
    out=FOLDER/(name+".json")
    bank=rows(DEV) if split=="dev" else read(FOLDER/(split+".json"))
    if out.exists():
        old=read(out);assert old["sha256"]==digest(f);return old
    with ProcessPoolExecutor(workers,initializer=init,initargs=(f,ablate)) as pool:
        trips=list(pool.map(play,[(r,split=="dev") for r in bank]))
    gates=[r["handover"] is not None for r in trips];wins=[r["outcome"]=="SUCCESS" for r in trips]
    phases=[sum(g for g,r in zip(gates,trips) if r["tier"]==f"H{i}") for i in range(1,5)]
    tiers=[sum(w for w,r in zip(wins,trips) if r["tier"]==f"H{i}") for i in range(1,5)]
    summary=dict(n=len(trips),reached=sum(gates),reached_tiers=phases,early_success=sum(g or w for g,w in zip(gates,wins)),
                 wins=sum(wins),tiers=tiers,both=sum(r["both_at"] is not None for r in trips),
                 same_step=sum(r["handover"] is not None and r["handover"]==r["both_at"] for r in trips),
                 failed_before=sum(not g and not w for g,w in zip(gates,wins)),
                 failed_after=sum(g and not w for g,w in zip(gates,wins)) if split=="dev" else None,
                 limits=sum(r["outcome"]=="LIMIT" for r in trips))
    result=dict(tag=tag,sha256=digest(f),ablate=ablate,bank=digest(DEV if split=="dev" else FOLDER/(split+".json")),
                summary=summary,episodes=trips)
    write(out,result);print(name,summary,flush=True)
    return result


def main(workers=2):
    tags=("early600","early1200","early1800")
    train={tag:evaluate(tag,"select",workers=workers)["summary"] for tag in ("wave1",*tags)}
    chosen=max(tags,key=lambda t:(train[t]["early_success"],sum(train[t]["reached_tiers"][2:])))
    write(FOLDER/"selection.json",dict(chosen=chosen,train=train,selected_sha256=digest(bundle(chosen))))
    for tag in ("wave1",chosen):evaluate(tag,"hold",workers=workers)
    ref=evaluate("wave1","dev",workers=workers)
    assert ref["summary"]["wins"]==76 and ref["summary"]["tiers"]==[24,22,18,12],ref["summary"]
    run=evaluate(chosen,"dev",workers=workers);abl=evaluate(chosen,"dev",True,workers)
    write(FOLDER/"paired.json",dict(chosen=chosen,versus_parent=differences(ref,run),versus_ablation=differences(abl,run)))


if __name__=="__main__":main(*map(int,sys.argv[1:]))
