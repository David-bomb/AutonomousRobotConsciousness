"""Проверка входов контроллера, совпадения вне допуска и физического исполнения."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import sys
import torch
from maze.environment import Environment,Status
from .adapter import MOVES
from .cautious_bank import OUT as FIRST
from .cautious_policy import load as load_first,urgency
from .common import HERE,DEV,digest,level,read,rows,write
from .early_bank import FOLDER
from .early_policy import complete

CTRL=None


def init(path,abl):
    global CTRL
    torch.set_num_threads(1)
    if path.parent==FIRST:factory=load_first
    else:
        from .path_policy import load as factory
    CTRL=factory(path.read_bytes(),abl)


def audit(job):
    row,record,repeat=job;env=Environment(level(row));CTRL.reset();previous=(None,None)
    edits=0;gate=None;both=None;outside=0;cache={}
    for c in record["actions"]:
        v=env.view
        if gate is None and complete(v.memory):gate=env.steps
        if gate is None:
            histogram_key=v,previous
            if histogram_key not in cache:
                cache[histogram_key]=CTRL.decision(v,(None,None) if CTRL.ablate else previous)
            own,change,delta=cache[histogram_key]
            pred=change if delta>CTRL.threshold else own
            assert pred==int(c)
            if pred!=own:
                assert 0<=urgency(v)[1]<=2;edits+=1
            else:outside+=1
        if repeat:
            assert CTRL.previous==previous
            expected=MOVES[int(c)];assert CTRL.choose(v)==expected
            CTRL.previous=previous
            assert CTRL.choose(v,{(v,previous):expected})==expected
            assert CTRL.previous==(v.fuel_distance,v.repair_distance)
        previous=v.fuel_distance,v.repair_distance
        env.step(MOVES[int(c)]);assert env.moved
        if both is None and env.refuels and env.repairs:both=env.steps
    assert env.status!=Status.RUNNING
    assert (gate,both,env.status.name,env.steps)==(record["handover"],record["both_at"],record["outcome"],record["steps"])
    return row["name"],dict(edits=edits,identical_early=outside,steps=env.steps,uncached=repeat)


def main(folder):
    path=HERE/folder;f=path/"frozen.pt";sha=digest(f)
    ref=read(FOLDER/"dev-wave1.json");bank=rows(DEV)
    modes={}
    for abl in (False,True):
        file=path/("dev-ablated.json" if abl else "dev.json");saved=read(file)
        assert sha==saved["sha256"]==read(path/"selection.json")["sha256"]
        cases=saved["episodes"][0]
        changed=[i for i,(r,t) in enumerate(zip(ref["episodes"],cases)) if (r["outcome"]=="SUCCESS")!=(t["outcome"]=="SUCCESS")]
        picks={0,99,*sorted(changed,key=lambda i:cases[i]["steps"])[:2]}
        with ProcessPoolExecutor(2,initializer=init,initargs=(f,abl)) as pool:
            outputs=dict(pool.map(audit,[(r,t,i in picks) for i,(r,t) in enumerate(zip(bank,cases,strict=True))]))
        edits=sum(r["edits"] for r in outputs.values())
        # Первичный оценщик считает замены при сравнении train-порогов. Для dev счётчик восстанавливается здесь.
        saved["summaries"][0]["edits"]=edits
        write(file,saved)
        modes["ablated" if abl else "honest"]=dict(episodes=outputs,edits=edits,edited_maps=sum(r["edits"]>0 for r in outputs.values()),
                                                     uncached_maps=[bank[i]["name"] for i in sorted(picks)])
    write(path/"audit.json",dict(sha256=sha,modes=modes,registered_dev=digest(DEV)))
    print("audited",folder,{k:v["edits"] for k,v in modes.items()},flush=True)


if __name__=="__main__":main(sys.argv[1])
