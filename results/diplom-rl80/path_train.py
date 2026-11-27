"""Fit path laws on thousands of train maps and select their use by physical terminal reward."""
import ctypes
from dataclasses import replace
import numpy as np
import torch
from early_trials import compare,physical
from exact_train import balanced
from path_policy import PathNet,load
from renewable_train import EXP,HERE,Environment,data,digest,moves,save
from renewable_policy import targets
from service_window_policy import load as previous_load
from maze.grid import destination

OUT=HERE/"path-search"


def train_counts(rows):
    dest=OUT/"counts.pt"
    if dest.exists():return torch.load(dest,weights_only=True,map_location="cpu")
    table=np.zeros((576,9,8,9),dtype="f4");prior=np.zeros((576,56),dtype="f4")
    dll=ctypes.CDLL(str(HERE/"path_counts.dll")).learn_paths
    ci=ctypes.c_int;floats=np.ctypeslib.ndpointer(np.float32)
    dll.argtypes=[ctypes.c_char_p,ci,ci,np.ctypeslib.ndpointer(np.int32),floats,floats];dll.restype=None
    for i,r in enumerate(rows):
        grid=tuple(r["grid"]);goal=next((row.index("B"),y) for y,row in enumerate(grid) if "B" in row)
        ids=targets(grid,goal)
        dll("".join(grid).encode(),len(grid[0]),len(grid),ids,table,prior)
        if (i+1)%1000==0:print("training road laws",i+1,flush=True)
    counts=dict(path_counts=torch.from_numpy(table),path_lengths=torch.from_numpy(prior))
    tmp=dest.with_suffix(".tmp");torch.save(counts,tmp);tmp.replace(dest)
    save(OUT/"training.json",dict(maps=len(rows),target_cell_distances=int(prior.sum()),
                                  source=digest(HERE/"path_counts.dll"),weights=digest(dest)))
    return counts


def probe(net):
    base=previous_load((HERE/"service-window/fixed/model.pt").read_bytes())
    checks=0;records=[]
    for r in data.rows(HERE/"handover-search/train.jsonl"):
        env=Environment(data.level(r));v=replace(env.view,memory=env.level.grid)
        if checks<100:
            for k in (0,1):
                scores=net.paths.get(v,(None,None),k)
                true=env._lengths["FR"[k]]
                down=[a for a,m in enumerate(moves) if (q:=destination(v.memory,v.position,m)) is not None and true[q]<true[v.position]]
                assert np.isclose(scores[:,0].sum(),1.) and all(not scores[a].any() for a in range(8) if a not in down)
            checks+=1
        ref=data.read(HERE/"early-errors"/(r["name"]+".json"))
        if "fork" not in ref:continue
        f=ref["fork"];past=(None,None)
        for _ in range(f["step"]):
            v=env.view;a=int(base.early_scores(v,past).argmax());past=v.fuel_distance,v.repair_distance;env.step(moves[a])
        v=env.view;q=base.early_scores(v,past);k,_=net.api.urgency(v)
        assert int(q.argmax())==f["chosen"] and list(past)==f["previous"]
        z=net.paths.get(v,past,k);legal=np.asarray([a in f["legal"] for a in range(8)])
        picks=np.where(legal[:,None],z+np.clip(q-q.max(),-1,0)[:,None]*1.e-7,-3).argmax(0)
        records.append(dict(map=r["name"],reason=ref["reason"],valid=f["viable"],choices=picks.tolist(),scores=z.tolist()))
    stats={s:[sum(r["choices"][j] in r["valid"] for r in records if r["reason"]==s) for j in range(6)]
           for s in ("first-loss","matched-viable")}
    save(OUT/"probe.json",dict(known_maps=checks,known_gauge_cases=checks*2,summary=stats,records=records));print(stats,flush=True)


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    parent=HERE/"protected-search/fixed/model.pt";bank=EXP/"sequence/spatial/large.jsonl";rows=data.rows(bank)
    learn=rows[:4000];fit=balanced(rows[10000:10100],15);select=balanced(rows[10400:10500],25)
    eye=np.eye(12);options=np.vstack((np.zeros(12),8*eye[:6],eye[6:],4*eye[3]+4*eye[5],4*eye[0]+4*eye[2]))
    save(OUT/"protocol.json",dict(parent=digest(parent),training=[r["name"] for r in learn],fit=[r["name"] for r in fit],
        select=[r["name"] for r in select],bank_sha256=digest(bank),settings=options.tolist(),sources={n:digest(HERE/n) for n in
        ("path_train.py","path_policy.py","learned_paths.cpp","learned_paths.dll","path_counts.dll","early_trials.py")},
        rule="Fit aggregate action transitions and local-road frequencies for geodesics to each service and the exit on 4000 training terrains. Condition on remaining length, displacement, lattice parity, and heading. Integrate finite continuations through the actual observed walls and two service readings. Check every positive first move on fully observed training maps. Learn influence on complete episode returns using 60 balanced training starts; four best alternatives and mean against parent on 100 other balanced starts. Freeze before dev and ablation."))
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="learned-paths-1",paths=torch.zeros(12,dtype=torch.float64),**train_counts(learn))
    net=PathNet(pack);probe(net)
    a=compare(OUT,"fit",fit,options,net)
    top=options[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((options[0],top,top.mean(0)))
    b=compare(OUT,"select",select,pool,net)
    theta=pool[b.argmax()];dest=OUT/"model.pt"
    if not dest.exists():
        tmp=dest.with_suffix(".tmp");torch.save(dict(pack,paths=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(dest)
    save(OUT/"frozen.json",dict(model_sha256=digest(dest),fit=a.tolist(),select=b.tolist(),coefficients=theta.tolist(),
                                protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(dest.read_bytes(),ablate))


if __name__=="__main__":main()
