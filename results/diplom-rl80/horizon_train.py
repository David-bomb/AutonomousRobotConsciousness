"""Check receding-horizon predictions on known training maps, then select by episode returns."""
from dataclasses import replace
import numpy as np
import torch
from early_trials import compare,physical
from exact_train import balanced
from horizon_policy import Horizon,HorizonFields,load
from renewable_train import EXP,HERE,Environment,data,digest,moves,save
from service_window_policy import load as old_load
from maze.grid import destination

OUT=HERE/"horizon-search"


def probe():
    pilot=old_load((HERE/"service-window/fixed/model.pt").read_bytes());model=HorizonFields(192)
    corpus=data.rows(HERE/"handover-search/train.jsonl");full=0;records=[]
    for row in corpus:
        if full<8 and row["family"] in ("corridors","branching"):
            env=Environment(data.level(row));v=replace(env.view,memory=env.level.grid)
            scores=model.estimate(v,(None,None))
            legal=np.array([destination(v.memory,v.position,m) is not None for m in moves])
            assert scores[legal].max()>0 and not scores[~legal].any()
            assert (scores[legal,::2]+1.e-6>=scores[legal,1::2]).all(),row["name"]
            full+=1
        event=data.read(HERE/"early-errors"/(row["name"]+".json"))
        if "fork" not in event:continue
        f=event["fork"];env=Environment(data.level(row));past=(None,None)
        for _ in range(f["step"]):
            v=env.view;a=int(pilot.early_scores(v,past).argmax())
            past=v.fuel_distance,v.repair_distance;env.step(moves[a])
        v=env.view;q=pilot.early_scores(v,past)
        assert int(q.argmax())==f["chosen"] and list(past)==f["previous"]
        z=model.estimate(v,past);legal=np.array([j in f["legal"] for j in range(8)])
        picks=np.where(legal[:,None],z+np.clip(q-q[legal].max(),-1,0)[:,None]*1.e-7,-3).argmax(0)
        records.append(dict(map=row["name"],reason=event["reason"],viable=f["viable"],picks=picks.tolist()))
    summary={s:[sum(r["picks"][j] in r["viable"] for r in records if r["reason"]==s) for j in range(6)]
             for s in ("first-loss","matched-viable")}
    save(OUT/"probe.json",dict(known_training_boards=full,summary=summary,records=records));print(summary,flush=True)


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    parent=HERE/"protected-search/fixed/model.pt";bank=EXP/"sequence/spatial/large.jsonl"
    rows=data.rows(bank);fit=balanced(rows[6200:6300],12);select=balanced(rows[6900:7000],25)
    eye=np.eye(12);options=np.vstack((np.zeros(12),8*eye[:6],eye[6:],4*eye[0]+4*eye[4],4*eye[2]+4*eye[3]))
    save(OUT/"protocol.json",dict(parent=digest(parent),samples=192,bank=digest(bank),fit=[r["name"] for r in fit],
        select=[r["name"] for r in select],options=options.tolist(),sources={n:digest(HERE/n) for n in ("horizon_train.py",
        "horizon_policy.py","observation_tree.cpp","observation_tree.dll","early_trials.py")},
        rule="Plan depth 3/5/7 in hidden road hypotheses. Partition by legal next actions and two gauge changes. Fit plans on half of the generated hypotheses and evaluate only on the other half, then swap. Rare histories use an observation-only fallback. Compare adaptive and fallback values on fully known train trees and the recorded 48 irreversible / 31 matched early forks. Select by terminal SUCCESS on 48 balanced physical starts, four nonzero options and mean against unchanged parent on another 100. Freeze before physical dev and ablation."))
    probe()
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="cross-fit-horizon-1",horizon_samples=192,horizon=torch.zeros(12,dtype=torch.float64))
    net=Horizon(pack);a=compare(OUT,"fit",fit,options,net)
    top=options[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((options[0],top,top.mean(0)))
    b=compare(OUT,"select",select,pool,net);theta=pool[b.argmax()]
    path=OUT/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp");torch.save(dict(pack,horizon=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(path)
    save(OUT/"frozen.json",dict(model_sha256=digest(path),fit=a.tolist(),select=b.tolist(),coefficients=theta.tolist(),
                                protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(path.read_bytes(),ablate))


if __name__=="__main__":main()
