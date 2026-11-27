"""Learn a conditional direction likelihood on tree-compatible partial observations."""
from collections import Counter
import numpy as np
import torch
from geodesic_probe import BASE,EXP,HASH,HERE,Environment,Status,data,moves,prev
from renewable_train import digest,save
from tree_policy import TreeEvidence


def collect(rows,scripts,tag):
    out=HERE/"tree-search";target=out/(tag+".npz")
    if target.exists():
        a=np.load(target);return tuple(a[k] for k in ("x","y","possible","maps","pure","chosen","tier"))
    controller=prev.Controller(BASE,HASH);tree=TreeEvidence()
    records={r["map"]:{} for r in data.read(HERE/"geodesic-search"/(tag+".json"))}
    earlier=data.read(HERE/"geodesic-search"/(tag+".json"))
    for i,r in enumerate(earlier):records[r["map"]][r["step"]]=i
    with np.load(HERE/"geodesic-search"/(tag+".npz")) as saved:
        previous_x,previous_y=saved["x"],saved["y"]
    features=[];labels=[];valid=[];maps=[];pure=[];chosen=[];tier=[]
    for r in rows:
        points=records.get(r["name"],{})
        if not points:continue
        env=Environment(data.level(r));controller.reset();past=(None,None)
        is_tree=r["family"] in ("corridors","branching")
        for step in range(max(points)+1):
            v=env.view
            if step in points:
                i=points[step];e=earlier[i];k=e["kind"]
                z=tree.features(v,past)[:,k];xx=previous_x[i];y=previous_y[i]
                ds=[v.fuel_distance,v.repair_distance]
                if is_tree and ds[k]<=55:
                    assert ((z[:,1]>0)|~y).all(),(r["name"],step,k,ds,y.tolist(),z.tolist())
                if z[:,1].any() and len(e["legal"])>1:
                    mask=np.array([a in e["legal"] for a in range(8)]) & (xx[:,1]>0)
                    assert (mask & y).any()
                    features.append(np.column_stack((z,xx[:,-2:])))
                    labels.append(y);valid.append(mask);maps.append(r["name"]);pure.append(is_tree)
                    chosen.append(e["model"]);tier.append(r["tier"])
            m=moves[int(scripts[r["name"]][step])] if r["name"] in scripts else controller.choose(v)
            past=(v.fuel_distance,v.repair_distance);env.step(m)
    ans=tuple(np.asarray(a) for a in (features,labels,valid,maps,pure,chosen,tier))
    np.savez_compressed(target,**dict(zip(("x","y","possible","maps","pure","chosen","tier"),ans)))
    return ans


def main():
    torch.set_num_threads(1);out=HERE/"tree-search";out.mkdir(exist_ok=True)
    datafile=EXP/"sequence/train.jsonl";rows=data.rows(datafile)
    commands={r["map"]:r["actions"] for r in data.read(EXP/"sequence/clean-400.json")["episodes"]}
    registration=out/"sensor-protocol-v2.json"
    save(registration,dict(train_corpus=digest(datafile),parent=HASH,
        prior_attempt="Stopped before fitting: stored demonstration actions are numeric characters. Parse each as one action index.",
        files={n:digest(HERE/n) for n in ("tree_policy.py","tree_probe.py","tree_paths.cpp","tree_paths.dll")},
        scope="Fit first 200 demo and 100 own training trajectories from the prior geodesic registration, validate on 100 separate own training maps. Restrict measurement to supported tree hypotheses. All true decreasing moves must have positive support on known tree-family training maps with requested radius <=55. World family is used only for performance checks."))
    sets=[collect(r,c,tag) for r,c,tag in ((rows[:200],commands,"train-demo"),(rows[500:600],{},"train-rollout"),
                                             (rows[1500:1600],{},"probe-rollout"))]
    xx,yy,mask,names,*_=tuple(np.concatenate((a,b)) for a,b in zip(sets[0],sets[1]))
    x=torch.tensor(xx,dtype=torch.float32);y=torch.tensor(yy,dtype=torch.float32);m=torch.tensor(mask)
    freq=Counter(names);weight=torch.tensor([1./freq[n] for n in names])
    coeff=torch.zeros(8,requires_grad=True);opt=torch.optim.Adam([coeff],lr=.06)
    for i in range(360):
        pred=(x@coeff).masked_fill(~m,-1000)
        loss=(-((pred.log_softmax(1)*y).sum(1)/y.sum(1))*weight).sum()/weight.sum()+.001*coeff.square().sum()
        opt.zero_grad();loss.backward();opt.step()
    chosen=coeff.detach().numpy()
    save(out/"sensor.json",dict(weights=chosen.tolist(),protocol=digest(registration)))
    x,y,m,names,pure,parent,tier=sets[2]
    def evaluate(which):
        def total(raw):return int(y[np.arange(len(x)),np.where(m,raw,-1.e6).argmax(1)][which].sum())
        return dict(n=int(which.sum()),parent=int(y[np.arange(len(y)),parent][which].sum()),
                    sensor=total(x[:,:,-2]),tree=total(x[:,:,1]),learned=total(x@chosen))
    doc={k:evaluate(which) for k,which in (("eligible",np.ones(len(y),bool)),("tree",pure),("other",~pure),
                                          ("H4",tier=="H4"),("multiple",(x[:,:,1]>0).sum(1)>1))}
    save(out/"sensor-probe.json",doc);print(doc,flush=True)


if __name__=="__main__":main()
