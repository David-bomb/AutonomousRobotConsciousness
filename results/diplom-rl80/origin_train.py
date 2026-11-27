"""Fit conditional-road weights on whole training trips and freeze before dev."""
import numpy as np
import torch
from renewable_train import HERE,data,digest,save
from early_trials import compare,physical
from origin_policy import OriginNet,load


def main():
    torch.set_num_threads(1)
    out=HERE/"origin-search";out.mkdir(exist_ok=True)
    parent=HERE/"service-window/fixed/model.pt"
    assert digest(parent)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="origin-conditioned-1",particles=256,residual=torch.zeros(8,dtype=torch.float64))
    bank=HERE/"return-search/bank.jsonl";rows=data.rows(bank)
    def subset(group,start,n):
        return [r for q in zip(*[[r for r in group if r["tier"]==t][start:start+n]
                              for t in ("H1","H2","H3","H4")],strict=True) for r in q]
    fit=subset(rows[:100],2,13);selection=subset(rows[100:],3,18)
    options=np.array([[0]*8,[4,0,0,0,0,0,2,2],[0,4,0,0,0,0,2,2],[0,0,4,0,0,0,2,2],
        [0,0,0,7,0,0,1,4],[0,2,1,5,0,0,1,4],[0,1,2,4,1,1,2,5],
        [1,0,2,7,3,0,1,6],[0,2,0,3,1,2,5,3],[0,0,2,2,6,3,1,2]],float)
    save(out/"protocol.json",dict(parent=digest(parent),bank=digest(bank),fit=[r["name"] for r in fit],
             selection=[r["name"] for r in selection],options=options.tolist(),particles=pack["particles"],
             sources={n:digest(HERE/n) for n in ("origin_train.py","origin_policy.py","origin_trace.py",
                 "origin_worlds.cpp","origin_worlds.dll","belief_worlds.cpp","early_trials.py","handover_train.py")},
             rule="At ambiguous early forks, simulate latent connected roads conditional on current and two previous actual gauges. Compare weak and strong likelihoods of the resource itinerary marginalized by best compatible possible starting location/elapsed bounds, inferred from currently visible cells. No case identity, episode clock or remembered A. Compare 10 scoring heads on full training starts, best four alternatives and their mean against unchanged parent on independent selection. Stable ties. Physically evaluate reloaded weights on dev and ablation."))
    first=compare(out,"fit",fit,options,OriginNet(pack))
    top=options[np.argsort(-first[1:],kind="stable")[:4]+1]
    pool=np.vstack((options[0],top,top.mean(0)))
    second=compare(out,"select",selection,pool,OriginNet(pack))
    best=pool[second.argmax()]
    path=out/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp");torch.save(dict(pack,residual=torch.tensor(best,dtype=torch.float64)),tmp);tmp.replace(path)
    save(out/"frozen.json",dict(weights=best.tolist(),model_sha256=digest(path),fit=first.tolist(),
                               selection=second.tolist(),protocol_sha256=digest(out/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):
        physical(out,tag,load(path.read_bytes(),ablate))


if __name__=="__main__":main()
