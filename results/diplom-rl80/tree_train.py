"""Whole-trip policy improvement using the fitted conditional tree sensor."""
import numpy as np
import torch
from renewable_train import HERE,data,digest,save
from early_trials import compare,physical
from tree_policy import TreeNet,load


def main():
    torch.set_num_threads(1)
    out=HERE/"tree-search";parent=HERE/"service-window/fixed/model.pt"
    assert digest(parent)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="tree-length-window-1",sensor=torch.tensor(data.read(out/"sensor.json")["weights"],dtype=torch.float64),
                residual=torch.zeros(7,dtype=torch.float64))
    bank=HERE/"handover-search/train.jsonl";rows=data.rows(bank)
    def subset(group,start,n):
        groups=[[r for r in group if r["tier"]==t][start:start+n] for t in ("H1","H2","H3","H4")]
        return [r for q in zip(*groups,strict=True) for r in q]
    fit=subset(rows[:100],3,15);selection=subset(rows[100:],1,22)
    options=np.array([[0]*7,[3,1,0,0,0,0,0],[6,2,0,0,2,0,0],[12,3,0,0,3,0,0],
        [18,6,0,0,5,0,0],[4,0,3,0,3,4,0],[4,0,0,3,5,3,1],[7,3,3,0,5,3,0],
        [0,0,8,1,8,4,1],[2,3,6,0,2,2,1],[5,0,1,0,1,1,3],[0,0,1,0,0,1,0]],float)
    save(out/"protocol.json",dict(base=digest(parent),bank=digest(bank),fit=[r["name"] for r in fit],select=[r["name"] for r in selection],
         settings=options.tolist(),source={n:digest(HERE/n) for n in ("tree_train.py","tree_policy.py","tree_paths.cpp",
                 "tree_paths.dll","early_trials.py","handover_train.py")},direction_fit=digest(out/"sensor.json"),
         rule="Fixed terminal reward with joint400 after the two observed windows. Compare 12 early heads on 60 training starts, remaining four best nonzero plus mean and zero on 88 separate starts. At deployment the learned likelihood is masked with the base network's legal and sensor constraints. Freeze one checkpoint, then physical dev and ablation."))
    a=compare(out,"fit",fit,options,TreeNet(pack))
    best=options[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((options[0],best,best.mean(0)))
    b=compare(out,"select",selection,pool,TreeNet(pack))
    theta=pool[b.argmax()];p=out/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp");torch.save(dict(pack,residual=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(p)
    save(out/"frozen.json",dict(coefficients=theta.tolist(),model_sha256=digest(p),fit=a.tolist(),selection=b.tolist(),
                              protocol_sha256=digest(out/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):
        physical(out,tag,load(p.read_bytes(),ablate))


if __name__=="__main__":main()
