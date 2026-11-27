"""Select a symmetry-averaged early action head by held-back physical terminal returns."""
import numpy as np
import torch
from renewable_train import HERE, data, digest, save
from early_trials import compare,physical
from symmetry_policy import Symmetric,load


def main():
    torch.set_num_threads(1)
    out=HERE/"symmetry-search";out.mkdir(exist_ok=True)
    source=HERE/"service-window/fixed/model.pt"
    assert digest(source)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    pack=torch.load(source,weights_only=True,map_location="cpu")
    pack["format"]="symmetry-window-1"
    bank=HERE/"handover-search/train.jsonl"
    rows=data.rows(bank)
    def subset(group,begin,n):
        return [r for quad in zip(*[[r for r in group if r["tier"]==t][begin:begin+n]
                    for t in ("H1","H2","H3","H4")],strict=True) for r in quad]
    train=subset(rows[:100],5,12);select=subset(rows[100:],2,18)
    options=np.array([[0,0,0,0,0,0,0],[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
         [.6,0,.8,0,0,0,0],[1,0,0,3,0,0,0],[1,0,0,1,5,0,0],[0,1,0,0,5,0,0],
         [.5,0,0,1,4,2,4],[.75,.25,1,1,7,3,5],[0,0,0,0,6,0,3]],float)
    save(out/"protocol.json",dict(source_sha256=digest(source),bank_sha256=digest(bank),
        fit=[r["name"] for r in train],select=[r["name"] for r in select],options=options.tolist(),
        source={n:digest(HERE/n) for n in ("symmetry_policy.py","symmetry_train.py","early_trials.py","handover_train.py")},
        method="The same ordinary observation, including previous action, is reexpressed in all eight coordinate systems. Remap logits, service probabilities and viability back to the true action axes. Fit ten early heads against full terminal success, then four best alternatives and their mean against the parent on 72 distinct training starts. Freeze and reload before dev and ablation."))
    model=Symmetric(pack)
    result=compare(out,"fit",train,options,model)
    top=options[np.argsort(-result[1:],kind="stable")[:4]+1]
    shortlist=np.vstack((options[0],top,top.mean(0)))
    final=compare(out,"select",select,shortlist,model)
    coeff=shortlist[final.argmax()]
    ck=out/"model.pt"
    if not ck.exists():
        tmp=ck.with_suffix(".tmp");torch.save(dict(pack,residual=torch.tensor(coeff,dtype=torch.float64)),tmp);tmp.replace(ck)
    save(out/"frozen.json",dict(model_sha256=digest(ck),coefficients=coeff.tolist(),fit=result.tolist(),selection=final.tolist(),
                               protocol_sha256=digest(out/"protocol.json")))
    for label,ablate in (("dev",False),("ablated",True)):
        physical(out,label,load(ck.read_bytes(),ablate))


if __name__=="__main__":main()
