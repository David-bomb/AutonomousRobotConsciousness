"""Fit early actions by terminal reward using a simulated next sensor observation."""
import numpy as np
import torch
from renewable_train import EXP,HERE,data,digest,save
from early_trials import compare,physical
from lookahead_policy import Lookahead,load


def main():
    torch.set_num_threads(1);out=HERE/"lookahead-search";out.mkdir(exist_ok=True)
    source=HERE/"protected-search/fixed/model.pt"
    parent=data.read(source.with_name("frozen.json"))["model_sha256"];assert digest(source)==parent
    pack=torch.load(source,weights_only=True,map_location="cpu")
    pack.update(format="partial-two-step-1",lookahead_samples=256,lookahead=torch.zeros(10,dtype=torch.float64))
    bank=EXP/"sequence/spatial/large.jsonl";rows=data.rows(bank)
    # Two balanced blocks, independent of the conditional morphology reward comparison.
    fit=[r for q in zip(*[[r for r in rows[8900:9000] if r["tier"]==t][:15] for t in ("H1","H2","H3","H4")],strict=True) for r in q]
    select=rows[9300:9400]
    choices=np.array([[0]*10,[0,0,8,0,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0,0],
        [0,0,5,4,4,0,0,0,0,0],[0,0,8,2,1,0,0,0,0,0],[3,2,2,2,2,1,0,0,0,0],
        [3,1,0,3,0,8,2,0,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,.6,.6],[0,0,3,0,3,0,1,3,0,0],[0,0,4,0,0,0,-3,0,0,0]],float)
    save(out/"protocol.json",dict(parent=parent,bank=digest(bank),samples=pack["lookahead_samples"],options=choices.tolist(),
        fit=[r["name"] for r in fit],select=[r["name"] for r in select],source={n:digest(HERE/n) for n in
        ("sensor_lookahead.cpp","sensor_lookahead.dll","lookahead_policy.py","lookahead_train.py","protected_policy.py","early_trials.py")},
        method="At an ambiguous early fork, all sampled worlds with the same observable next local legality and same sensor changes share one next action. Train relative expected viability, progress, value of the adaptive observation and two sparse switching rules against full trip completion. Twelve candidates on 60 balanced training starts. Four leading alternatives and their average against zero on 100 separate training starts. Freeze, reload, dev and ablation."))
    model=Lookahead(pack);a=compare(out,"fit",fit,choices,model)
    leading=choices[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((choices[0],leading,leading.mean(0)))
    b=compare(out,"select",select,pool,model);theta=pool[b.argmax()]
    path=out/"model.pt"
    if not path.exists():
        tmp=path.with_suffix(".tmp");torch.save(dict(pack,lookahead=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(path)
    save(out/"frozen.json",dict(weights=theta.tolist(),model_sha256=digest(path),fit=a.tolist(),select=b.tolist(),
                               protocol_sha256=digest(out/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(out,tag,load(path.read_bytes(),ablate))


if __name__=="__main__":main()
