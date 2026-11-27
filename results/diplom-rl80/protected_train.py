"""Preregister a zero-shot composition and a separate training-only policy-improvement arm."""
import sys
import numpy as np
import torch
from renewable_train import EXP,HERE,data,digest,save
from protected_policy import Protected,load
from early_trials import compare,physical


def main(which):
    torch.set_num_threads(1)
    top=HERE/"protected-search";top.mkdir(exist_ok=True)
    out=top/which;out.mkdir(exist_ok=True)
    tree=HERE/"tree-search/model.pt";previous=EXP/"prev-two/conservative/frozen.pt"
    assert digest(tree)=="839140fa1346c675d550f6ee4ace6aaa5eb9f7a1d07c254a2be8a6e86f6f0bed"
    assert digest(previous)=="f9bd509a7d4778f4d0e33cd1ad3924f8eb8a0923086c8448b42ac7ebddee35bf"
    pack=torch.load(tree,weights_only=True,map_location="cpu");change=torch.load(previous,weights_only=True,map_location="cpu")
    assert all(torch.equal(v,change["early"][k]) for k,v in pack["early"].items())
    pack.update(format="protected-tree-morph-1",samples=384,cautious_dim=change["dim"],
            cautious_state=change["correction"],cautious_threshold=float(change["threshold"]),morph=torch.zeros(12,dtype=torch.float64))
    bank=HERE/"panel-search/holdout.jsonl";rows=data.rows(bank)
    fit=[r for quad in zip(*[[r for r in rows[:100] if r["tier"]==t][:16] for t in ("H1","H2","H3","H4")],strict=True) for r in quad]
    selection=rows[100:]
    options=np.vstack((np.zeros(12),np.eye(12),np.eye(12)[2]+np.eye(12)[7],
                       .7*np.eye(12)[2]+.7*np.eye(12)[8]+.7*np.eye(12)[9]))
    save(out/"protocol.json",dict(tree=digest(tree),conservative=digest(previous),samples=384,fit=[r["name"] for r in fit],
        select=[r["name"] for r in selection],bank_sha256=digest(bank),settings=options.tolist(),
        branch=which,sources={n:digest(HERE/n) for n in ("protected_policy.py","protected_train.py","tree_policy.py","morph_policy.py",
            "morphology_worlds.dll","tree_paths.dll","early_trials.py")},
        rule="When the existing conservative head exceeds its frozen margin, take its action. Otherwise use the existing tree77 policy. Zero-shot arm freezes this composition directly. Learned arm compares twelve confidence-gated switches and two combinations of conditional road predictions plus the parent. Use actual terminal SUCCESS on 64 fit maps; top four nonzero plus their mean versus zero on 100 separate starts. Freeze, reload, physically assess dev plus ablation. No outcome, family or ID in the policy."))
    best=options[0];a=b=None
    if which=="learned":
        net=Protected(pack);a=compare(out,"fit",fit,options,net)
        leading=options[np.argsort(-a[1:],kind="stable")[:4]+1]
        pool=np.vstack((options[0],leading,leading.mean(0)));b=compare(out,"select",selection,pool,net)
        best=pool[b.argmax()]
    elif which!="fixed":raise ValueError(which)
    dest=out/"model.pt"
    if not dest.exists():
        tmp=dest.with_suffix(".tmp");torch.save(dict(pack,morph=torch.tensor(best,dtype=torch.float64)),tmp);tmp.replace(dest)
    save(out/"frozen.json",dict(model_sha256=digest(dest),coefficients=best.tolist(),fit=a.tolist() if a is not None else None,
              select=b.tolist() if b is not None else None,protocol_sha256=digest(out/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(out,tag,load(dest.read_bytes(),ablate))


if __name__=="__main__":main(sys.argv[1])
