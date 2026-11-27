"""Fit counterfactual outcome predictions, then improve the policy by terminal reward."""
import numpy as np
import torch
import torch.nn.functional as F
from renewable_train import EXP,HERE,data,digest,save
from early_trials import compare,physical
from rollout_policy import BranchHead,BranchNet,load

OUT=HERE/"rollout-search"


def read(names):
    columns=[[],[],[],[]]
    for name in names:
        path=OUT/"branches"/(name+".npz")
        rec=data.read(path.with_suffix(".json"))
        assert rec["features"]==digest(path)
        if not rec["samples"]:continue
        with np.load(path) as archive:
            for c,k in zip(columns,("x","y","legal","parent")):c.append(archive[k])
    return tuple(torch.from_numpy(np.concatenate(col)).to(dtype) for col,dtype in
                 zip(columns,(torch.float32,torch.float32,torch.bool,torch.long)))


def losses(z,y,legal):
    error=F.binary_cross_entropy_with_logits(z,y,reduction="none")
    err=(error*legal[:,:,None]).sum(1)/legal.sum(1,keepdim=True)
    # The comparison stays within the same observed fork and the same environment.
    diff=z[:,:,None,0]-z[:,None,:,0]
    pairs=(y[:,:,None,0]>y[:,None,:,0]) & legal[:,:,None] & legal[:,None,:]
    rank=(F.softplus(-diff)*pairs).sum((1,2))/pairs.sum((1,2)).clamp(min=1)
    return err[:,0]+.45*err[:,1]+.30*err[:,2]+.65*rank


def learn_head(entries,config):
    x,y,legal,_=read(entries["fit"]);hx,hy,hm,hp=read(entries["held"])
    torch.manual_seed(497316)
    head=BranchHead(x.shape[2])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0018,weight_decay=.012)
    rng=np.random.default_rng(497316)
    best=float("inf");state=None;stats=[]
    for iteration in range(1,501):
        indices=rng.integers(len(x),size=96)
        q=head(x[indices],legal[indices])
        loss=losses(q,y[indices],legal[indices]).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if iteration in (100,240,500):
            with torch.inference_mode():
                vals=head(hx,hm)
                loss=float(losses(vals,hy,hm).mean())
                pick=vals[:,:,0].masked_fill(~hm,-100.).argmax(1)
                actual=hy[torch.arange(len(hx)),pick,0];base=hy[torch.arange(len(hx)),hp,0]
                stats.append(dict(iteration=iteration,loss=loss,samples=len(hx),gains=int(((actual>0)&(base==0)).sum()),
                                  losses=int(((base>0)&(actual==0)).sum()),base=int(base.sum()),alternative=int(actual.sum())))
                if loss<best:best=loss;state={k:v.clone() for k,v in head.state_dict().items()}
    path=OUT/"outcome-head.pt"
    if not path.exists():torch.save(dict(head=state,width=x.shape[2]),path)
    save(OUT/"outcome-fit.json",dict(train_forks=len(x),held_forks=len(hx),checkpoints=stats,weights=digest(path),protocol=digest(config)))
    return torch.load(path,weights_only=True,map_location="cpu")


def main():
    torch.set_num_threads(1)
    parent=HERE/"service-window/fixed/model.pt";bank=EXP/"sequence/spatial/large.jsonl"
    corpus=data.rows(bank)
    entries=data.read(OUT/"collection.json")
    fit=corpus[3200:3260];held=corpus[5100:5200]
    assert not (({r["name"] for r in fit+held})&set(entries["fit"]+entries["held"]))
    choices=np.array([[0]*7,[4,1,1,0,0,0,-.3],[7,2,0,0,0,0,-.5],[0,4,2,0,0,0,-.3],
        [3,0,0,5,1,1,-.4],[0,0,0,7,2,1,-.7],[0,0,0,12,0,0,-1],[0,0,0,2,4,3,-.5],
        [4,1,1,6,0,2,-1.5],[2,0,0,3,2,1,-1],[2,2,0,4,1,0,0]],float)
    config=OUT/"protocol.json"
    save(config,dict(base=digest(parent),collection=digest(OUT/"collection.json"),fit=[r["name"] for r in fit],
        select=[r["name"] for r in held],options=choices.tolist(),seed=497316,
        sources={n:digest(HERE/n) for n in ("rollout_train.py","rollout_policy.py","early_trials.py","tree_paths.dll",
                                             "geodesic_evidence.dll")},
        criterion="Shared action MLP predicts autonomous terminal SUCCESS, reached gate and true urgent descent after each real intervention. Checkpoints 100/240/500 selected by held average weighted BCE and within-fork paired ranking. Then compare eleven combinations by true complete returns on 60 training starts, four leaders and their mean versus base on a fresh training hundred. Freeze, reload and physically evaluate on dev with and without the previous readings."))
    component=learn_head(entries,config)
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="early-rollout-return-1",**component)
    first=compare(OUT,"fit",fit,choices,BranchNet(pack))
    top=choices[np.argsort(-first[1:],kind="stable")[:4]+1]
    pool=np.vstack((choices[0],top,top.mean(0)));second=compare(OUT,"select",held,pool,BranchNet(pack))
    theta=pool[second.argmax()];p=OUT/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp");torch.save(dict(pack,residual=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(p)
    save(OUT/"frozen.json",dict(model_sha256=digest(p),coefficients=theta.tolist(),fit=first.tolist(),select=second.tolist(),
                               protocol_sha256=digest(config)))
    for label,ablate in (("dev",False),("ablated",True)):
        physical(OUT,label,load(p.read_bytes(),ablate))


if __name__=="__main__":main()
