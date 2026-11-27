"""Подражание демонстрациям полного старта и корректирующим продолжениям."""
import os
for key in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from datetime import datetime, timezone
import sys
import numpy as np
import torch
from torch.nn import functional as F

from .adapter import Explorer
from .bank import HERE, SEED, digest, read, save
from .evaluate import evaluate

ADAPTER=("adapter.py","features.py","fronts.cpp","fronts.dll")


def load_samples(kind):
    buckets={k:[] for k in ("edges","tokens","legal","actions","weights")}
    manifest=read(HERE/(kind+"-samples.json"))
    for name,sha in manifest["files"].items():
        path=HERE/f"{kind}-samples"/(name+".npz")
        assert digest(path)==sha
        with np.load(path) as data:
            if not len(data["actions"]):
                continue
            for k in ("edges","tokens","legal","actions"):
                buckets[k].append(data[k])
            degree=data["legal"].sum(1)
            weights=(1+.5*(data["phase"]==2))*(1+.5*float(data["win"]))
            weights*=np.where(degree>=3,2.,np.where(degree==1,.15,1.))
            buckets["weights"].append(weights)
    arrays={k:np.concatenate(parts) for k,parts in buckets.items()}
    assert len(arrays["actions"])==manifest["records"] and np.isfinite(arrays["weights"]).all()
    arrays["weights"]/=arrays["weights"].sum()
    arrays["weights"]=arrays["weights"].astype(float)
    return arrays


def checkpoint(tag,net,history,sources):
    folder=HERE/"weights"
    folder.mkdir(exist_ok=True)
    weight=folder/(tag+".pt")
    meta=folder/(tag+".json")
    if meta.exists():
        prior=read(meta)
        assert prior["sha256"]==digest(weight)
        state=torch.load(weight,weights_only=True,map_location="cpu")
        assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items()),tag
    else:
        tmp=folder/(tag+".pending")
        torch.save(net.state_dict(),tmp)
        tmp.replace(weight)
        save(meta,dict(tag=tag,sha256=digest(weight),frozen_utc=datetime.now(timezone.utc).isoformat(),
                       adapter={f:digest(HERE/f) for f in ADAPTER},sources=sources,history=history,
                       network_parameters=sum(p.numel() for p in net.parameters())))
    torch.set_num_threads(1)
    result=evaluate(tag)
    torch.set_num_threads(2)
    return result


def best(tags):
    def rank(tag):
        s=read(HERE/"dev"/(tag+".json"))["summary"]
        return s["wins"],min(s["tiers"].values())
    return max(tags,key=rank)


def train(phase):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED+(phase=="correction"))
    bank=load_samples("teacher")
    net=Explorer(bank["edges"].shape[-1],bank["tokens"].shape[-1])
    history=[]
    sources={p:digest(HERE/p) for p in ("teacher.json","teacher-samples.json","dataset.py","train.py",
                                         "evaluate.py","registration.json")}
    if phase=="teacher":
        checkpoint("random",net,history,sources)
        sets=[bank]
        epochs=32
    else:
        start=best(["imitation12","imitation32"])
        weight=HERE/"weights"/(start+".pt")
        net.load_state_dict(torch.load(weight,weights_only=True,map_location="cpu"))
        sources.update(parent=start,parent_sha256=digest(weight),
                       correction_data=digest(HERE/"correction-samples.json"))
        sets=[bank,load_samples("correction")]
        epochs=16
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.0001)
    for epoch in range(1,epochs+1):
        lr=(.0006 if phase=="correction" else .0012 if epoch<=12 else .00065 if epoch<=24 else .0004)
        for group in optimizer.param_groups:
            group["lr"]=lr
        count=min(8192,sum(len(b["actions"]) for b in sets))
        draws=[rng.choice(len(b["actions"]),size=count//len(sets),p=b["weights"]) for b in sets]
        batch_size=128//len(sets)
        losses=[]
        correct=total=0
        net.train()
        for i in range(0,len(draws[0]),batch_size):
            sample={k:torch.from_numpy(np.concatenate([b[k][ids[i:i+batch_size]] for b,ids in zip(sets,draws)]))
                    for k in ("edges","tokens","legal","actions")}
            pred=net(sample["edges"].float(),sample["tokens"].float(),sample["legal"])
            loss=F.cross_entropy(pred,sample["actions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            optimizer.step()
            losses.append(loss.detach().item())
            correct+=int((pred.argmax(1)==sample["actions"]).sum())
            total+=len(pred)
        assert np.isfinite(losses).all()
        history.append(dict(epoch=epoch,updates=len(losses),examples=total,loss=float(np.mean(losses)),
                            sampled_accuracy=correct/total,lr=lr))
        if epoch%4==0:
            print(phase,"epoch",epoch,round(correct/total,4),flush=True)
        if phase=="teacher" and epoch in (12,32):
            checkpoint(f"imitation{epoch}",net,history,sources)
        if phase=="correction" and epoch==16:
            checkpoint("corrected16",net,history,sources)
    if phase=="correction":
        winner=best(["imitation12","imitation32","corrected16"])
        torch.set_num_threads(1)
        evaluate(winner,repeat=True)
        save(HERE/"selected.json",dict(tag=winner,weights=f"weights/{winner}.pt",
                                      sha256=digest(HERE/"weights"/(winner+".pt")),
                                      result=f"dev/{winner}.json",registration=digest(HERE/"registration.json")))


if __name__=="__main__":
    assert sys.argv[1] in ("teacher","correction")
    train(sys.argv[1])
