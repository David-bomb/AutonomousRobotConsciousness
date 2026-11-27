"""Обучение выбора после обслуживания с сохранением раннего поведения сети."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .common import HERE,SEED,digest,read
from .compare import rank
from .fit import freeze,load_data,loadnet
from .patch_adapter import PatchNet
from .route_search import route_net
from ..critical_policy import CriticalPolicy
from ..rich_policy import RichPolicy

INPUTS=("core","grad","past","possible","legal")


def make_patch(tag):
    path=HERE/"weights"/(tag+".pt")
    old=loadnet(path)
    if not hasattr(old,"route_parent_mix"):old=route_net(path)
    kind="rich" if isinstance(old.base.base,RichPolicy) else "critical" if isinstance(old.base.base,CriticalPolicy) else "plain"
    net=PatchNet(kind)
    state=old.state_dict();weights=state.pop("mix")
    missing,extra=net.load_state_dict(state,strict=False)
    assert not extra and all(k=="mix" or k.startswith("patch_") for k in missing)
    with torch.no_grad():net.mix[:-1].copy_(weights)
    net.requires_grad_(False)
    for layer in (net.patch_features,net.patch_score,net.patch_aux):layer.requires_grad_(True)
    return net


def collect():
    records=defaultdict(list)
    for name,info in read(HERE/"samples/continuation.json")["files"].items():
        file=HERE/"samples/continuation"/(name+".npz");assert digest(file)==info["sha256"]
        with np.load(file) as data:
            for k in data:records[k].append(data[k])
    return {k:np.concatenate(v) for k,v in records.items()}


def main():
    torch.manual_seed(SEED+320);torch.set_num_threads(2)
    parent=max(["wave1-pg","wave1","route-rl"],key=rank)
    net=make_patch(parent)
    data=collect();prior=load_data(["clean","deviate"])
    groups=[np.flatnonzero(data["win"]&(data["source"]==i)) for i in (0,1)]
    assert all(len(g)>0 for g in groups)
    known=np.flatnonzero((prior["phase"]==2)&prior["demo"])
    early=np.flatnonzero(prior["phase"]<2)
    rng=np.random.default_rng(SEED+320)
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.00085,weight_decay=.001)
    logs=[]
    for step in range(1,901):
        ids=np.concatenate([rng.choice(g,32) for g in groups]+[rng.integers(len(data["win"]),size=16)])
        old=np.r_[rng.choice(known,24),rng.choice(early,40)]
        inputs=[torch.from_numpy(np.concatenate((data[k][ids],prior[k][old])))
                for k in INPUTS]
        for i in range(3):inputs[i]=inputs[i].float()
        legal=inputs[-1];basis,_,_,aux=net.augmented(*inputs)
        control=(basis[:,:,:-1]@net.mix[:-1]).masked_fill(~legal,-1e5).detach()
        score=(basis@net.mix).masked_fill(~legal,-1e5);logp=score.log_softmax(1)
        actions=torch.from_numpy(data["action"][ids])
        demo=F.nll_loss(logp[:64],actions[:64])
        best=inputs[0][:64,:,1:41].reshape(64,8,5,8)
        length=best[:,:,:,4].masked_fill((best[:,:,:,6]<.5)|(best[:,:,:,2]<0),8).amin(2).masked_fill(~legal[:64],8)
        low=length.amin(1,keepdim=True)
        target=-(logp[:64].masked_fill(length!=low,-1e5).logsumexp(1))*(low[:,0]<8)
        rehearsal=F.nll_loss(logp[80:104],torch.from_numpy(prior["action"][old[:24]]))
        anchor=F.kl_div(logp[104:],control[104:].softmax(1),reduction="batchmean")
        labels=torch.from_numpy(data["labels"][ids]).float()
        labels_loss=F.binary_cross_entropy_with_logits(aux[:80][legal[:80]],labels[legal[:80]])
        loss=demo+.30*target.mean()+.25*rehearsal+1.2*anchor+.8*labels_loss
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,2);opt.step()
        if step%100==0:
            logs.append([round(float(v.detach()),5) for v in (demo,labels_loss,anchor)])
    freeze(net,"post-teacher",dict(parent=parent,steps=900,seed=SEED+320,
           successful_examples=[len(g) for g in groups],total_examples=len(data["win"]),trace=logs,
           samples_sha256=digest(HERE/"samples/continuation.json")))
    print("post-teacher",parent,len(data["win"]),[len(g) for g in groups],flush=True)


if __name__=="__main__":main()
