"""Обучение по реальным знакам датчиков, жизнеспособности и учебному поиску выхода."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from datetime import datetime,timezone
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .adapter import PrevNet
from .common import BASE,BASE_SHA,HERE,SEED,digest,read,write


def loadnet(path=None):
    state=torch.load(BASE,weights_only=True,map_location="cpu")
    state2=torch.load(path,weights_only=True,map_location="cpu") if path else None
    layer=PrevNet
    if state2 is not None and "patch_score.0.weight" in state2:
        from .patch_adapter import PatchNet
        layer=PatchNet
    elif state2 is not None and "route_parent_mix" in state2:
        from .route_adapter import RouteNet
        layer=RouteNet
    net=layer("rich" if "base.wall_prior" in state else "critical" if "base.risk.0.weight" in state else "plain")
    net.base.load_state_dict(state)
    if path: net.load_state_dict(state2)
    return net


def load_data(groups):
    parts={}
    for kind in groups:
        meta=read(HERE/"samples"/(kind+".json"))
        for name,sha in meta["files"].items():
            path=HERE/"samples"/kind/(name+".npz")
            assert digest(path)==sha
            with np.load(path) as data:
                for k in data.files:
                    if k!="parent":parts.setdefault(k,[]).append(data[k])
    return {k:np.concatenate(p) for k,p in parts.items()}


def get_batch(data,ids):
    batch={k:torch.from_numpy(v[ids]) for k,v in data.items()}
    for k in ("core","grad","past","missing"): batch[k]=batch[k].float()
    return batch


def instruments(data):
    # Образцы доступны только для физических действий. Возможность возврата помечена лишь на train.
    legal=data["legal"]
    p=data["past"][np.arange(len(legal)),legal.argmax(1)]
    urgent=np.where(p[:,:,11]>0,p[:,:,12],10).argmin(1)
    r=p[np.arange(len(p)),urgent]
    tight=(r[:,13]>.5)&(r[:,11]>0)
    back=data["back"]
    rescue=tight&(r[:,3]>.5)&(back>=0)&data["viable"][np.arange(len(p)),back.clip(0)]
    danger=(~data["viable"]&legal).any(1)&data["viable"].any(1)
    return [np.flatnonzero(g) for g in (tight,rescue,(data["phase"]==2)&data["demo"],danger)]


def loss_for(net,b,idx,drop,sign_only=False):
    past=torch.where(drop[:,None,None,None],b["missing"],b["past"])
    possible=torch.where(drop[:,None,None,None],b["abounds"],b["possible"])
    scores,signs,risk=net.heads(b["core"],b["grad"],past,possible,b["legal"])
    obs=signs[idx,b["action"]]
    sign_loss=F.cross_entropy(obs.flatten(0,1),b["signs"].flatten())
    if sign_only:return sign_loss
    logp=scores.log_softmax(1)
    label_set=lambda mask:-logp.masked_fill(~mask,-1e4).logsumexp(1)
    best=b["best"]; has=best.any(1); active=b["legal"]
    imitation=label_set(best)
    guide=-logp[idx,b["action"]]
    w=torch.where(b["phase"]==2,.78,.45)*b["demo"]
    imitation=(1-w)*imitation+w*guide
    known=b["core"][:,:,1:41].reshape(len(idx),8,5,8)
    cost=known[:,:,:,4].masked_fill((known[:,:,:,6]<.5)|(known[:,:,:,2]<0),8).amin(2).masked_fill(~active,8)
    low=cost.amin(1,keepdim=True)
    imitation=torch.where(low[:,0]<8,.85*label_set((cost==low)&active)+.15*imitation,imitation)
    ref=b["past"][idx,active.long().argmax(1)]
    urgent=ref[:,:,12].masked_fill(ref[:,:,11]<=0,10).argmin(1)
    critical=(ref[idx,urgent,13]>.5)&(ref[idx,urgent,3]>.5)&(ref[idx,urgent,11]>0)
    back=b["back"].clamp_min(0)
    rescue=critical&(b["back"]>=0)&b["viable"][idx,back]&~drop&(low[:,0]>=8)
    imitation=torch.where(rescue,.8*(-logp[idx,back])+.2*imitation,imitation)
    risk_loss=F.binary_cross_entropy_with_logits(risk[active],b["viable"][active].float())
    return imitation[has].mean()+.55*risk_loss+1.2*sign_loss


@torch.inference_mode()
def diagnose(net,data):
    result={}
    for mode in ("prior","masked","honest"):
        totals=correct=tight_n=tight_ok=0
        for i in range(0,len(data["action"]),128):
            b=get_batch(data,slice(i,i+128)); idx=torch.arange(len(b["action"]))
            grad=b["grad"]
            if mode!="prior":
                p=b["missing"] if mode=="masked" else b["past"]
                bounds=b["abounds"] if mode=="masked" else b["possible"]
                grad=net.heads(b["core"],grad,p,bounds,b["legal"])[1]
            hit=grad[idx,b["action"]].argmax(-1)==b["signs"]
            urgent=b["past"][idx,b["action"],:,13]>.5
            totals+=hit.numel();correct+=int(hit.sum())
            tight_n+=int(urgent.sum());tight_ok+=int((hit&urgent).sum())
        result[mode]=dict(correct=correct,n=totals,tight_correct=tight_ok,tight_n=tight_n)
    return result


def freeze(net,tag,details):
    folder=HERE/"weights";folder.mkdir(exist_ok=True)
    path=folder/(tag+".pt"); info=path.with_suffix(".json")
    if info.exists():
        old=read(info)
        assert old["sha256"]==digest(path)
        state=torch.load(path,weights_only=True,map_location="cpu")
        assert all(torch.equal(v,state[k]) for k,v in net.state_dict().items()),tag
        return
    temp=path.with_suffix(".pending")
    torch.save(net.state_dict(),temp);temp.replace(path)
    write(info,dict(tag=tag,sha256=digest(path),base_sha256=BASE_SHA,utc=datetime.now(timezone.utc).isoformat(),
          adapter_sha256=digest(HERE/"adapter.py"),registration=digest(HERE/"registration.json"),training=details,
          route_adapter_sha256=digest(HERE/"route_adapter.py") if hasattr(net,"route_parent_mix") else None,
          patch_adapter_sha256=digest(HERE/"patch_adapter.py") if hasattr(net,"patch_score") else None))


def fit(phase,parent=None):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    net=loadnet(HERE/"weights"/(parent+".pt") if parent else None)
    data=load_data(["clean","deviate"]+(["extra"] if phase=="extra" else []))
    probe=load_data(["probe"])
    subsets=instruments(data)
    rng=np.random.default_rng(SEED+int(bool(parent)))
    print("train examples",len(data["action"]),"strata",[len(x) for x in subsets],flush=True)
    net.base.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.00072 if not parent else .00038)
    logs=[]
    sigsteps=250 if not parent else 100
    end=1500 if not parent else 1300
    for step in range(1,end+1):
        ids=np.concatenate([rng.integers(len(data["action"]),size=48),
                            *[rng.choice(pool,size=n) for pool,n in zip(subsets,(20,20,24,16))]])
        b=get_batch(data,ids)
        loss=loss_for(net,b,torch.arange(len(ids)),torch.from_numpy(rng.random(len(ids))<.12),step<=sigsteps)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if step%100==0:logs.append(float(loss.detach()))
        if step in (600,end):
            tag=("extra" if parent else "super")+str(step)
            metrics=diagnose(net.eval(),probe)
            freeze(net,tag,dict(parent=parent or "mission600",steps=step,sign_only=sigsteps,
                                probe=metrics,loss=logs.copy(),sign_train=int(len(data["action"])),
                                groups=["clean","deviate"]+(["extra"] if parent else [])))
            print(tag,metrics,flush=True)


if __name__=="__main__":
    fit(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
