import numpy as np
import torch
from torch.nn import functional as F
from ....data import read,save,digest
from ..gate_model import GateNet
from .cells import HERE
from .model import Family,Action,gate_channels


def dataset(net,group):
    data={}
    for n in group["parts"]:
        with np.load(HERE.parent.parent/"cache/rooms"/n) as f:
            for k in ("state","legal","down","safe","best","family","tier","first","slack","last","stock","meta"):
                data.setdefault(k,[]).append(torch.from_numpy(f[k]))
            for k in ("old","room"):
                g=torch.from_numpy(f[k]).cuda()
                data.setdefault(k,[]).append(torch.cat([gate_channels(net,g[i:i+128]) for i in range(0,len(g),128)]).cpu())
    return {k:torch.cat(v).cuda() for k,v in data.items()}


def action_outputs(action,family,d,idx,room):
    old=d["old"][idx];state=d["state"][idx]
    probability=family(state).softmax(-1).detach()
    return action(old,d["room"][idx] if room else old,state,probability,d["stock"][idx],d["meta"][idx],d["last"][idx],d["legal"][idx])


@torch.inference_mode()
def score(family,action,d):
    f=family(d["state"]).softmax(-1)
    truth=d["family"]
    matrix=torch.bincount(4*truth+f.argmax(1),minlength=16).reshape(4,4).tolist()
    result=dict(states=len(truth),confusion=matrix,room_brier=float(((f[:,3]-(truth==3).float())**2).mean()),groups={})
    b=torch.arange(len(truth),device=truth.device);legal=d["legal"]
    critical=(legal&~d["safe"]).any(1)
    labels=torch.stack((d["down"],d["safe"],d["best"]),2)
    for hypothesis in (False,True):
        q=action_outputs(action,family,d,b,hypothesis)
        picks=q.masked_fill(~legal[:,:,None],-1e5).argmax(1)
        hits=labels.gather(1,picks[:,None,:])[:,0]
        for name,subset in (("all",critical),("slack01",critical&(d["slack"]<=1)),
                ("rooms",critical&(truth==3)),("rooms_H4",critical&(truth==3)&(d["tier"]==4)),
                ("loops_H4",critical&(truth==2)&(d["tier"]==4))):
            dest=result["groups"].setdefault(name,dict(critical=int(subset.sum())))
            dest["with_room" if hypothesis else "ablated"]=hits[subset].sum(0).tolist()
    return result


def main():
    torch.set_num_threads(1);torch.manual_seed(1633);rng=np.random.default_rng(1633)
    spec=read(HERE/"data.json")
    stage=read(HERE.parent/"gate-3000/result.json")["selected"]
    gate=GateNet().cuda().eval().requires_grad_(False)
    gate_file=HERE.parent/f"gate-3000/s{stage}.pt"
    gate.load_state_dict(torch.load(gate_file,weights_only=True,map_location="cuda"))
    train,held=[dataset(gate,g) for g in spec["groups"]]
    family=Family().cuda();action=Action().cuda()
    fopt=torch.optim.AdamW(family.parameters(),lr=.001);opt=torch.optim.AdamW(action.parameters(),lr=.0008)
    n=len(train["state"]);hard=np.flatnonzero((train["legal"]&~train["safe"]).any(1).cpu().numpy())
    for i in range(800):
        ix=rng.integers(n,size=256)
        loss=F.cross_entropy(family(train["state"][ix]),train["family"][ix].long())
        fopt.zero_grad();loss.backward();fopt.step()
    family.eval()
    for i in range(1200):
        ix=np.r_[rng.integers(n,size=144),rng.choice(hard,96)]
        # Снятие гипотезы на части примеров обучает сопоставимую абляцию того же выхода.
        q=action_outputs(action,family,train,ix,i%4!=0);ok=train["legal"][ix]
        label=torch.stack([train[k][ix] for k in ("down","safe","best")],2)&ok[:,:,None]
        logp=q.masked_fill(~ok[:,:,None],-1e5).log_softmax(1)
        rank=-logp.masked_fill(~label,-1e5).logsumexp(1)[label.any(1)].mean()
        loss=F.binary_cross_entropy_with_logits(q[ok],label[ok].float())+.35*rank
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(action.parameters(),1.5);opt.step()
    result=score(family,action.eval(),held)
    weights=HERE/"supervised.pt"
    torch.save(dict(family=family.cpu().state_dict(),action=action.cpu().state_dict(),gate=gate.cpu().state_dict(),
               counts=torch.from_numpy(np.load(HERE/"counts.npy"))),weights)
    save(HERE/"held.json",dict(seed=1633,updates=[800,1200],weights=digest(weights),geometry_split=[3000,200],
        gate=digest(gate_file),bank=digest(HERE/"data.json"),code=digest(HERE/"fit.py"),score=result))
    print(result,flush=True)


if __name__=="__main__":main()
