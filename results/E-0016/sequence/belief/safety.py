"""Две учебные задачи: изменения штатных длин и физическая достижимость финиша."""
import argparse
import numpy as np
import torch
from .safety_net import Safety
from torch.nn import functional as F
from ...data import digest,read,save
from .features import HERE


def dataset(files):
    parts={}
    for name in files:
        with np.load(HERE.parent/"cache/belief"/name) as f:
            for k in f:
                if k not in ("best","phase"):parts.setdefault(k,[]).append(f[k])
    return {k:torch.from_numpy(np.concatenate(vals)) for k,vals in parts.items()}


def forward(net,d,ids):
    return net(d["x"][ids].float(),d["exit"][ids].float(),d["walk"][ids].float(),d["ok"][ids])


@torch.inference_mode()
def diagnose(net,d):
    n=len(d["x"])
    preds=[forward(net,d,slice(i,i+256)) for i in range(0,n,256)]
    g,rv=(torch.cat(c) for c in zip(*preds))
    ok=d["ok"]
    b=torch.arange(n)
    urgent=d["slack"].argmin(1)
    p=g.softmax(-1)[b,:,urgent,0]
    actual=(d["sign"][b,:,urgent]==0)&ok
    viable=d["viable"]
    default=d["default"].long()
    fatal=~viable[b,default]
    other=(ok&~viable).any(1)&~fatal
    results={}
    for key,score in (("sensor",p),("viability",rv),("combined",p+rv.sigmoid())):
        a=score.masked_fill(~ok,-1e9).argmax(1)
        results[key]=dict(fatal_n=int(fatal.sum()),retained_after_default_fatal=int((fatal&viable[b,a]).sum()),
                          preserved_when_default_viable=int((other&viable[b,a]).sum()),viable_default_n=int(other.sum()),
                          downhill_when_default_fatal=int((fatal&actual[b,a]).sum()))
    return results


def main():
    rng=np.random.default_rng(1616)
    torch.manual_seed(1616)
    torch.set_num_threads(1)
    src=read(HERE/"collect-400.json")
    names=list(src["parts"])
    held=sorted(rng.choice(len(names),8,replace=False))
    tr=dataset([n for j,n in enumerate(names) if j not in held])
    val=dataset([n for j,n in enumerate(names) if j in held])
    n=len(tr["x"])
    costly=np.flatnonzero(~tr["viable"][torch.arange(n),tr["default"].long()].numpy())
    marginal=np.flatnonzero((tr["ok"]&~tr["viable"]).any(1).numpy())
    net=Safety()
    opt=torch.optim.AdamW(net.parameters(),lr=.00065)
    epochs=[]
    for step in range(1,1801):
        ids=np.concatenate((rng.integers(n,size=48),rng.choice(marginal,48),rng.choice(costly,32)))
        g,r=forward(net,tr,ids)
        ok=tr["ok"][ids]
        s=tr["sign"][ids].long()
        out=tr["viable"][ids].float()
        difficulty=1+2*(tr["slack"][ids]<=5)
        error=F.cross_entropy(g.flatten(0,2),s.flatten(),reduction="none").view_as(s)
        weights=ok[:,:,None]*difficulty[:,None,:]
        loss=(error*weights).sum()/weights.sum()+1.6*F.binary_cross_entropy_with_logits(r[ok],out[ok])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
        opt.step()
        if step in (450,1800):
            path=HERE/f"safety-{step}.pt"
            torch.save(net.state_dict(),path)
            result=diagnose(net,val)
            epochs.append(dict(step=step,sha256=digest(path),held_geometry_result=result))
            print(step,result,flush=True)
    # Предпочесть больше сохранённых шансов при равной цене утраты уже жизнеспособного хода.
    def balance(epoch):
        x=epoch["held_geometry_result"]["combined"]
        return x["retained_after_default_fatal"]/x["fatal_n"]+x["preserved_when_default_viable"]/x["viable_default_n"]
    chosen=max(epochs,key=balance)["step"]
    save(HERE/"safety.json",dict(seed=1616,train_maps=320,held_maps=80,training_states=n,
        default_fatal_training=len(costly),held_parts=[names[i] for i in held],source=digest(HERE/"collect-400.json"),
        checkpoints=epochs,selected=chosen))


if __name__=="__main__":
    main()
