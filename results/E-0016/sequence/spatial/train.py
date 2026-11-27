"""Обучение предсказателя соседних показаний на неполных учебных картах."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .model import HERE,SpatialSensor


def dataset(spec):
    cols={}
    for name,sig in spec["parts"].items():
        path=HERE.parent/"cache/spatial"/name
        assert digest(path)==sig
        with np.load(path) as part:
            for k in part:cols.setdefault(k,[]).append(part[k])
    return {k:torch.from_numpy(np.concatenate(vals)) for k,vals in cols.items()}


@torch.inference_mode()
def diagnostics(net,data):
    correct=np.zeros(3,int);n=0;down=np.zeros(3,int);exists=np.zeros(3,int)
    for i in range(0,len(data["maps"]),96):
        sl=slice(i,i+96)
        x,meta,y=[data[k][sl].cuda() for k in ("maps","meta","y")]
        pred,ok=net(x,meta)
        assert torch.equal(ok,y[:,:,0]>=0)
        correct+=((pred.argmax(-1)==y)&ok[:,:,None]).sum((0,1)).cpu().numpy()
        n+=int(ok.sum())
        valid=((y==0)&ok[:,:,None]).any(1)&(ok.sum(1)[:,None]>1)
        a=pred.softmax(-1)[:,:,:,0].masked_fill(~ok[:,:,None],-1).argmax(1)
        choose=y.gather(1,a[:,None,:]).squeeze(1)==0
        down+=(choose&valid).sum(0).cpu().numpy();exists+=valid.sum(0).cpu().numpy()
    return dict(labels_per_target=n,correct=correct.tolist(),downhill=down.tolist(),eligible=exists.tolist())


def main(n,steps):
    rng=np.random.default_rng(1617)
    torch.manual_seed(1617)
    torch.set_num_threads(1)
    specs=read(HERE/f"data-{n}.json")["sets"]
    net=SpatialSensor().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.01)
    tr,probe=map(dataset,specs)
    run=HERE/f"train-{n}"
    run.mkdir(exist_ok=False)
    save(run/"config.json",dict(seed=1617,steps=steps,lr=.0008,batch=72,data_sha256=digest(HERE/f"data-{n}.json"),
                               source_sha256={p:digest(HERE/p) for p in ("model.py","train.py","region.cpp","region.dll")}))
    history=[]
    for step in range(1,steps+1):
        ids=rng.integers(len(tr["maps"]),size=72)
        x,meta,y=[tr[k][ids].cuda() for k in ("maps","meta","y")]
        pred,ok=net(x,meta)
        if step==1:assert torch.equal(ok,y[:,:,0]>=0)
        error=F.cross_entropy(pred.flatten(0,2),y.long().flatten(),ignore_index=-1,reduction="none").reshape_as(y)
        weights=ok[:,:,None]*torch.tensor([1.,1.,.4],device=x.device)
        loss=(error*weights).sum()/weights.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
        opt.step()
        if step in (steps//2,steps):
            path=run/f"s{step}.pt"
            torch.save(net.state_dict(),path)
            result=diagnostics(net.eval(),probe)
            net.train()
            history.append(dict(step=step,sha256=digest(path),probe=result))
            print(step,result,flush=True)
    selected=max(history,key=lambda r:sum(r["probe"]["downhill"][:2]))
    save(run/"result.json",dict(history=history,selected=selected["step"]))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("n",type=int)
    p.add_argument("steps",type=int)
    a=p.parse_args();main(a.n,a.steps)
