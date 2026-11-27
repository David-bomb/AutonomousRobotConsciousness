"""Обучение на разности жизнеспособности и оценка числа приобретённых и утраченных решений."""
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .model import HERE
from .correction_model import Calibrator


def dataset(spec):
    cols={}
    for n,check in spec["parts"].items():
        path=HERE.parent/"cache/correction"/n
        assert digest(path)==check
        with np.load(path) as f:
            for k in f:cols.setdefault(k,[]).append(f[k])
    return {k:torch.from_numpy(np.concatenate(v)) for k,v in cols.items()}


@torch.inference_mode()
def rank(net,held):
    x=held["x"].float();ok=held["ok"]
    q=torch.cat([net(x[i:i+512],ok[i:i+512]).sigmoid() for i in range(0,len(x),512)]).numpy()
    b=np.arange(len(x));a=held["a"].numpy();viable=held["viable"].numpy()
    v0=viable[b,a]
    z=np.stack((x[:,:,0].numpy()*6,x[:,:,1].numpy(),q[:,:,0],q[:,:,1],q[:,:,2],
               x[:,:,8:10].mean(2).numpy(),x[:,:,5].numpy(),x[:,:,6].numpy()),-1)
    configs=[]
    for offset in (0,-.5,-1):
        for rho in (3,7,14):
            for angle in (0,5,12):
                for prefix,conditional in ((0,0),(5,4),(14,3)):
                    configs.append([1,offset,angle,rho,2,prefix,conditional,6])
    configs.append([1,0,0,0,0,0,0,0])
    result=[]
    for w in configs:
        ch=np.where(ok.numpy(),z@w,-1e9).argmax(1)
        v1=viable[b,ch]
        gain=int((v1&~v0).sum());loss=int((~v1&v0).sum())
        result.append(dict(weights=w,gained=gain,lost=loss,net=gain-loss,changed=int((ch!=a).sum()),
                           downhill=int(held["down"].numpy()[b,ch].sum())))
    result.sort(key=lambda r:(-r["net"],r["lost"],-r["downhill"]))
    return dict(n=len(x),default_irreversible=int((~v0).sum()),candidates=result)


def main():
    spec=read(HERE/"correction-data.json")
    train,held=map(dataset,spec["sets"])
    rng=np.random.default_rng(1621);torch.manual_seed(1621);torch.set_num_threads(1)
    net=Calibrator()
    opt=torch.optim.AdamW(net.parameters(),lr=.0006)
    n=len(train["x"])
    default=train["a"].long()
    danger=np.flatnonzero(~train["viable"][torch.arange(n),default].numpy())
    critical=np.flatnonzero((train["ok"]&~train["viable"]).any(1).numpy())
    folder=HERE/"calibration";folder.mkdir(exist_ok=False)
    cfg=dict(seed=1621,train=digest(HERE/"correction-data.json"),fit_updates=[450,1600],lr=.0006)
    save(folder/"config.json",cfg)
    history=[]
    for step in range(1,1601):
        ids=np.concatenate((rng.integers(n,size=56),rng.choice(critical,48),rng.choice(danger,24)))
        x,ok=train["x"][ids].float(),train["ok"][ids]
        pred=net(x,ok)
        truth=torch.stack([train[k][ids] for k in ("down","viable","best")],-1).float()
        classification=F.binary_cross_entropy_with_logits(pred[ok],truth[ok],reduction="none")
        diff=truth[:,:,None,1]-truth[:,None,:,1]
        pairs=(diff>0)&ok[:,:,None]&ok[:,None,:]
        delta=pred[:,:,None,1]-pred[:,None,:,1]
        loss=(classification*torch.tensor([1.,1.7,.45])).mean()-.4*F.logsigmoid(delta[pairs]).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step()
        if step in (450,1600):
            p=folder/f"s{step}.pt";torch.save(net.state_dict(),p)
            report=rank(net.eval(),held);net.train()
            history.append(dict(step=step,sha256=digest(p),**report))
            print(step,report["n"],report["default_irreversible"],report["candidates"][0],flush=True)
    chosen=max(history,key=lambda r:r["candidates"][0]["net"])
    save(folder/"result.json",dict(selected=chosen["step"],history=history))


if __name__=="__main__":main()
