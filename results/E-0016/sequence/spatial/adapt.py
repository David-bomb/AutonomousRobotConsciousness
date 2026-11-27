"""Калибровка пространственного прогноза на собственных поездках и поездках разведчика."""
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .train import dataset,diagnostics
from .model import HERE,SpatialSensor


@torch.inference_mode()
def errors(net,probe):
    counts=np.zeros((3,4),np.int64)
    for j in range(0,len(probe["y"]),96):
        sl=slice(j,j+96)
        maps,meta,y,viable,orig,s=[probe[k][sl].cuda() for k in ("maps","meta","y","viable","default","slack")]
        g,ok=net(maps,meta)
        b=torch.arange(len(ok),device="cuda")
        k=s.argmin(1)
        a=g.softmax(-1)[b,:,k,0].masked_fill(~ok,-1).argmax(1)
        for h,mask in enumerate((~viable[b,orig],(s.amin(1)<=3)&ok.any(1),ok.any(1))):
            stats=(mask.sum(),(mask&(y[b,a,k]==0)).sum(),(mask&viable[b,a]).sum(),(mask&viable[b,orig]).sum())
            counts[h]+=np.array([int(v) for v in stats])
    return dict(zip(("default_fatal","tight","all"),counts.tolist()))


def main():
    torch.manual_seed(1618);torch.set_num_threads(1)
    rng=np.random.default_rng(1618)
    spatial=HERE/"train-800/s2200.pt"
    net=SpatialSensor().cuda()
    net.load_state_dict(torch.load(spatial,weights_only=True,map_location="cuda"))
    real,probe=map(dataset,read(HERE/"real.json")["sets"])
    actual=dataset(read(HERE/"actual.json"))
    synthetic=dataset(read(HERE/"data-800.json")["sets"][0])
    out=HERE/"adapt-400"
    out.mkdir(exist_ok=False)
    save(out/"config.json",dict(seed=1618,init=digest(spatial),steps=1800,lr=.00018,train_actual=digest(HERE/"actual.json"),
          demos=digest(HERE/"real.json"),choice="fixed 1800, probe only for diagnostics",source=digest(HERE/"adapt.py")))
    fatal=np.flatnonzero(~actual["viable"][torch.arange(len(actual["y"])),actual["default"].long()].numpy())
    tight=np.flatnonzero((actual["slack"].amin(1)<=5).numpy())
    opt=torch.optim.AdamW(net.parameters(),lr=.00018,weight_decay=.01)
    report=[dict(step=0,physical_probe=errors(net.eval(),probe))]
    for step in range(1,1801):
        a=np.concatenate((rng.choice(fatal,10),rng.choice(tight,14),rng.integers(len(actual["y"]),size=12)))
        r=rng.integers(len(real["y"]),size=24)
        s=rng.integers(len(synthetic["y"]),size=12)
        x,meta,y=[torch.cat((actual[k][a],real[k][r],synthetic[k][s])).cuda() for k in ("maps","meta","y")]
        g,ok=net(x,meta)
        ce=F.cross_entropy(g.flatten(0,2),y.long().flatten(),ignore_index=-1,reduction="none").reshape_as(y)
        weights=ok[:,:,None]*torch.tensor([1.,1.,.45],device="cuda")
        action=g.log_softmax(-1)[:,:,:,0].masked_fill(~ok[:,:,None],-1e6).log_softmax(1)
        has=(y==0).any(1)
        ranking=-action.masked_fill(y!=0,-1e6).logsumexp(1)
        loss=(ce*weights).sum()/weights.sum()+.45*ranking[has].mean()
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
        opt.step()
        if step in (600,1800):
            path=out/f"s{step}.pt"
            torch.save(net.state_dict(),path)
            report.append(dict(step=step,sha256=digest(path),physical_probe=errors(net.eval(),probe)))
            net.train()
            print(step,report[-1]["physical_probe"],flush=True)
    save(out/"result.json",dict(training_fatal=len(fatal),report=report))


if __name__=="__main__":
    main()
