from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .local_model import LocalNet
from .local_data import crop,HERE


class Stream:
    def __init__(self,names,kind):
        self.names=list(names);self.kind=kind;self.data=None

    def sample(self,rng,n,refresh):
        if self.data is None or refresh:
            folder="local" if self.kind=="synthetic" else "spatial"
            p=HERE.parent/"cache"/folder/rng.choice(self.names)
            with np.load(p) as f:
                data={k:f[k] for k in f if k not in ("dense","maps")}
                if "patch" not in data:data["patch"]=crop(f["maps"],data["meta"])
            m=len(data["meta"])
            if "stocks" not in data:
                data["stocks"]=data["meta"][:,6:]+(data["slack"] if "slack" in data else 15)
                data["last"]=np.full(m,8,np.int8)
            if "viable" not in data:data["viable"]=np.zeros((m,8),bool)
            if "best" not in data:data["best"]=np.zeros((m,8),bool)
            if "default" in data:
                data["fatal"]=np.flatnonzero(~data["viable"][np.arange(m),data["default"]])
            self.data=data
        d=self.data
        ids=rng.integers(len(d["meta"]),size=n)
        if self.kind=="actual" and len(d["fatal"]):
            ids[:max(1,n//3)]=rng.choice(d["fatal"],size=max(1,n//3))
        return [d[k][ids] for k in ("patch","meta","y","stocks","last","viable","best")]


@torch.inference_mode()
def audit(net,spec):
    total=critical=retained=urgent_down=0
    for name in spec["parts"]:
        with np.load(HERE.parent/"cache/spatial"/name) as f:
            m=f["meta"];p=crop(f["maps"],m)
            data=[torch.from_numpy(z).cuda() for z in (p,m,f["y"],f["stocks"],f["last"],f["viable"])]
        for i in range(0,len(m),128):
            x,m0,y,s,last,v=[z[i:i+128] for z in data]
            g,rv,ok=net(x,m0,s,last)
            assert torch.equal(ok,y[:,:,0]>=0)
            b=torch.arange(len(ok),device=ok.device)
            k=(s.float()-m0[:,6:]).argmin(1)
            hard=(ok&~v).any(1)
            pick=rv[:,:,0].masked_fill(~ok,-1e5).argmax(1)
            down=g.softmax(-1)[b,:,k,0].masked_fill(~ok,-1e5).argmax(1)
            total+=len(ok);critical+=int(hard.sum())
            retained+=int((hard&v[b,pick]).sum());urgent_down+=int((hard&(y[b,down,k]==0)).sum())
    return dict(states=total,critical=critical,critical_viable=retained,critical_downhill=urgent_down)


def main():
    rng=np.random.default_rng(1622);torch.manual_seed(1622);torch.set_num_threads(1)
    src=read(HERE/"local-data.json")
    early,held=read(HERE/"early.json")["sets"]
    streams=[Stream(src["parts"],"synthetic"),Stream(early["parts"],"early"),Stream(read(HERE/"actual.json")["parts"],"actual")]
    net=LocalNet().cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.01)
    out=HERE/"local-12000";out.mkdir(exist_ok=False)
    save(out/"config.json",dict(seed=1622,learning_rate=.00065,steps=[2500,6500],
         source=digest(HERE/"local_train.py"),network=digest(HERE/"local_model.py"),
         inputs={n:digest(HERE/n) for n in ("actual.json","early.json","local-data.json")}))
    history=[]
    for step in range(1,6501):
        inputs=[stream.sample(rng,n,step%17==1) for stream,n in zip(streams,(48,44,20))]
        x,m,y,s,last,v,best=[torch.from_numpy(np.concatenate(col)).cuda() for col in zip(*inputs)]
        if step%2==0:
            m=m[:,[0,3,4,1,2,5,7,6]];y=y[:,:,[1,0,2]];s=s[:,[1,0]]
        g,rv,ok=net(x,m,s,last)
        if step==1:assert torch.equal(ok,y[:,:,0]>=0)
        count=ok[:,:,None]*torch.tensor([1.,1.,.4],device=x.device)
        error=F.cross_entropy(g.flatten(0,2),y.long().flatten(),ignore_index=-1,reduction="none").reshape_as(y)
        sig_loss=(error*count).sum()/count.sum()
        action=g.log_softmax(-1)[:,:,:,0].masked_fill(~ok[:,:,None],-1e6).log_softmax(1)
        good=y==0
        select=-action.masked_fill(~good,-1e6).logsumexp(1)
        safety=F.binary_cross_entropy_with_logits(rv[48:,:,0][ok[48:]],v[48:][ok[48:]].float())
        fastest=F.binary_cross_entropy_with_logits(rv[48:92,:,1][ok[48:92]],best[48:92][ok[48:92]].float())
        loss=sig_loss+.3*select[good.any(1)].mean()+safety+.4*fastest
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);opt.step()
        if step in (2500,6500):
            path=out/f"s{step}.pt";torch.save(net.state_dict(),path)
            stats=audit(net.eval(),held);net.train()
            history.append(dict(step=step,sha256=digest(path),score=stats))
            print(step,stats,flush=True)
    chosen=max(history,key=lambda z:z["score"]["critical_viable"]+z["score"]["critical_downhill"])["step"]
    save(out/"results.json",dict(history=history,selected=chosen))


if __name__=="__main__":main()
