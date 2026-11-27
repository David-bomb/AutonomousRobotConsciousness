from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .model import HERE
from .field_model import FieldNet

CACHE=HERE.parent/"cache/spatial"


class Shards:
    def __init__(self,spec,synthetic=False):
        self.names=list(spec["parts"])
        self.synthetic=synthetic
        self.current=None
        self.data={}

    def sample(self,rng,n,refresh):
        if refresh or self.current is None:
            name=rng.choice(self.names)
            path=CACHE/name
            with np.load(path) as f:self.data={k:f[k] for k in f}
            if self.synthetic:
                with np.load(CACHE/("dense-"+name)) as f:self.data["dense"]=f["dense"]
            self.current=name
        data=self.data
        size=len(data["meta"])
        ids=rng.integers(size,size=n)
        return [torch.from_numpy(data[k][ids]).cuda() for k in
               (("maps","meta","y","dense") if self.synthetic else ("maps","meta","y","dense","stocks","last","viable","best"))]


@torch.inference_mode()
def heldout(net,spec):
    hits=np.zeros(3,int);total=0;losses=rescued=sharp=0
    for name in spec["parts"]:
        with np.load(CACHE/name) as f:
            d={k:torch.from_numpy(f[k]).cuda() for k in ("maps","meta","y","stocks","last","viable")}
        for i in range(0,len(d["meta"]),96):
            x,m,y,s,last,v=[z[i:i+96] for z in d.values()]
            q,_,ok=net(x,m,s,last)
            truth=y==0
            actions=q[:,:,:3].masked_fill(~ok[:,:,None],-1e6).argmax(1)
            b=torch.arange(len(q),device=q.device)
            valid=truth.any(1)
            hits+=((truth.gather(1,actions[:,None,:]).squeeze(1))&valid).sum(0).cpu().numpy()
            total+=len(q)
            tight=(ok&~v).any(1)&v.any(1)
            chosen=q[:,:,3].masked_fill(~ok,-1e6).argmax(1)
            losses+=int(tight.sum());rescued+=int((tight&v[b,chosen]).sum())
            sidx=(s.float()-m[:,6:]).argmin(1)
            sharp+=int((tight&truth[b,actions[b,sidx],sidx]).sum())
    return dict(observations=total,downhill=hits.tolist(),critical=losses,critical_viable=rescued,
                critical_urgent_downhill=sharp)


def main(clean=False):
    rng=np.random.default_rng(1620);torch.manual_seed(1620);torch.set_num_threads(1)
    synth=Shards(read(HERE/"extra-data.json"),True)
    if clean:
        synth.names=[name for name in synth.names if int(Path(name).stem.split("-")[1]) < 3000]
        assert len(synth.names)==150
    early,held=read(HERE/"early.json")["sets"]
    actual=Shards(early)
    net=FieldNet().cuda()
    base=HERE/"adapt-400/s1800.pt"
    state=torch.load(base,weights_only=True,map_location="cuda")
    trunk={k:v for k,v in state.items() if k.startswith(("local.","mid.","wide."))}
    net.load_state_dict(trunk,strict=False)
    out=HERE/("field-3000-independent" if clean else "field-3000")
    out.mkdir(exist_ok=False)
    save(out/"config.json",dict(seed=1620,stages=[1400,3500],lr=.00065,init=digest(base),
            early_sha256=digest(HERE/"early.json"),synthetic_sha256=digest(HERE/"dense.json"),
            synthetic_parts=synth.names, held_geometry_range=[3000,3200],
            sources={n:digest(HERE/n) for n in ("field_model.py","field_train.py")}))
    opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.01)
    checkpoints=[]
    for step in range(1,3501):
        f=step%15==1
        A=actual.sample(rng,48,f)
        S=synth.sample(rng,24,f)
        x,m,y,truth=[torch.cat((a,b)) for a,b in zip(A[:4],S)]
        s=torch.cat((A[4],(S[1][:,6:]+12).clamp_max(100)))
        last=torch.cat((A[5],torch.full((24,),8,device=x.device)))
        q,p,ok=net(x,m,s,last)
        if step==1:assert torch.equal(ok,y[:,:,0]>=0)
        mask=((truth[:,None].int()>>torch.arange(4,device=x.device)[None,:,None,None])&1).float()
        loss_map=F.binary_cross_entropy_with_logits(p,mask,pos_weight=torch.tensor([12.,12.,12.,1.],device=x.device)[:,None,None])
        down=(y==0).float()
        direct=F.binary_cross_entropy_with_logits(q[:,:,:3][ok],down[ok])
        viable=F.binary_cross_entropy_with_logits(q[:48,:,3][ok[:48]],A[6][ok[:48]].float())
        best=F.binary_cross_entropy_with_logits(q[:48,:,4][ok[:48]],A[7][ok[:48]].float())
        loss=.7*loss_map+direct+viable+.5*best
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
        opt.step()
        if step in (1400,3500):
            path=out/f"s{step}.pt"
            torch.save(net.state_dict(),path)
            score=heldout(net.eval(),held)
            checkpoints.append(dict(step=step,sha256=digest(path),score=score));net.train()
            print(step,score,flush=True)
    chosen=max(checkpoints,key=lambda u:u["score"]["critical_viable"]+u["score"]["critical_urgent_downhill"])["step"]
    save(out/"results.json",dict(checkpoints=checkpoints,selected=chosen))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--independent",action="store_true")
    main(parser.parse_args().independent)
