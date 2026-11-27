import argparse
import numpy as np
import torch
from torch.nn import functional as F
from ...data import read,save,digest
from .landmark_model import LandmarkNet,true_routes
from .landmarks import HERE

CACHE=HERE.parent/"cache/landmarks"
KEYS=("pair","act","meta","stock","last","length","sign","safe","best")


def arrays(name):
    with np.load(CACHE/name) as data:
        rows=[data[n] for n in KEYS]
        bad=np.flatnonzero(~rows[7][np.arange(len(rows[7])),data["default"]]) if "default" in data else []
    return rows,bad


def target(distance,sign,meta,stock):
    future=distance[:,None,0,1:]+sign-1
    nearest=torch.stack((future[:,:,:2].amin(2),future[:,:,2:4].amin(2)),2)
    urgent=(stock-meta[:,6:]).argmin(1)
    b=torch.arange(len(future),device=stock.device)
    return nearest[b,:,urgent]<meta[b,6+urgent,None]


@torch.inference_mode()
def audit(net,spec):
    n=hard=retain=closer=0;length_err=0.;pairs=0
    for file in spec["parts"]:
        rows,_=arrays(file)
        for i in range(0,len(rows[0]),128):
            p,a,m,s,last,d,sign,v,_=[torch.from_numpy(t[i:i+128]).cuda() for t in rows]
            q,_,ls,_,ok=net(p.float(),a.float(),m,s,last)
            b=torch.arange(len(q),device=q.device)
            critical=(ok & ~v).any(1)
            aa=q[:,:,0].masked_fill(~ok,-1e5).argmax(1)
            ab=q[:,:,2].masked_fill(~ok,-1e5).argmax(1)
            n+=len(q);hard+=int(critical.sum())
            retain+=int((v[b,aa]&critical).sum())
            closer+=int((target(d,sign,m,s)[b,ab]&critical).sum())
            predicted=(ls.softmax(-1)*net.lengths).sum(-1)
            unknown=(p[...,1]>p[...,0]) & ~torch.eye(6,device=q.device,dtype=torch.bool)[None]
            length_err+=float((predicted-d.clamp_max(127)).abs()[unknown].sum())
            pairs+=int(unknown.sum())
    return dict(states=n,critical=hard,critical_viable=retain,critical_downhill=closer,
                uncertain_pairs=pairs,mean_length_error=length_err/max(1,pairs))


def main(own):
    seed=1630+int(own)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed);torch.set_num_threads(1)
    fit,held=read(HERE/"landmark-data.json")["sets"]
    sets=[fit]
    if own:sets.append(read(HERE/"landmark-own.json"))
    net=LandmarkNet().cuda()
    if own:
        parent=HERE/"landmark-initial"
        chosen=read(parent/"result.json")["selected"]
        net.load_state_dict(torch.load(parent/f"s{chosen}.pt",weights_only=True,map_location="cuda"))
    lr=.0003 if own else .0008
    stop=(700,1800) if own else (1200,3200)
    output=HERE/("landmark-own-fit" if own else "landmark-initial")
    output.mkdir(exist_ok=False)
    save(output/"config.json",dict(seed=seed,lr=lr,checkpoints=stop,data=digest(HERE/"landmark-data.json"),
         extra=digest(HERE/"landmark-own.json") if own else None,source=digest(HERE/"landmark_fit.py"),
         network=digest(HERE/"landmark_model.py")))
    opt=torch.optim.AdamW(net.parameters(),lr=lr)
    stages=[]
    for step in range(1,stop[-1]+1):
        if step%13==1:
            shards=[arrays(rng.choice(list(spec["parts"]))) for spec in sets]
        batches=[]
        for j,(shard,bad) in enumerate(shards):
            count=64 if own else 112
            ids=rng.integers(len(shard[0]),size=count)
            if j and len(bad):ids[:24]=rng.choice(bad,24)
            batches.append([val[ids] for val in shard])
        p,a,m,s,last,d,sign,v,fast=[torch.from_numpy(np.concatenate(col)).cuda() for col in zip(*batches)]
        q,route,ls,sg,ok=net(p.float(),a.float(),m,s,last)
        if step==1:assert torch.equal(ok,sign[:,:,0]>=0)
        down=target(d,sign,m,s)
        truth=torch.stack((v,fast,down),2) & ok[:,:,None]
        policy=F.binary_cross_entropy_with_logits(q[ok],truth[ok].float())
        pred=q.masked_fill(~ok[:,:,None],-1e5).log_softmax(1)
        ranking=-pred.masked_fill(~truth,-1e5).logsumexp(1)[truth.any(1)].mean()
        feasible=true_routes(d,sign,s)
        transition=F.binary_cross_entropy_with_logits(route[ok],feasible[ok].float())
        direction=F.cross_entropy(sg.reshape(-1,3),sign.long().reshape(-1),ignore_index=-1)
        uncertain=p[...,1]>p[...,0]
        distances=F.cross_entropy(ls[uncertain],d[uncertain].long().clamp_max(127))
        loss=policy+.4*ranking+.55*transition+.5*direction+.4*distances
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
        if step in stop:
            path=output/f"s{step}.pt";torch.save(net.state_dict(),path)
            measure=audit(net.eval(),held);net.train()
            stages.append(dict(step=step,sha256=digest(path),score=measure))
            print(step,measure,flush=True)
    best=max(stages,key=lambda z:z["score"]["critical_viable"]+z["score"]["critical_downhill"])
    save(output/"result.json",dict(selected=best["step"],stages=stages))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--own",action="store_true")
    main(p.parse_args().own)
