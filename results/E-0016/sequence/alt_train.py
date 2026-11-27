"""Обучение на выходах из изученной области, затем RL по фактическим альтернативам."""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment
from ..data import OUT,level,rows,read,save,digest
from ..dev import evaluate
from ..policy import MOVES
from .bank import HERE
from .alt_model import ExitNet,exits,load
from .model import encode
from .train import dataset,sensor_loss


def data(split,count):
    d=dataset([(split,count)],device="cpu")
    raw=read(HERE/"exits-data.json")
    spec=next(s for s in raw["sets"] if s["split"]==split)
    parts=[]
    for name,signature in spec["parts"].items():
        file=HERE/"cache"/name
        assert digest(file)==signature
        with np.load(file) as p:
            parts.append(p["exit"])
    d["exits"]=torch.from_numpy(np.concatenate(parts))
    assert len(d["exits"])==len(d["legal"])
    return d


def supervised(net,reg,clean,weight,target,opt,rng,steps):
    critical=torch.nonzero((reg["legal"]&~reg["viable"]).any(1)).flatten().numpy()
    clean_ids=rng.choice(len(weight),size=(steps,80),p=weight)
    losses=[]
    for j in range(steps):
        ids=np.concatenate((rng.integers(len(reg["x"]),size=52),rng.choice(critical,size=28)))
        c=clean_ids[j]
        x,extra,mask,y,slack,best,viable=(torch.cat((reg[k][ids],clean[k][c])).cuda() for k in
            ("x","exits","legal","gradient","slack","best","viable"))
        scores,grad,gate=net.full(x.float(),extra.float(),mask)
        logp=scores.log_softmax(1)
        move=-logp.masked_fill(~best,-1e6).logsumexp(1)
        act=clean["partial"][c].long().cuda()
        demonstration=-logp[80:].gather(1,act.clamp_min(0)[:,None]).flatten()
        move[80:]=torch.where(act>=0,demonstration,move[80:])
        known=x[:,:,255:295].float().reshape(*x.shape[:2],5,8)
        lengths=known[:,:,:,4].masked_fill(known[:,:,:,6]<.5,4).amin(2).masked_fill(~mask,4)
        shortest=lengths.amin(1,keepdim=True)
        exact=shortest.flatten()<3
        term=-logp.masked_fill((lengths>shortest)|~mask,-1e6).logsumexp(1)
        move=torch.where(exact,term,move)
        fails=-logp.masked_fill(~viable,-1e6).logsumexp(1)
        valid=viable.any(1)
        phase=torch.from_numpy(target[c].astype(np.int64)).cuda()
        safe=x[80:,:,447:451].float().amax((1,2))>0
        gate_y=torch.where(phase>0,phase+2,torch.where(safe,1,2))
        gate_y[exact[80:]]=0
        loss=move[best.any(1)].mean()+.5*fails[valid].mean()+1.2*sensor_loss(grad,y,mask,slack)+.5*F.cross_entropy(gate[80:],gate_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
        opt.step()
        losses.append(float(loss.detach()))
    return dict(updates=steps,loss_first=float(np.mean(losses[:10])),loss_last=float(np.mean(losses[-10:])))


@torch.inference_mode()
def sensor_check(net,probe):
    hits=count=tight=down=0
    for j in range(0,len(probe["x"]),256):
        x,extra,mask,y,slack=[probe[k][j:j+256].cuda() for k in ("x","exits","legal","gradient","slack")]
        _,g,_=net.full(x.float(),extra.float(),mask)
        hits+=int(((g.argmax(-1)==y)&mask[:,:,None]).sum())
        count+=2*int(mask.sum())
        b=torch.arange(len(x),device=x.device)
        k=slack.argmin(1)
        good=(y[b,:,k]==0)&mask
        sel=(slack.amin(1)<=3)&(mask.sum(1)>1)&good.any(1)
        a=g.softmax(-1)[b,:,k,0].masked_fill(~mask,-1).argmax(1)
        tight+=int(sel.sum())
        down+=int((sel&good[b,a]).sum())
    return dict(labels=count,correct=hits,tight_decisions=tight,tight_downhill=down)


def checkpoint(net,path,dev,probe,info):
    torch.save(net.state_dict(),path)
    sig=digest(path)
    t=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),dev)
    sensor=sensor_check(net,probe)
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,summary=summary,episodes=episodes,sensor_probe=sensor,
            dev_sha256=digest(OUT/"data/dev.jsonl"),training=info,seconds=time.monotonic()-t))
    print(path.name,summary,sensor,flush=True)
    return summary["wins"],min(summary["by_tier"].values())


def branch_data(records,prior):
    bank={r["name"]:r for r in rows(HERE/"train.jsonl")}
    items=[]
    for rec in records:
        for s in rec["examples"]:
            e=Environment(level(bank[rec["map"]]))
            for c in s["prefix"]:
                e.step(MOVES[int(c)])
            x,ok=encode(e.view,prior)
            ret=np.zeros(8,np.float32)
            for branch in s["outcomes"]:
                if branch["outcome"]=="SUCCESS":
                    ret[branch["action"]]+=.5
            if np.ptp(ret[ok])>0:
                items.append((x,exits(e.view,prior),ok,ret))
    return [torch.from_numpy(np.stack(v)).cuda() for v in zip(*items)]


def improve(net,items,rng,steps):
    x,extra,mask,y=items
    for name,p in net.named_parameters():
        p.requires_grad_(name.startswith(("exit_values","exit_gate","exit_readout","exit_old")))
    opt=torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=.0002)
    with torch.no_grad():
        old=torch.cat([net.full(x[i:i+96],extra[i:i+96],mask[i:i+96])[0] for i in range(0,len(x),96)])
    for _ in range(steps):
        s=rng.integers(len(x),size=80)
        score,_,_=net.full(x[s],extra[s],mask[s])
        # Преимущество обусловлено полными исходами после сравниваемого первого хода.
        dy=y[s,:,None]-y[s,None,:]
        pairs=(dy>0)&mask[s,:,None]&mask[s,None,:]
        ds=score[:,:,None]-score[:,None,:]
        shift=score-old[s]
        loss=-(dy[pairs]*F.logsigmoid(ds[pairs])).mean()+.09*shift[mask[s]].square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    return dict(updates=steps,states=len(x),reward="terminal SUCCESS from complete train continuations")


def main(mode):
    seed=1614
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    init=HERE/"runs/mission600/rl.pt"
    net=ExitNet().cuda()
    missing,extra=net.load_state_dict(torch.load(init,weights_only=True,map_location="cuda"),strict=False)
    assert not extra and all(k.startswith("exit_") for k in missing)
    folder=HERE/"runs/exit600"
    dev=rows(OUT/"data/dev.jsonl")
    probe=data("probe",100)
    if mode=="imitate":
        folder.mkdir(exist_ok=False,parents=True)
        save(folder/"config.json",dict(seed=seed,init=digest(init),reg_maps=600,clean_maps=400,
             supervised=[900,2700],rl=[75,250],sources={n:digest(HERE/n) for n in (
             "alt_train.py","alt_model.py","alt_adapter.py","alternatives.cpp","alternatives.dll","exits-data.json")}))
        reg,clean=data("train",600),data("clean",400)
        with np.load(HERE/"cache/clean-meta.npz") as m:
            weight=m["weight"]*m["success"]
            weight=weight.astype(np.float64)/weight.sum(dtype=np.float64)
            target=m["target"]
        for name,p in net.named_parameters():
            p.requires_grad_(name.startswith("exit_"))
        opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0006)
        for tag,steps in (("s900",900),("s2700",1800)):
            info=supervised(net,reg,clean,weight,target,opt,rng,steps)
            checkpoint(net,folder/(tag+".pt"),dev,probe,info)
    else:
        candidates=[]
        for tag in ("s900","s2700"):
            summary=read(folder/(tag+".physical.json"))["summary"]
            candidates.append(((summary["wins"],min(summary["by_tier"].values())),tag))
        chosen=max(candidates,key=lambda t:t[0])[1]
        net.load_state_dict(torch.load(folder/(chosen+".pt"),weights_only=True,map_location="cuda"))
        src=HERE/"branches-240.json"
        items=branch_data(read(src)["records"],net.wall_prior.cpu().numpy().tobytes())
        for name,n in (("q75",75),("q250",175)):
            info=improve(net,items,rng,n)
            checkpoint(net,folder/(name+".pt"),dev,probe,dict(**info,from_checkpoint=chosen,data_sha256=digest(src)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("imitate","reinforce"))
    main(p.parse_args().mode)
