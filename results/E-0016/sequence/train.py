"""Совместное обучение прогноза датчиков, решения о ходе и последующее RL."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment,Status
from ..data import OUT,read,rows,save,digest,level
from ..dev import evaluate
from ..policy import MOVES
from ..priors import fit_prior
from .adapter import load
from .bank import HERE
from .model import SequenceNet,encode


def dataset(specs,device="cuda"):
    output={}
    for split,n in specs:
        manifest=read(HERE/f"prepare-{split}-{n}.json")
        for name,meta in manifest["parts"].items():
            path=HERE/"cache"/name
            assert digest(path)==meta["sha256"]
            with np.load(path) as part:
                for key in part:
                    output.setdefault(key,[]).append(part[key])
    for key in output:
        output[key]=torch.from_numpy(np.concatenate(output[key])).to(device)
    return output


def sensor_loss(pred,y,mask,slack):
    errors=F.cross_entropy(pred.flatten(0,2),y.long().flatten(),reduction="none").view_as(y)
    weights=mask[:,:,None]*(1+2*(slack[:,None,:]<=3))*(1.+.5*(mask.sum(1)[:,None,None]>1))
    return (errors*weights).sum()/weights.sum()


@torch.inference_mode()
def diagnose(net,probe):
    totals=Counter()
    n=len(probe["x"])
    for lo in range(0,n,256):
        sl=slice(lo,lo+256)
        mask=probe["legal"][sl]
        expected=probe["gradient"][sl].long()
        _,grad=net.sensor(probe["x"][sl].float(),mask)
        hit=grad.argmax(-1)==expected
        totals["labels"]+=int(mask.sum())*2
        totals["correct"]+=int((hit&mask[:,:,None]).sum())
        # На каждой позиции — сервис с меньшим запасом сверх показанной длины.
        slack=probe["slack"][sl]
        kind=slack.argmin(1)
        b=torch.arange(len(mask),device=mask.device)
        downhill=(expected[b,:,kind]==0)&mask
        eligible=(slack.min(1).values<=3)&downhill.any(1)&(mask.sum(1)>1)
        confidence=grad.softmax(-1)[b,:,kind,0].masked_fill(~mask,-1)
        choice=confidence.argmax(1)
        totals["tight_decisions"]+=int(eligible.sum())
        totals["tight_downhill"]+=int((eligible&downhill[b,choice]).sum())
        totals["uniform_expected"]+=float(((downhill.sum(1)/mask.sum(1))*eligible).sum())
    return dict(totals)


def supervised(net,data,opt,rng,epochs):
    losses=[]
    net.train()
    for _ in range(epochs):
        order=rng.permutation(len(data["x"]))
        for lo in range(0,len(order),256):
            ids=torch.as_tensor(order[lo:lo+256],device="cuda")
            legal=data["legal"][ids]
            scores,grad=net.heads(data["x"][ids].float(),legal)
            best=data["best"][ids]
            partial=data["partial"][ids].long()
            logp=scores.log_softmax(1)
            has=best.any(1)
            imitation=-logp.masked_fill(~best,-1e6).logsumexp(1)
            if has.any():
                guide=-logp.gather(1,partial.clamp_min(0)[:,None]).flatten()
                imitation=torch.where(partial>=0,.45*imitation+.55*guide,imitation)
                move_loss=imitation[has].mean()
            else:
                move_loss=scores.sum()*0
            loss=move_loss+1.5*sensor_loss(grad,data["gradient"][ids],legal,data["slack"][ids])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            losses.append(float(loss.detach()))
    return dict(epochs=epochs,updates=len(losses),first_loss=float(np.mean(losses[:10])),last_loss=float(np.mean(losses[-10:])))


def paired_rl(net,train,rng,groups,train_prefixes=("mission.","action.")):
    # Прогноз датчиков остаётся калиброванным на train. RL меняет взвешивание признаков выбора хода.
    for name,weight in net.named_parameters():
        weight.requires_grad_(name.startswith(train_prefixes))
    opt=torch.optim.Adam((p for p in net.parameters() if p.requires_grad),lr=.00008)
    prior=net.wall_prior.detach().cpu().numpy().tobytes()
    stats=Counter()
    for g in range(groups):
        selected=rng.choice(len(train),24,replace=False)
        envs=[Environment(level(train[i])) for i in selected for _ in range(2)]
        cache={}
        states=[]
        logs=[[] for _ in selected]
        active=list(range(len(envs)))
        while active:
            unique=list(dict.fromkeys(e.view for i in active if (e:=envs[i]).view not in cache))
            if unique:
                pairs=[encode(v,prior) for v in unique]
                x,mask=map(np.stack,zip(*pairs))
                with torch.no_grad():
                    logp=(net(torch.from_numpy(x).cuda(),torch.from_numpy(mask).cuda())/.6).log_softmax(1).cpu().numpy()
                for v,arr,ok,lp in zip(unique,x,mask,logp):
                    cache[v]=(len(states),int(lp.argmax()),np.cumsum(np.exp(lp),dtype=np.float64),lp)
                    states.append((arr.astype(np.float16),ok))
            following=[]
            for i in active:
                env=envs[i]
                index,greedy,cdf,lp=cache[env.view]
                if i%2:
                    a=min(7,int(np.searchsorted(cdf,rng.random()*cdf[-1],side="right")))
                    logs[i//2].append((index,a,lp[a]))
                else:
                    a=greedy
                env.step(MOVES[a])
                stats["physical_steps"]+=1
                if env.status==Status.RUNNING:
                    following.append(i)
            active=following
        samples=[]
        for i,trajectory in enumerate(logs):
            base=envs[i*2].status==Status.SUCCESS
            win=envs[i*2+1].status==Status.SUCCESS
            stats["greedy_wins"]+=int(base)
            stats["sample_wins"]+=int(win)
            adv=int(win)-int(base)
            if adv:
                stats["gained" if adv>0 else "lost"]+=1
                # Равная масса длинного и короткого эпизодов при их сравнении.
                choices=rng.choice(len(trajectory),size=96,replace=True)
                samples.extend((*trajectory[j],adv) for j in choices)
        rng.shuffle(samples)
        for lo in range(0,len(samples),192):
            index,action,old,adv=map(np.asarray,zip(*samples[lo:lo+192]))
            x,mask=map(np.stack,zip(*(states[k] for k in index)))
            scores=net(torch.from_numpy(x).cuda().float(),torch.from_numpy(mask).cuda())/.6
            chosen=scores.log_softmax(1).gather(1,torch.from_numpy(action[:,None]).cuda()).flatten()
            ratio=(chosen-torch.from_numpy(old).cuda()).exp()
            advantage=torch.from_numpy(adv).cuda().float()
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.9,1.1)*advantage).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
            opt.step()
            stats["updates"]+=1
        print("paired RL",g+1,dict(stats),flush=True)
    return dict(stats)


def checkpoint(net,folder,name,dev,probe,info):
    path=folder/f"{name}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save(net.state_dict(),path)
    signature=digest(path)
    sensor=diagnose(net.eval(),probe)
    t=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),dev)
    save(path.with_suffix(".physical.json"),dict(weights_sha256=signature,dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,sensor_probe=sensor,evaluation_seconds=time.monotonic()-t,training=info))
    print(name,summary,sensor,flush=True)
    return summary["wins"],min(summary["by_tier"].values())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--maps",type=int,default=600)
    p.add_argument("--partial",type=int,default=100)
    p.add_argument("--groups",type=int,default=6)
    args=p.parse_args()
    seed=1611
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    frozen=read(OUT/"frozen.json")
    path=OUT/frozen["weights"]
    assert digest(path)==frozen["weights_sha256"]
    base=torch.load(path,map_location="cpu",weights_only=True)
    net=SequenceNet("rich" if "wall_prior" in base else "critical" if "risk.0.weight" in base else "plain").cuda()
    net.base.load_state_dict(base)
    sensor_weights=torch.load(HERE/"sensor.pt",weights_only=True,map_location="cuda")
    net.load_state_dict(sensor_weights,strict=False)
    net.base.requires_grad_(False)
    net.wall_prior.copy_(torch.from_numpy(fit_prior(rows(OUT/"data/train.jsonl"))))
    folder=HERE/"runs"/args.name
    folder.mkdir(exist_ok=False,parents=True)
    save(folder/"config.json",dict(seed=seed,init=frozen["weights_sha256"],maps=args.maps,partial_maps=args.partial,
         rl_groups=args.groups,learning_rate=.0006,sensor_sha256=digest(HERE/"sensor.pt"),source_sha256={n:digest(HERE/n) for n in
         ("train.py","model.py","geometry.cpp","sequence.dll","adapter.py","prepare.py","protocol.md")}))
    data=dataset([("train",args.maps),("partial",args.partial)])
    probe=dataset([("probe",100)])
    dev=rows(OUT/"data/dev.jsonl")
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0006)
    results=[]
    for name,epochs in (("s3",3),("s8",5)):
        t=time.monotonic()
        info=supervised(net,data,opt,rng,epochs)
        info["seconds"]=time.monotonic()-t
        results.append((checkpoint(net,folder,name,dev,probe,info),name))
    selected=max(results,key=lambda pair:pair[0])[1]
    net.load_state_dict(torch.load(folder/f"{selected}.pt",map_location="cuda",weights_only=True))
    del data,opt
    t=time.monotonic()
    info=paired_rl(net,rows(HERE/"train.jsonl")[:args.maps],rng,args.groups)
    info.update(seconds=time.monotonic()-t,from_checkpoint=selected)
    checkpoint(net,folder,"rl",dev,probe,info)


if __name__=="__main__":
    main()
