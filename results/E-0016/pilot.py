"""Обучение на train. Запуск из корня через -m results.E-0016.pilot."""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment, Status
from .data import OUT, digest, level, read, rows, save
from .policy import MOVES, Policy, features
from .reward_test import shaped
from .teacher import Teacher
from .dev import verify

SEED=1601
DEVICE="cuda"


def tensors(xs, masks):
    return (torch.as_tensor(np.asarray(xs),device=DEVICE),torch.as_tensor(np.asarray(masks),device=DEVICE))


def train_rows(count):
    batch=rows(OUT/"data/train.jsonl")[:count]
    assert len(batch)==count and count%100==0
    return batch




def checkpoint(net,folder,tag,dev,info):
    path=folder/f"{tag}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save(net.state_dict(),path)
    weights=digest(path)
    start=time.monotonic()
    summary,episodes=verify(path)
    save(folder/f"{tag}.json",dict(weights_sha256=weights,dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,evaluation_seconds=time.monotonic()-start,**info))
    print(tag,summary,flush=True)


def collect(batch,net,rng,rollin):
    data=[]
    counts=Counter()
    for j,r in enumerate(batch):
        env=Environment(level(r))
        teacher=Teacher(env.level)
        steps=0
        while env.status==Status.RUNNING:
            targets,viable=teacher.labels(env)
            if not targets.any():
                counts["no_viable_label"]+=1
                break
            x,legal=features(env.view)
            data.append((x.astype(np.float16),legal,targets,viable))
            a=int(rng.choice(np.flatnonzero(targets)))
            if net is not None and rng.random()<rollin:
                with torch.inference_mode():
                    a=int(net(*tensors([x],[legal])).argmax())
            env.step(MOVES[a])
            steps+=1
            if steps>=240:  # Только сбор состояний train, полная оценка не имеет этой отсечки.
                counts["collection_cap"]+=1
                break
        counts[env.status.name]+=1
        if net is None:
            assert env.status==Status.SUCCESS and steps==r["optimum"],(r["name"],steps,r["optimum"])
        teacher.remaining.cache_clear()
        if (j+1)%100==0:
            print("train maps",j+1,"states",len(data),flush=True)
    return tuple(np.stack(v) for v in zip(*data)), dict(counts)


def imitation(net,data,rng,epochs,lr):
    x,legal,best,viable=data
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr)
    losses=[]
    for _ in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(order),256):
            ids=order[start:start+256]
            pred=net(*tensors(x[ids].astype(np.float32),legal[ids]))
            correct,possible=tensors(best[ids],viable[ids])
            logp=pred.log_softmax(1)
            loss=-(logp.masked_fill(~correct,-1e6).logsumexp(1)
                  +.25*logp.masked_fill(~possible,-1e6).logsumexp(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            optimizer.step()
            losses.append(float(loss.detach()))
    return dict(states=len(x),updates=len(losses),first_loss=float(np.mean(losses[:10])),
                last_loss=float(np.mean(losses[-10:])),epochs=epochs,lr=lr)


def rl(net,batch,rng,groups):
    opt=torch.optim.Adam(net.parameters(),lr=.00015)
    outcomes=Counter()
    steps=updates=0
    for group in range(groups):
        envs=[Environment(level(batch[i])) for i in rng.choice(len(batch),32,replace=False)]
        running=list(range(len(envs)))
        traces=[[] for e in envs]
        while running:
            obs=[features(envs[i].view) for i in running]
            with torch.no_grad():
                logp=(net(*tensors(*zip(*obs)))/.8).log_softmax(1)
                acts=torch.multinomial(logp.exp(),1)
                chosen=logp.gather(1,acts).flatten().tolist()
            pending=[]
            for i,(x,legal),a,old in zip(running,obs,acts.flatten().tolist(),chosen):
                env=envs[i]
                before=env.view
                env.step(MOVES[a])
                reward=shaped(before,env.view,env.status,.97)
                traces[i].append((x,legal,a,old,reward))
                if env.status==Status.RUNNING:
                    pending.append(i)
            running=pending
        outcomes.update(e.status.name for e in envs)
        samples=[]
        for trace in traces:
            value=0
            for x,legal,a,old,r in reversed(trace):
                value=r+.97*value
                samples.append((x,legal,a,old,value))
        steps+=len(samples)
        x,legal,a,old,ret=[np.asarray(v) for v in zip(*samples)]
        adv=(ret-ret.mean())/(ret.std()+1e-6)
        for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
            logp=(net(*tensors(x[ids],legal[ids]))/.8).log_softmax(1)
            chosen=logp.gather(1,torch.as_tensor(a[ids,None],device=DEVICE)).flatten()
            ratio=(chosen-torch.as_tensor(old[ids],device=DEVICE,dtype=torch.float32)).exp()
            advantage=torch.as_tensor(adv[ids],device=DEVICE,dtype=torch.float32)
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.85,1.15)*advantage).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            updates+=1
        print("RL",group+1,steps,dict(outcomes),flush=True)
    return dict(steps=steps,updates=updates,training_outcomes=dict(outcomes),gamma=.97,groups=groups)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--maps",type=int,default=200)
    p.add_argument("--init",type=Path)
    p.add_argument("--rollin",type=float,default=0)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--groups",type=int,default=4)
    args=p.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng=np.random.default_rng(SEED)
    folder=OUT/"runs"/args.name
    folder.mkdir(parents=True,exist_ok=False)
    train=train_rows(args.maps)
    dev=rows(OUT/"data/dev.jsonl")
    assert not {r["geometry"] for r in train} & {r["geometry"] for r in dev}
    save(folder/"config.json",dict(seed=SEED,maps=args.maps,rollin=args.rollin,epochs=args.epochs,
         groups=args.groups,init=str(args.init) if args.init else None,
         init_sha256=digest(args.init) if args.init else None,sources={n:digest(OUT/n) for n in
         ("pilot.py","policy.py","geometry.cpp","geometry.dll","teacher.py","reward_test.py")},
         bank=read(OUT/"data/manifest.json")["splits"]))
    net=Policy().to(DEVICE)
    if args.init:
        net.load_state_dict(torch.load(args.init,weights_only=True,map_location=DEVICE))
    else:
        checkpoint(net,folder,"initial",dev,{})
    t=time.monotonic()
    data,collection=collect(train,net if args.rollin else None,rng,args.rollin)
    info=imitation(net,data,rng,args.epochs,.0008 if not args.init else .00035)
    checkpoint(net,folder,"imitation",dev,dict(training=info,collection=collection,train_seconds=time.monotonic()-t))
    del data
    t=time.monotonic()
    info=rl(net,train,rng,args.groups)
    checkpoint(net,folder,"rl",dev,dict(training=info,train_seconds=time.monotonic()-t))


if __name__=="__main__":
    main()
