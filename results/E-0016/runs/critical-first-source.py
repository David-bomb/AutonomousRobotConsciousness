"""Третья волна. Разметка критических состояний и RL по физическому финишу."""
import argparse
from collections import Counter
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment, Status
from scripts.audit_automaton import Reachability
from .data import OUT, digest, level, rows, save
from .critical_policy import CriticalPolicy
from .policy import MOVES,features
from .pilot import tensors,train_rows,checkpoint
from .teacher import Teacher


def gather(net,train,rng):
    buffer=[]
    counts=Counter()
    for i,r in enumerate(train):
        env=Environment(level(r))
        teacher=Teacher(env.level)
        oracle=Reachability(env.level)
        seen=Counter()
        while env.status==Status.RUNNING and env.steps<240:
            best,viable=teacher.labels(env)
            x,legal=features(env.view)
            if not best.any():
                break
            critical=(legal & ~viable).any()
            if critical:
                exact=np.zeros(8,bool)
                for m,q in oracle.edges[env.position]:
                    exact[MOVES.index(m)]=oracle.possible(q,*oracle.after(q,env.fuel,env.health),env.limit-env.steps-1)
                counts["label_disagreements"]+=int(any(exact!=viable))
                viable=exact
                critical=(legal & ~viable).any()
            buffer.append((x.astype(np.float16),legal,best,viable))
            with torch.inference_mode():
                a=int(net(*tensors([x],[legal])).argmax())
            counts["critical"]+=int(critical)
            counts["intercepted"]+=int(not viable[a])
            seen[env.view.memory,env.position]+=1
            if not viable[a] or seen[env.view.memory,env.position]>1 or rng.random()<.12:
                a=int(rng.choice(np.flatnonzero(best)))
            env.step(MOVES[a])
        counts[env.status.name]+=1
        teacher.remaining.cache_clear()
        if (i+1)%100==0:
            print("critical maps",i+1,len(buffer),dict(counts),flush=True)
    return tuple(np.stack(col) for col in zip(*buffer)),counts


def fit(net,data,rng):
    x,legal,best,viable=data
    risky=np.flatnonzero(np.any(legal & ~viable,axis=1))
    opt=torch.optim.AdamW(net.parameters(),lr=.0004)
    losses=[]
    for _ in range(8):
        order=rng.permutation(len(x))
        for start in range(0,len(x),192):
            ids=np.concatenate((order[start:start+192],rng.choice(risky,64)))
            values,mask=tensors(x[ids].astype(np.float32),legal[ids])
            good,possible=tensors(best[ids],viable[ids])
            pred,safety=net.heads(values,mask)
            classify=F.binary_cross_entropy_with_logits(safety[mask],possible[mask].float())
            logp=pred.log_softmax(1)
            loss=classify-(logp.masked_fill(~good,-1e6).logsumexp(1)
                         +.5*logp.masked_fill(~possible,-1e6).logsumexp(1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
            losses.append(float(loss.detach()))
    return dict(states=len(x),critical=len(risky),updates=len(losses),last_loss=float(np.mean(losses[-10:])))


def finish_rl(net,train,rng,groups):
    opt=torch.optim.Adam(net.parameters(),lr=.00008)
    outcomes=Counter()
    steps=updates=0
    for g in range(groups):
        envs=[Environment(level(train[i])) for i in rng.choice(len(train),32,replace=False)]
        active=list(range(32))
        samples=[]
        while active:
            inputs=[features(envs[i].view) for i in active]
            with torch.no_grad():
                logp=(net(*tensors(*zip(*inputs)))/.6).log_softmax(1)
                actions=torch.multinomial(logp.exp(),1)
                old=logp.gather(1,actions).flatten().tolist()
            pending=[]
            for i,(x,mask),a,l in zip(active,inputs,actions.flatten().tolist(),old):
                samples.append((i,x,mask,a,l))
                env=envs[i]
                env.step(MOVES[a])
                if env.status==Status.RUNNING:
                    pending.append(i)
            active=pending
        steps+=len(samples)
        outcomes.update(e.status.name for e in envs)
        wins=np.asarray([e.status==Status.SUCCESS for e in envs],np.float32)
        # Средний исход остальных поездок группы независим от действий этой поездки.
        adv=wins-(wins.sum()-wins)/31
        lane,x,mask,act,old=map(np.asarray,zip(*samples))
        for ids in np.array_split(rng.permutation(len(x)),max(1,len(x)//384)):
            logp=(net(*tensors(x[ids],mask[ids]))/.6).log_softmax(1)
            loga=logp.gather(1,torch.as_tensor(act[ids,None],device="cuda")).flatten()
            ratio=(loga-torch.as_tensor(old[ids],device="cuda",dtype=torch.float32)).exp()
            advantage=torch.as_tensor(adv[lane[ids]],device="cuda")
            loss=-torch.minimum(ratio*advantage,ratio.clamp(.9,1.1)*advantage).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
            opt.step()
            updates+=1
        print("finish RL",g+1,steps,dict(outcomes),flush=True)
    return dict(groups=groups,steps=steps,training_outcomes=dict(outcomes),updates=updates,gamma=1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("init",type=Path)
    p.add_argument("--maps",type=int,default=800)
    p.add_argument("--groups",type=int,default=12)
    args=p.parse_args()
    folder=OUT/"runs"/args.name
    folder.mkdir(parents=True,exist_ok=False)
    seed=1602
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    train=train_rows(args.maps)
    dev=rows(OUT/"data/dev.jsonl")
    net=CriticalPolicy().cuda()
    prior=torch.load(args.init,weights_only=True,map_location="cuda")
    missing,extra=net.load_state_dict(prior,strict=False)
    assert not extra and all(k.startswith("risk.") for k in missing)
    save(folder/"config.json",dict(seed=seed,train_sha256=digest(OUT/"data/train.jsonl"),dev_sha256=digest(OUT/"data/dev.jsonl"),
         maps=args.maps,groups=args.groups,init=str(args.init),init_sha256=digest(args.init),
         sources={n:digest(OUT/n) for n in ("critical.py","critical_policy.py","policy.py","pilot.py","teacher.py","dev.py","adapter.py","geometry.cpp","geometry.dll")}))
    t=time.monotonic()
    data,counts=gather(net,train,rng)
    info=fit(net,data,rng)
    checkpoint(net,folder,"imitation",dev,dict(training=info,collection=counts,train_seconds=time.monotonic()-t))
    del data
    t=time.monotonic()
    info=finish_rl(net,train,rng,args.groups)
    checkpoint(net,folder,"rl",dev,dict(training=info,train_seconds=time.monotonic()-t))


if __name__=="__main__":
    main()
