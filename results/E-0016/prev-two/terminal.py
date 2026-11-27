"""Терминальный RL на CPU. Парный градиент и поиск весов линейного выхода по SUCCESS."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import sys
import numpy as np
import torch
from maze.environment import Environment,Status
from .adapter import MOVES,tensor_batch
from .common import HERE,SEED,group,level,read,write
from .fit import freeze,loadnet

NET=None


def init_worker(path):
    global NET
    torch.set_num_threads(1)
    NET=loadnet(path).eval().requires_grad_(False)


@torch.inference_mode()
def rollout(env,weights,cache,rng=None,previous=(None,None)):
    trajectory=[]
    while env.status==Status.RUNNING:
        v=env.view
        key=(v,previous)
        if key not in cache:
            pack=NET.features(v,previous)
            parts=NET.parts(*tensor_batch([pack]))[0][0].numpy()
            cache[key]=(parts,pack[-1])
        parts,ok=cache[key]
        score=np.where(ok,parts@weights,-1e6)
        if rng is None:
            a=int(score.argmax())
        else:
            prob=np.exp((score-score.max())/.8,dtype=np.float64)
            prob/=prob.sum()
            a=int(rng.choice(8,p=prob))
            trajectory.append((parts,ok,a,float(np.log(prob[a]))))
        previous=(v.fuel_distance,v.repair_distance)
        env.step(MOVES[a])
        assert env.moved
    return dict(win=int(env.status==Status.SUCCESS),steps=env.steps,outcome=env.status.name,
                both=int(min(env.refuels,env.repairs)>0)),trajectory


def paired(job):
    row,weights,seed=job
    rng=np.random.default_rng(seed)
    cache={}
    env=Environment(level(row))
    greedy,_=rollout(env,weights,cache)
    env.reset()
    random,trajectory=rollout(env,weights,cache,rng)
    adv=random["win"]-greedy["win"]
    sample=[]
    if adv:
        for j in rng.choice(len(trajectory),size=96,replace=True):
            sample.append((*trajectory[j],adv))
    return row["name"],greedy,random,sample


def population(job):
    row,options=job
    env=Environment(level(row));cache={};out=[]
    for w in options:
        env.reset()
        out.append(rollout(env,w,cache)[0])
    return row["name"],out


def teach_mix(workers,net,maps,rng):
    optimizer=torch.optim.Adam([net.mix],lr=.09)
    history=[]
    for start in range(0,len(maps),12):
        weights=net.mix.detach().numpy().copy()
        jobs=[(r,weights,int(rng.integers(2**30))) for r in maps[start:start+12]]
        samples=[]; rows=[]
        for name,a,b,d in workers.map(paired,jobs):
            samples.extend(d)
            rows.append(dict(map=name,greedy=a,sampled=b))
        if samples:
            parts,legal,actions,old,adv=(torch.from_numpy(np.asarray(k)) for k in zip(*samples))
            for _ in range(4):
                logp=((parts@net.mix)/.8).masked_fill(~legal,-1e6).log_softmax(1)
                ratio=(logp[torch.arange(len(actions)),actions]-old.float()).exp()
                loss=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_([net.mix],.8);optimizer.step()
        history.append(dict(pairs=rows,updates=4 if samples else 0,weights=net.mix.detach().tolist()))
        print("terminal gradient",start+len(rows),"maps; differing",len(samples)//96,flush=True)
    return history


def evaluate_population(workers,maps,options):
    results={}
    for name,records in workers.map(population,[(r,options) for r in maps]):
        results[name]=records
        if len(results)%12==0: print("population maps",len(results),"/",len(maps),flush=True)
    totals=[sum(row[i]["win"] for row in results.values()) for i in range(len(options))]
    return results,totals


def train(parent,label,round_no=1):
    torch.set_num_threads(1)
    stage=HERE/"weights"/(parent+".pt")
    rng=np.random.default_rng(SEED+round_no*71)
    net=loadnet(stage)
    bank=group("terminal")
    order=rng.permutation(len(bank))
    maps=[bank[i] for i in order]
    initial=net.mix.detach().numpy().copy()
    with ProcessPoolExecutor(2,initializer=init_worker,initargs=(stage,)) as workers:
        grad=teach_mix(workers,net,maps[:48],rng)
        policy=net.mix.detach().numpy().copy()
        freeze(net,label+"-pg",dict(parent=parent,method="paired terminal policy gradient",
                                    groups=grad,seed=SEED+round_no*71))
        # Начальная популяция охватывает разные уровни доверия прогнозу и
        # подтверждённой дороге. Окончательные веса определяет число финишей на train.
        options=[initial,policy]
        for j in range(14):
            centre=policy.copy()
            centre[0]=rng.uniform(.55,1.2)
            centre[1]=rng.uniform(.25,1.15)
            centre[2]+=rng.uniform(1,12)
            centre[3]+=rng.uniform(-1,7)
            centre[4]+=rng.uniform(1,16)
            centre[5]+=rng.uniform(1,13)
            centre[6]+=rng.uniform(0,6)
            centre[7]+=rng.uniform(-.5,2)
            centre[8]+=rng.uniform(-1,9)
            options.append(centre)
        stage1,scores=evaluate_population(workers,maps[48:72],options)
        top=sorted(range(len(options)),key=lambda i:-scores[i])[:4]
        pool=sorted(set([0,1,*top]))
        stage2,test=evaluate_population(workers,maps[72:120],[options[i] for i in pool])
    best=pool[max(range(len(pool)),key=lambda j:(test[j],scores[pool[j]],-pool[j]))]
    with torch.no_grad():net.mix.copy_(torch.from_numpy(options[best]))
    stats=dict(parent=parent,gradient_tag=label+"-pg",reward="SUCCESS",seed=SEED+round_no*71,
               weights=[x.tolist() for x in options],first=stage1,first_scores=scores,
               finalist_indices=pool,second=stage2,second_scores=test,chosen=best)
    write(HERE/"terminal"/(label+".json"),stats)
    freeze(net,label,dict(parent=parent,method="terminal population after policy gradient",
                          selected=best,first_scores=scores,second_scores=test,
                          train_file="terminal/"+label+".json"))
    print("terminal selected",best,scores,test,flush=True)


if __name__=="__main__":
    parent=sys.argv[1]
    if parent in ("auto1","auto2"):
        from .compare import rank
        tags=["super600","super1500"] if parent=="auto1" else ["extra600","extra1300"]
        parent=max(tags,key=rank)
    train(parent,sys.argv[2],int(sys.argv[3]))
