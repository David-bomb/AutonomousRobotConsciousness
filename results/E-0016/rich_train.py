"""Параметры оценки неоткрытых клеток встроены в один файл весов."""
import argparse
from functools import partial
from pathlib import Path
import time
import numpy as np
import torch
from .data import OUT,read,rows,digest,save
from .critical import gather,fit,finish_rl
from .pilot import checkpoint,train_rows
from .priors import fit_prior
from .rich_policy import RichPolicy,inputs


def gather_batches(net,train,rng,encode):
    from collections import Counter
    from maze.environment import Environment,Status
    from scripts.audit_automaton import Reachability
    from .data import level
    from .pilot import tensors
    from .policy import MOVES
    from .teacher import Teacher

    data=[]
    counts=Counter()
    for start in range(0,len(train),20):
        active=[]
        for row in train[start:start+20]:
            env=Environment(level(row))
            active.append((env,Teacher(env.level),Reachability(env.level),Counter()))
        while active:
            encoded=[encode(env.view) for env,_,_,_ in active]
            with torch.inference_mode():
                actions=net(*tensors(*zip(*encoded))).argmax(1).tolist()
            pending=[]
            for state,(x,legal),a in zip(active,encoded,actions):
                env,teacher,oracle,seen=state
                best,viable=teacher.labels(env)
                if not best.any():
                    counts["no_viable_label"]+=1
                    teacher.remaining.cache_clear()
                    continue
                if (legal & ~viable).any():
                    exact=np.zeros(8,bool)
                    for m,q in oracle.edges[env.position]:
                        exact[MOVES.index(m)]=oracle.possible(q,*oracle.after(q,env.fuel,env.health),env.limit-env.steps-1)
                    counts["label_disagreements"]+=int(any(exact!=viable))
                    viable=exact
                data.append((x.astype(np.float16),legal,best,viable))
                counts["critical"]+=int((legal & ~viable).any())
                counts["intercepted"]+=int(not viable[a])
                key=env.view.memory,env.position
                seen[key]+=1
                if not viable[a] or seen[key]>1 or rng.random()<.12:
                    a=int(rng.choice(np.flatnonzero(best)))
                env.step(MOVES[a])
                if env.status==Status.RUNNING and env.steps<240:
                    pending.append(state)
                else:
                    counts[env.status.name]+=1
                    teacher.remaining.cache_clear()
            active=pending
        if (start+20)%100==0:
            print("train",start+20,"states",len(data),dict(counts),flush=True)
    return tuple(np.stack(column) for column in zip(*data)),counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("init",type=Path)
    parser.add_argument("--maps",type=int,default=800)
    parser.add_argument("--groups",type=int,default=8)
    args=parser.parse_args()
    folder=OUT/"runs"/args.name
    folder.mkdir(parents=True,exist_ok=False)
    seed=1603
    rng=np.random.default_rng(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net=RichPolicy().cuda()
    prior=torch.load(args.init,weights_only=True,map_location="cuda")
    missing,extra=net.load_state_dict(prior,strict=False)
    assert not extra and all(k.startswith(("ext_","risk.")) or k=="wall_prior" for k in missing)
    train=train_rows(args.maps)
    dev=rows(OUT/"data/dev.jsonl")
    probability=fit_prior(train)
    net.wall_prior.copy_(torch.from_numpy(probability))
    encode=partial(inputs,prior_bytes=probability.tobytes())
    save(folder/"config.json",dict(init=str(args.init),init_sha256=digest(args.init),seed=seed,
         maps=args.maps,groups_per_phase=args.groups,rl_phases=2,bank=read(OUT/"data/manifest.json")["splits"],
         sources={n:digest(OUT/n) for n in ("rich_train.py","rich_policy.py","priors.py","critical.py","critical_policy.py",
         "policy.py","pilot.py","dev.py","geometry.cpp","geometry.dll","adapter.py","teacher.py")}))
    t=time.monotonic()
    data,counts=gather_batches(net,train,rng,encode)
    # Убираем сдвиг уже выученных признаков в контролируемой фазе.
    for name, parameter in net.named_parameters():
        parameter.requires_grad_(name.startswith("ext_"))
    info=fit(net,data,rng)
    checkpoint(net,folder,"imitation",dev,dict(training=info,collection=counts,train_seconds=time.monotonic()-t))
    del data
    net.requires_grad_(True)
    for phase in (1,2):
        t=time.monotonic()
        info=finish_rl(net,train,rng,args.groups,encode)
        checkpoint(net,folder,f"rl{phase}",dev,dict(training=info,train_seconds=time.monotonic()-t))


if __name__=="__main__":
    main()
