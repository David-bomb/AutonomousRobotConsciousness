from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from torch.nn import functional as F

from maze.datasets import geometry
from maze.environment import Environment, Status
from ....data import digest, read, rows, level, save
from ....teacher import Teacher
from .bank import HERE, CACHE, PACK, RANGES
from .features import MOVES, frame, phase, sign_index, early_model
from .fit import dataset, targets, audit
from .model import Decision, FIELDS


def extra(split):
    whole = rows(HERE.parent/"large.jsonl")
    return whole[7600:7800] if split == "fit" else whole[7800:7900]


def register():
    group={}
    used={r["geometry"] for r in rows(HERE.parent/"large.jsonl")[7000:7600]}
    for s in ("fit","selection"):
        bank=extra(s)
        keys=[geometry(r["grid"]) for r in bank]
        assert keys == [r["geometry"] for r in bank] and len(set(keys))==len(bank) and not set(keys)&used
        used.update(keys)
        tiers=Counter(r["tier"] for r in bank)
        assert set(tiers.values())=={len(bank)//4}
        group[s]=dict(tiers=tiers,geometries=keys)
    cfg=dict(primary=digest(HERE/"decision.pt"), protocol=digest(HERE/"dagger-plan.md"),
             source=digest(HERE/"dagger.py"), earlier=digest(HERE/"registration.json"),groups=group)
    lock=HERE/"dagger-registration.json"
    if lock.exists():assert read(lock)==cfg
    else:save(lock,cfg)


def initialize():
    global early, net
    torch.set_num_threads(1)
    early=early_model(torch.load(PACK,weights_only=True,map_location="cpu")["early"])
    net=Decision().requires_grad_(False).eval()
    net.load_state_dict(torch.load(HERE/"decision.pt",weights_only=True,map_location="cpu"))


@torch.inference_mode()
def collect(job):
    row,index=job
    path=CACHE/"dagger"/(row["name"]+".npz")
    if path.exists():return
    env=Environment(level(row));teacher=Teacher(env.level)
    previous=None; samples=[];seen=set();repairs=0
    for t in range(100):
        v=env.view
        if env.status!=Status.RUNNING or phase(v.memory):break
        d=frame(early,v)
        current=np.array([v.fuel_distance,v.repair_distance],np.int16)
        label=sign_index(previous,current)
        if label<9:assert d["allowed"][label]
        p,_=net(*(torch.as_tensor(d[k])[None] for k in FIELDS))
        scores=p[0].softmax(0).numpy()@d["options"] if label<9 else d["missing"]
        chosen=int(np.where(d["legal"],scores,-np.inf).argmax())
        truth=d["options"][label] if label<9 else d["missing"]
        _,safe=teacher.labels(env)
        sensitive=len(set(d["options"][d["allowed"]].argmax(1)))>1
        key=(v,label)
        if key not in seen:
            samples.append(dict(**d,previous=np.full(2,-1,np.int16) if previous is None else previous,
                                sensitive=sensitive,label=label,viable=safe,step=t))
        elif index%2==0 or repairs>=2:break
        seen.add(key)
        if index%2==1 and repairs<2 and not safe[chosen] and safe.any():
            chosen=int(np.where(safe,truth,-np.inf).argmax());repairs+=1
        previous=current
        env.step(MOVES[chosen])
    teacher.remaining.cache_clear()
    arr={k:np.stack([r[k] for r in samples]) for k in samples[0]}
    arr["x"]=arr["x"].astype(np.float16)
    with path.with_suffix(".pending").open("wb") as f:np.savez_compressed(f,**arr)
    path.with_suffix(".pending").replace(path)


def main():
    register()
    bank=extra("fit")
    folder=CACHE/"dagger";folder.mkdir(exist_ok=True)
    with ProcessPoolExecutor(3,initializer=initialize) as pool:
        fs=[pool.submit(collect,(r,i)) for i,r in enumerate(bank)]
        for i,f in enumerate(as_completed(fs),1):
            f.result()
            if i%50==0:print("own",i,flush=True)
    save(HERE/"dagger-data.json",dict(registration=digest(HERE/"dagger-registration.json"),
                                      files={r["name"]:digest(folder/(r["name"]+".npz")) for r in bank}))
    train,own,held=dataset("fit"),dataset("dagger"),dataset("held")
    net=Decision().cuda();net.load_state_dict(torch.load(HERE/"decision.pt",weights_only=True))
    optimizer=torch.optim.AdamW(net.parameters(),lr=.00016)
    rng=np.random.default_rng(593);torch.manual_seed(593)
    ids=[]
    for data in (train,own):
        good=data["label"]<9
        ids.append([torch.where(good)[0].cpu().numpy(),torch.where(good&data["sensitive"])[0].cpu().numpy()])
    for step in range(480):
        batch={}
        for d,(all_ids,hard) in zip((train,own),ids):
            ix=np.r_[rng.choice(all_ids,72),rng.choice(hard,48)]
            for k,v in d.items():batch.setdefault(k,[]).append(v[ix])
        d={k:torch.cat(v) for k,v in batch.items()}
        p,act=net(*(d[k] for k in FIELDS))
        y=d["label"];q=d["options"][torch.arange(len(y),device=y.device),y]
        target=q.masked_fill(~d["viable"]&d["viable"].any(1,keepdim=True),-1000).argmax(1)
        loss=F.cross_entropy(p,y)+.85*F.cross_entropy(act,target)+.25*F.kl_div(
            (act/1.5).log_softmax(1),(q/1.5).softmax(1),reduction="batchmean")
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.5);optimizer.step()
    groups=audit(net.eval(),held,read(HERE/"held.json")["prior"])
    dest=HERE/"dagger.pt";torch.save(net.cpu().state_dict(),dest)
    save(HERE/"dagger-held.json",dict(weights=digest(dest),own_states=len(own["label"]),updates=480,groups=groups))
    print("own training",len(own["label"]),groups["sensitive"]["joint"],flush=True)


if __name__=="__main__":main()
