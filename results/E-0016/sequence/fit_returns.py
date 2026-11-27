"""Контрфактическое RL с терминальной наградой и оценкой обычных полных поездок."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from maze.environment import Environment
from ..data import OUT, level, rows, read, save, digest
from ..dev import evaluate
from ..policy import MOVES
from .bank import HERE
from .model import encode
from .value_model import ReturnNet,load


def assemble(records,prior):
    bank={r["name"]:r for r in rows(HERE/"train.jsonl")}
    items=[]
    for record in records:
        row=bank[record["map"]]
        for s in record["examples"]:
            env=Environment(level(row))
            for c in s["prefix"]:
                env.step(MOVES[int(c)])
            x,ok=encode(env.view,prior)
            assert np.flatnonzero(ok).tolist()==s["legal"]
            returns=np.zeros(8,np.float32)
            for a in s["outcomes"]:
                returns[a["action"]]+=.5*(a["outcome"]=="SUCCESS")
            items.append((x,ok,returns))
    return [torch.from_numpy(np.stack(col)).cuda() for col in zip(*items)]


def fit(net,items,opt,rng,steps):
    inputs,legal,returns=items
    spread=returns.amax(1)-returns.masked_fill(~legal,2).amin(1)
    mixed=torch.nonzero(spread>0).flatten().cpu().numpy()
    losses=[]
    for _ in range(steps):
        ids=np.concatenate((rng.integers(len(inputs),size=32),rng.choice(mixed,64)))
        x,mask,y=inputs[ids],legal[ids],returns[ids]
        _,val=net.values(x,mask)
        calibration=F.binary_cross_entropy_with_logits(val[mask],y[mask])
        diff=y[:,:,None]-y[:,None,:]
        pairs=mask[:,:,None]&mask[:,None,:]&(diff>0)
        pred=val[:,:,None]-val[:,None,:]
        ranking=-(F.logsigmoid(pred[pairs])*diff[pairs]).mean()
        loss=calibration+ranking
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        losses.append(float(loss.detach()))
    return dict(updates=steps,mixed=int(len(mixed)),loss_first=float(np.mean(losses[:10])),loss_last=float(np.mean(losses[-10:])))


def checkpoint(net,path,data,training):
    torch.save(net.state_dict(),path)
    t=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),data)
    save(path.with_suffix(".physical.json"),dict(weights_sha256=digest(path),dev_sha256=digest(OUT/"data/dev.jsonl"),
         summary=summary,episodes=episodes,evaluation_seconds=time.monotonic()-t,training=training))
    print(path.name,summary,flush=True)


def main(n):
    torch.manual_seed(1613)
    torch.set_num_threads(1)
    rng=np.random.default_rng(1613)
    init=HERE/"runs/mission600/rl.pt"
    path=HERE/f"branches-{n}.json"
    records=read(path)
    net=ReturnNet().cuda()
    old=torch.load(init,weights_only=True,map_location="cuda")
    miss,extra=net.load_state_dict(old,strict=False)
    assert not extra and all(k.startswith("return_") for k in miss)
    for name,p in net.named_parameters():
        p.requires_grad_(name.startswith("return_"))
    out=HERE/"runs"/f"returns{n}"
    out.mkdir(exist_ok=False,parents=True)
    dev=rows(OUT/"data/dev.jsonl")
    save(out/"config.json",dict(seed=1613,train_data=digest(path),init=digest(init),reference_dev=36,
        checkpoints=[[120,2],[450,2],[450,5]],sources={i:digest(HERE/i) for i in (
        "fit_returns.py","value_model.py","returns_adapter.py","branches.py")}))
    items=assemble(records["records"],net.wall_prior.cpu().numpy().tobytes())
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0008,weight_decay=.01)
    for name,steps,strength in (("q120",120,2),("q450",330,2),("q450x5",0,5)):
        if steps:
            training=fit(net,items,opt,rng,steps)
        net.return_strength.fill_(strength)
        checkpoint(net,out/(name+".pt"),dev,dict(**training,strength=strength))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("n",type=int)
    main(p.parse_args().n)
