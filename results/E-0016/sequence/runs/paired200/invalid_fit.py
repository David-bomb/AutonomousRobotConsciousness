"""Градиент терминальной отдачи со сравнимых первых действий одного физического состояния."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from maze.grid import locations
from ...data import OUT, digest, read, rows, save
from ...dev import evaluate
from .model import HERE
from .terminal_model import OutcomeHead, Signals, load


def foundation():
    gate=read(HERE/"gate-3000/result.json")["selected"]
    local=read(HERE/"local-12000/results.json")["selected"]
    return dict(foundation=torch.load(HERE.parent/"runs/field3200/rl.pt",weights_only=True,map_location="cpu"),
          gate=torch.load(HERE/f"gate-3000/s{gate}.pt",weights_only=True,map_location="cpu"),
          local=torch.load(HERE/f"local-12000/s{local}.pt",weights_only=True,map_location="cpu"))


def dataset(name,bank,signals):
    cols={}
    for row in bank:
        path=HERE.parent/"cache/terminal-choices"/name/(row["name"]+".npz")
        start=locations(tuple(row["grid"]),"A")[0]
        with np.load(path) as data:
            if not data.files: continue
            n=len(data["last"])
            for k in data.files: cols.setdefault(k,[]).append(data[k])
            cols.setdefault("start",[]).append(np.tile(start,(n,1)))
    d={k:torch.from_numpy(np.concatenate(v)) for k,v in cols.items()}
    x=[]
    names=("gates","patch","meta","stocks","last","scores","legal","start")
    for i in range(0,len(d["meta"]),64):
        x.append(signals(*(d[k][i:i+64] for k in names)))
    return torch.cat(x),d["scores"],d["legal"],d["wins"],d["safe"]


@torch.inference_mode()
def evaluate_snapshots(net,data):
    x,base,ok,wins,_=data
    q=net(x,ok)
    idx=torch.arange(len(x))
    original=base.masked_fill(~ok,-1e6).argmax(1)
    old=wins[idx,original]
    configs=[]
    for gain in (.5,1,2,4,6):
        for value in (0,2):
            raw=base+gain*q[:,:,0].clamp(-4,4)+value*q[:,:,1].sigmoid()
            a=raw.masked_fill(~ok,-1e6).argmax(1)
            hit=wins[idx,a]
            up=int((hit&~old).sum());down=int((old&~hit).sum())
            configs.append(dict(weights=[1,gain,value],gained=up,lost=down,net=up-down,changed=int((a!=original).sum())))
    configs.sort(key=lambda z:(-z["net"],z["lost"],z["changed"]))
    return dict(baseline=int(old.sum()),snapshots=len(x),mixed=int((wins.any(1)&(ok&~wins).any(1)).sum()),rank=configs)


def main():
    torch.manual_seed(1627)
    torch.set_num_threads(1)
    rng=np.random.default_rng(1627)
    pack=foundation()
    signals=Signals(pack)
    fit=dataset("fit",rows(HERE/"large.jsonl")[200:400],signals)
    held=dataset("held",rows(HERE/"reward-probe.jsonl"),signals)
    x,base,ok,wins,safe=fit
    mixed=np.flatnonzero((wins.any(1)&(ok&~wins).any(1)).numpy())
    assert len(mixed)>0
    net=OutcomeHead()
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.04)
    out=HERE.parent/"runs/paired200"
    out.mkdir(exist_ok=False)
    save(out/"config.json",dict(seed=1627,steps=[150,600],lr=.0008,temperature=2,trust=.12,
            source=digest(HERE/"terminal_fit.py"),implementation=digest(HERE/"terminal_model.py"),
            train=digest(HERE/"terminal-choices-fit.json"),selection=digest(HERE/"terminal-choices-held.json")))
    summary=[]
    for k in range(1,601):
        ids=np.concatenate((rng.integers(len(x),size=40),rng.choice(mixed,40)))
        feat,b,legal,win,alive=[t[ids] for t in fit]
        q=net(feat,legal)
        warm=(b/2).masked_fill(~legal,-1e5).log_softmax(1)
        logp=(b/2+q[:,:,0].clamp(-4,4)).masked_fill(~legal,-1e5).log_softmax(1)
        p=logp.exp()
        gain=(p*win).sum(1).mean()
        kl=(p*(logp-warm)).sum(1).mean()
        pair=win[:,:,None] & ~win[:,None,:] & legal[:,None,:]
        ranks=q[:,:,1,None]-q[:,None,:,1]
        loss=-gain+.12*kl-.16*F.logsigmoid(ranks[pair]).mean()
        loss=loss+.12*F.binary_cross_entropy_with_logits(q[:,:,1][legal],win[legal].float())
        loss=loss+.06*F.binary_cross_entropy_with_logits(q[:,:,2][legal],alive[legal].float())
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if k in (150,600):
            checkpoint=out/f"head{k}.pt"
            torch.save(net.state_dict(),checkpoint)
            score=evaluate_snapshots(net.eval(),held)
            summary.append(dict(step=k,score=score,sha256=digest(checkpoint)))
            print(k,score["rank"][0],flush=True)
            net.train()
    best=max(summary,key=lambda r:(r["score"]["rank"][0]["net"],-r["score"]["rank"][0]["lost"]))
    chosen=best["score"]["rank"][0]
    save(out/"counterfactual-selection.json",dict(history=summary,selected=best["step"],choice=chosen))
    pack["head"]=torch.load(out/f"head{best['step']}.pt",weights_only=True,map_location="cpu")
    pack["linear"]=torch.tensor(chosen["weights"],dtype=torch.float64)
    path=out/"rl.pt";torch.save(pack,path)
    sig=digest(path);dev=OUT/"data/dev.jsonl"
    save(out/"locked-before-dev.json",dict(weights_sha256=sig,dev_sha256=digest(dev),selection=chosen))
    start=time.monotonic()
    result,episodes=evaluate(load(path.read_bytes()),rows(dev))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(dev),summary=result,
         episodes=episodes,seconds=time.monotonic()-start))
    print(result,flush=True)


if __name__=="__main__":main()
