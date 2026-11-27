from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import torch
from ...data import OUT,digest,read,rows,save
from ...dev import evaluate
from . import transfer
from .model import HERE
from .risk_model import RiskModel,load
from .terminal_fit import foundation


def package():
    pack=foundation()
    pack["foundation"]=torch.load(HERE.parent/"runs/field3000-clean/rl.pt",weights_only=True,map_location="cpu")
    return pack


def initialize():
    global net
    torch.set_num_threads(1)
    net=RiskModel(package())
    transfer.net=net


def offline(split):
    files=sorted((HERE.parent/"cache/terminal-choices"/split).glob("*.npz"))
    raw={}
    for p in files:
        with np.load(p) as f:
            for k in f:raw.setdefault(k,[]).append(f[k])
    d={k:torch.from_numpy(np.concatenate(v)) for k,v in raw.items()}
    feature=[]
    for i in range(0,len(d["last"]),64):
        feature.append(net.evidence(*(d[k][i:i+64] for k in ("gates","patch","meta","stocks","last","scores"))).numpy())
    return np.concatenate(feature),d["legal"].numpy(),d["safe"].numpy(),d["wins"].numpy()


def compare(data,choices):
    x,legal,safe,wins=data
    b=np.arange(len(x))
    original=np.where(legal,x[:,:,0],-1e12).argmax(1)
    results=[]
    for w in choices:
        a=np.where(legal,x@w,-1e12).argmax(1)
        gain=int((wins[b,a]&~wins[b,original]).sum())
        lost=int((~wins[b,a]&wins[b,original]).sum())
        results.append(dict(weights=w,gain=gain,lost=lost,net=gain-lost,
            safe_gain=int(safe[b,a].sum())-int(safe[b,original].sum()),changed=int((a!=original).sum())))
    return sorted(results,key=lambda r:(-r["net"],r["lost"],-r["safe_gain"],r["changed"]))


def distinct(ranked,n):
    seen=set();out=[]
    for r in ranked:
        key=(r["gain"],r["lost"],r["safe_gain"],r["changed"])
        if key not in seen:
            seen.add(key);out.append(r["weights"])
        if len(out)==n:break
    return out


def main():
    initialize()
    zero=[1]+[0]*10
    candidates=[zero]
    for mix in ((1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),
                (0,0,0,0,1),(.5,.5,0,0,0),(.5,0,1,0,0)):
        for k0 in (3,7,14,28):
            for ratio in (0,.5,1,2):
                candidates.append([1,*[k0*v for v in mix],*[k0*ratio*v for v in mix]])
    fit=compare(offline("fit"),candidates)
    held=compare(offline("held"),distinct(fit,28)+[zero])
    choices=distinct(held,4)
    if zero not in choices:choices.append(zero)
    out=HERE.parent/"runs/risk01"
    cache=HERE.parent/"cache/physical-risk01"
    out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
    config=dict(bank=digest(HERE/"risk-selection.jsonl"),foundation=digest(HERE.parent/"runs/field3000-clean/rl.pt"),
                fit=digest(HERE/"terminal-choices-fit.json"),held=digest(HERE/"terminal-choices-held.json"),
                source=digest(HERE/"risk_rl.py"),coefficients=choices)
    if (out/"config.json").exists():assert read(out/"config.json")==config
    else:save(out/"config.json",config)
    save(out/"preselection.json",dict(fit=fit,held=held))
    with ProcessPoolExecutor(2,initializer=initialize) as pool:
        result=transfer.stage(pool,rows(HERE/"risk-selection.jsonl"),choices,cache/"selection")
    save(out/"selection.json",result)
    best=max(result["rank"],key=lambda r:(r["wins"],r["minimum"]))
    pack=package();pack["linear"]=torch.tensor(best["weights"],dtype=torch.float64)
    path=out/"rl.pt"
    if path.exists():raise FileExistsError(path)
    torch.save(pack,path);sig=digest(path);dev=OUT/"data/dev.jsonl"
    save(out/"locked-before-dev.json",dict(weights_sha256=sig,selection=best,dev_sha256=digest(dev)))
    start=time.monotonic()
    summary,episodes=evaluate(load(path.read_bytes()),rows(dev))
    save(path.with_suffix(".physical.json"),dict(weights_sha256=sig,dev_sha256=digest(dev),summary=summary,
                episodes=episodes,seconds=time.monotonic()-start))
    print(best,summary,flush=True)


if __name__=="__main__":main()
