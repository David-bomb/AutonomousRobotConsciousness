"""Дообучение на расходе после хода и успешных поездках с разведкой."""
from pathlib import Path
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from ..data import OUT,read,rows,digest,save
from .bank import HERE
from .guard_model import GuardNet
from .train import dataset,checkpoint,paired_rl


def fit(net,regular,clean,meta,rng,opt,steps):
    critical=torch.nonzero((regular["legal"]&~regular["viable"]).any(1)).flatten().cpu().numpy()
    probs=meta["weight"]*meta["success"]
    clean_ids=rng.choice(len(probs),size=(steps,96),p=probs/probs.sum())
    target=torch.from_numpy(meta["target"]).cuda()
    losslog=[]
    for j in range(steps):
        reg_ids=np.concatenate((rng.integers(len(regular["x"]),size=72),rng.choice(critical,24)))
        u=torch.from_numpy(reg_ids).cuda()
        c=torch.from_numpy(clean_ids[j]).cuda()
        x,mask,viable,best=(torch.cat((regular[k][u],clean[k][c])) for k in ("x","legal","viable","best"))
        scores,_,risk,phase=net.parts(x.float(),mask)
        logp=scores.log_softmax(1)
        action=clean["partial"][c].long()
        imitation=-logp.masked_fill(~best,-1e6).logsumexp(1)
        has=best.any(1)
        imitation[96:]=torch.where(action>=0,-logp[96:].gather(1,action.clamp_min(0)[:,None]).flatten(),imitation[96:])
        # Когда наблюдаемая карта уже содержит проходимую до B последовательность,
        # её первый ход даёт дополнительную учебную метку. При оценке меток нет.
        known=x[:,:,255:295].float().reshape(len(x),8,5,8)
        costs=known[:,:,:,4].masked_fill(known[:,:,:,6]<.5,1e6).amin(2).masked_fill(~mask,1e6)
        minimum=costs.amin(1,keepdim=True)
        confirmed=minimum.flatten()<1e5
        suggested=(costs==minimum)&mask
        known_loss=-logp.masked_fill(~suggested,-1e6).logsumexp(1)
        imitation=torch.where(confirmed,.85*known_loss+.15*imitation,imitation)
        safety=F.binary_cross_entropy_with_logits(risk[mask],viable[mask].float())
        mission=F.cross_entropy(phase[96:],target[c])
        loss=imitation[has].mean()+.5*safety+.35*mission
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
        opt.step()
        losslog.append(float(loss.detach()))
    return dict(updates=steps,first_loss=float(np.mean(losslog[:10])),last_loss=float(np.mean(losslog[-10:])))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("init",type=Path)
    args=p.parse_args()
    seed=1612
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    prior=torch.load(args.init,weights_only=True,map_location="cpu")
    kind="rich" if "base.wall_prior" in prior else "critical" if "base.risk.0.weight" in prior else "plain"
    net=GuardNet(kind).cuda()
    miss,extra=net.load_state_dict(prior,strict=False)
    heads=("route_embed","patch_","risk_head","phase_next")
    assert not extra and all(k.startswith(heads) for k in miss)
    for name,p in net.named_parameters():
        p.requires_grad_(name.startswith(heads))
    folder=HERE/"runs"/args.name
    folder.mkdir(exist_ok=False,parents=True)
    save(folder/"config.json",dict(seed=seed,init=args.init.as_posix(),sha256=digest(args.init),updates=[1200,2000],
         regular="train-1600",teacher="clean-400",rl_groups=8,lr=.0006,
         sources={n:digest(HERE/n) for n in ("refine.py","guard_model.py","model.py","train.py","adapter.py","geometry.cpp",
         "sequence.dll","clean_routes.py","exploration_teacher.py","phase_index.py","prepare-train-1600.json","prepare-clean-400.json")}))
    regular=dataset([("train",1600)])
    clean=dataset([("clean",400)])
    probe=dataset([("probe",100)])
    dev=rows(OUT/"data/dev.jsonl")
    with np.load(HERE/"cache/clean-meta.npz") as f:
        meta={k:f[k] for k in f}
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.0006)
    scores=[]
    for tag,steps in (("s1200",1200),("s3200",2000)):
        t=time.monotonic()
        info=fit(net,regular,clean,meta,rng,opt,steps)
        info["seconds"]=time.monotonic()-t
        scores.append((checkpoint(net,folder,tag,dev,probe,info),tag))
    chosen=max(scores,key=lambda s:s[0])[1]
    net.load_state_dict(torch.load(folder/f"{chosen}.pt",weights_only=True,map_location="cuda"))
    del regular,clean,opt
    t=time.monotonic()
    info=paired_rl(net,rows(HERE/"train.jsonl"),rng,8,("route_embed","patch_","phase_next"))
    info.update(seconds=time.monotonic()-t,from_checkpoint=chosen)
    checkpoint(net,folder,"rl",dev,probe,info)


if __name__=="__main__":
    main()
