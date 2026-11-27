import numpy as np
import torch
from torch.nn import functional as F

from ....data import digest, read, save
from .bank import HERE, CACHE
from .model import Decision, FIELDS


def dataset(split):
    parts = []
    arrays = (*FIELDS, "options", "label", "sensitive", "viable", "previous", "step")
    reg = read(HERE/(split+"-data.json"))
    for name, sig in reg["files"].items():
        path = CACHE/split/(name+".npz")
        assert digest(path) == sig
        with np.load(path) as f:
            parts.append({k:f[k] for k in arrays})
    d = {k:torch.from_numpy(np.concatenate([r[k] for r in parts])).cuda() for k in arrays}
    paired = d["label"] < 9
    s = d["meta"][paired,6:] - d["previous"][paired]
    assert (d["label"][paired] == (s[:,0]+1)*3+s[:,1]+1).all()
    assert (d["previous"][~paired] == -1).all() and (d["step"][~paired] == 0).all()
    return d


def targets(d, ix):
    label = d["label"][ix]
    q = d["options"][ix,label.clamp_max(8)]
    return torch.where((label == 9)[:,None], d["missing"][ix], q)


@torch.inference_mode()
def audit(net, d, prior):
    post, direct = [], []
    for ix in torch.arange(len(d["label"]), device="cuda").split(256):
        p, a = net(*(d[k][ix] for k in FIELDS))
        post.append(p)
        direct.append(a)
    post = torch.cat(post)
    direct = torch.cat(direct)
    label = d["label"]
    b = torch.arange(len(label), device="cuda")
    truth = targets(d,b).argmax(1)
    modal = post.argmax(1)
    allowed = d["allowed"]
    prior_pick = torch.as_tensor(prior,device="cuda")[None,:].expand_as(post).masked_fill(~allowed,-1).argmax(1)
    marginal = torch.bmm(post.softmax(-1)[:,None,:], d["options"])[:,0]
    actions = dict(missing=d["missing"].argmax(1), marginal=marginal.argmax(1),
                   direct=direct.argmax(1), modal=d["options"][b,modal].argmax(1), true_pair=truth)
    slack=(d["stock"]-d["meta"][:,6:]).amin(1)
    groups={}
    for name,use in (("nonstart",label<9),("sensitive",d["sensitive"]&(label<9)),
                     ("sensitive_slack01",d["sensitive"]&(label<9)&(slack<=1)),
                     ("sensitive_critical",d["sensitive"]&(label<9)&(d["legal"]&~d["viable"]).any(1)&d["viable"].any(1))):
        y=label[use]
        pred=modal[use]
        conf=torch.bincount(y*9+pred,minlength=81).reshape(9,9)
        g=dict(samples=int(use.sum()), joint=int((pred==y).sum()), fuel=int((pred//3==y//3).sum()),
               repair=int((pred%3==y%3).sum()), top2=int((post[use].topk(2,1).indices==y[:,None]).any(1).sum()),
               admissible_frequency=int((prior_pick[use]==y).sum()), unfiltered_frequency=int((np.argmax(prior)==y).sum()),
               negative_log_likelihood=float(F.cross_entropy(post[use],y)),
               singleton=int((allowed[use].sum(1)==1).sum()), confusion=conf.tolist(), actions={})
        viable_rows = use & d["viable"].any(1)
        g["viable_states"] = int(viable_rows.sum())
        for method, selected in actions.items():
            x = selected[use]
            v = d["viable"][b,selected]
            m = actions["missing"]
            g["actions"][method] = dict(agrees_with_true_pair=int((x==truth[use]).sum()),
                 viable=int(v[viable_rows].sum()), gains_over_missing=int(((selected==truth)&(m!=truth)&use).sum()),
                 losses_to_missing=int(((selected!=truth)&(m==truth)&use).sum()),
                 confusion=torch.bincount(truth[use]*8+x,minlength=64).reshape(8,8).tolist())
        groups[name]=g
    return groups


def main():
    torch.manual_seed(20411)
    rng=np.random.default_rng(20411)
    train, held = dataset("fit"), dataset("held")
    net=Decision().cuda()
    net.geom.load_state_dict(torch.load(HERE/"spatial.pt",map_location="cuda",weights_only=True))
    opt=torch.optim.AdamW([dict(params=net.geom.parameters(),lr=.00012),
                           dict(params=list(net.enc.parameters())+list(net.pair.parameters())+list(net.move.parameters()),lr=.00065)])
    nonstart=torch.where(train["label"]<9)[0].cpu().numpy()
    sensitive=torch.where(train["sensitive"])[0].cpu().numpy()
    prior=torch.bincount(train["label"][train["label"]<9],minlength=9).tolist()
    for step in range(850):
        ix=np.r_[rng.choice(nonstart,160),rng.choice(sensitive,80)]
        p, direct=net(*(train[k][ix] for k in FIELDS))
        q=targets(train,ix)
        # Смесь полного распределения предпочтений и правильного физического хода.
        soft=F.kl_div((direct/1.5).log_softmax(1),(q/1.5).softmax(1),reduction="batchmean")
        loss=F.cross_entropy(p,train["label"][ix])+.65*F.cross_entropy(direct,q.argmax(1))+.7*soft
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
        opt.step()
    groups=audit(net.eval(),held,prior)
    out=HERE/"decision.pt"
    torch.save(net.cpu().state_dict(),out)
    save(HERE/"held.json",dict(seed=20411,updates=850,weights=digest(out),fit=digest(HERE/"fit-data.json"),
         held=digest(HERE/"held-data.json"),prior=prior,source=digest(HERE/"fit.py"),
         spatial=digest(HERE/"spatial.pt"),class_order="3*(currentF-previousF+1)+(currentR-previousR+1)",groups=groups))
    print({k:(v["samples"],v["joint"],{s:p["agrees_with_true_pair"] for s,p in v["actions"].items()}) for k,v in groups.items()},flush=True)


if __name__ == "__main__":
    main()
