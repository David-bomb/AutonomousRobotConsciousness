import numpy as np
import torch
from torch.nn import functional as F
from ...data import digest, read, save
from .gates import HERE
from .gate_model import GateNet

CACHE = HERE.parent/"cache/gates"


def unpack(path):
    with np.load(path) as f:
        return [f[k] for k in ("x","witness","legal","down","safe","fast")]


@torch.inference_mode()
def measure(net, spec):
    total = tight = retained = reduced = 0
    for name in spec["parts"]:
        x, _, legal, down, safe, _ = unpack(CACHE/name)
        for i in range(0,len(x),128):
            s = slice(i,i+128)
            q = net(torch.from_numpy(x[s]).cuda().float())[1].cpu().numpy()
            lf = legal[s]
            hard = (lf & ~safe[s]).any(1)
            idx = np.arange(len(lf))
            a = np.where(lf, q[:,:,0], -1e6).argmax(1)
            b = np.where(lf, q[:,:,1], -1e6).argmax(1)
            total += len(idx)
            tight += int(hard.sum())
            reduced += int((hard & down[s][idx,a]).sum())
            retained += int((hard & safe[s][idx,b]).sum())
    return dict(states=total, critical=tight, critical_viable=retained, critical_downhill=reduced)


def main():
    rng=np.random.default_rng(1624)
    torch.manual_seed(1624)
    torch.set_num_threads(1)
    fit, held = read(HERE/"gate-data.json")["sets"]
    out = HERE/"gate-3000"
    out.mkdir(exist_ok=False)
    save(out/"config.json", dict(seed=1624, train_maps=3000, auxiliary_maps=200, lr=.001,
        source=digest(HERE/"gate_train.py"), input=digest(HERE/"gate-data.json"),
        network=digest(HERE/"gate_model.py"), steps=[1000,3600]))
    net = GateNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    history = []
    for step in range(1,3601):
        if step%11==1:
            data = unpack(CACHE/rng.choice(list(fit["parts"])))
            critical=np.flatnonzero((data[2] & ~data[4]).any(1))
        indices = rng.integers(len(data[0]), size=128)
        if len(critical):
            indices[:64]=rng.choice(critical,64)
        x,y,legal,down,safe,fast = [torch.from_numpy(z[indices]).cuda() for z in data]
        edge, action = net(x.float())
        present = x[...,0]>0
        labels = torch.stack((down,safe,fast),-1)
        bce = F.binary_cross_entropy_with_logits(action[legal], labels[legal].float())
        truth = labels & legal[...,None]
        choice = action.masked_fill(~legal[...,None],-1e5).log_softmax(1)
        rank = -choice.masked_fill(~truth,-1e5).logsumexp(1)[truth.any(1)].mean()
        loss = bce + .35*rank
        if present.any():
            loss = loss + .6*F.binary_cross_entropy_with_logits(edge[present], y[present].float(),
                pos_weight=torch.tensor(3.,device=edge.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
        opt.step()
        if step in (1000,3600):
            path=out/f"s{step}.pt"
            torch.save(net.state_dict(),path)
            score=measure(net.eval(),held)
            net.train()
            history.append(dict(step=step,sha256=digest(path),score=score))
            print(step,score,flush=True)
    best=max(history,key=lambda t:t["score"]["critical_viable"]+t["score"]["critical_downhill"])
    save(out/"result.json",dict(selected=best["step"],history=history))


if __name__=="__main__":
    main()
