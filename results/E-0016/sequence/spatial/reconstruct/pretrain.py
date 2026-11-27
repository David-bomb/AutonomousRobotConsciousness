import numpy as np
import torch
from torch.nn import functional as F

from ....data import digest, read, save
from ..local_data import crop
from .bank import HERE, CACHE
from .features import REVERSE
from .model import SpatialPair


def main():
    torch.manual_seed(20377)
    rng = np.random.default_rng(20377)
    spec = read(HERE.parent / "early.json")["sets"][0]
    chunks = []
    for name, expected in spec["parts"].items():
        path = CACHE.parent/"spatial"/name
        assert digest(path) == expected
        with np.load(path) as f:
            m, last, y = f["meta"], f["last"], f["y"]
            idx = np.flatnonzero(last < 8)
            back = np.asarray(REVERSE)[last[idx]]
            s = 2 - y[idx,back,:2]
            assert ((s >= 0) & (s <= 2)).all()
            labels = 3*s[:,0] + s[:,1]
            chunks.append((crop(f["maps"][idx], m[idx]), m[idx], last[idx], f["stocks"][idx], labels))
    arrays = [torch.as_tensor(np.concatenate(values), device="cuda") for values in zip(*chunks)]
    net = SpatialPair().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    n = len(arrays[0])
    counts = torch.bincount(arrays[-1].long()).tolist()
    for step in range(1600):
        ix = rng.integers(n, size=256)
        patch, meta, last, stock, y = [a[ix] for a in arrays]
        loss = F.cross_entropy(net(patch,meta,last,stock), y.long())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
        optimizer.step()
        if (step+1)%400 == 0:
            print(step+1, float(loss.detach()), flush=True)
    dest = HERE/"spatial.pt"
    torch.save(net.cpu().state_dict(), dest)
    save(HERE/"pretrain.json", dict(seed=20377, updates=1600, states=n, classes=counts,
         weights=digest(dest), source=digest(HERE/"pretrain.py"), archive=digest(HERE.parent/"early.json")))


if __name__ == "__main__":
    main()
