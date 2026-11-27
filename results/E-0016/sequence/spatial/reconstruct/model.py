from io import BytesIO

import numpy as np
import torch
from torch import nn

from ...belief.joint_model import Joint
from .features import MOVES, REVERSE, early_model, frame, phase


class SpatialPair(nn.Module):
    """Окружение обратного хода в двух отражённых представлениях."""
    def __init__(self):
        super().__init__()
        rotations = np.array([[0, -1], [1, 0]])
        offsets = np.array([(x, y) for y in range(-6, 7) for x in range(-6, 7)])
        transforms, indices = [], []
        for a in range(8):
            rotation = np.linalg.matrix_power(rotations, a % 4)
            reflection = np.diag([-1, 1]) if a < 4 else np.array([[0, -1], [-1, 0]])
            matrices = np.stack((rotation, rotation @ reflection))
            xy = offsets @ matrices.transpose(0, 2, 1)
            transforms.append(matrices)
            indices.append((xy[..., 1]+6)*13 + xy[..., 0]+6)
        self.register_buffer("coordinates", torch.tensor(np.stack(transforms), dtype=torch.float32))
        self.register_buffer("indices", torch.tensor(np.stack(indices)))
        self.register_buffer("reverse", torch.tensor(REVERSE))
        self.road = nn.Sequential(nn.Conv2d(4, 16, 3), nn.SiLU(), nn.Conv2d(16, 20, 3, stride=2),
                                  nn.SiLU(), nn.Conv2d(20, 24, 3), nn.SiLU(), nn.Flatten(), nn.Linear(216, 48), nn.SiLU())
        self.station = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 20), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(48+40+7, 96), nn.LayerNorm(96), nn.SiLU(),
                                     nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 9))

    def forward(self, patch, meta, last, stock):
        b = len(meta)
        back = self.reverse[last.long()].clamp_max(7)
        flat = patch.flatten(2)
        ix = self.indices[back]
        flat = flat[:, None].expand(-1, 2, -1, -1).gather(3, ix[:, :, None].expand(-1, -1, 2, -1))
        floor, owner = flat.unbind(2)
        territory = (owner.long() >> back[:, None, None]) & 1
        road = torch.stack((floor == 1, floor == 2, owner != 0, territory.bool()), 2).float()
        scene = self.road(road.reshape(b*2,4,13,13)).reshape(b,2,48)
        loc = torch.stack((meta[:,:5]%40, meta[:,:5]//40), -1).float()
        delta = loc[:,1:] - loc[:,:1]
        rel = torch.einsum("bsk,bgkj->bgsj", delta, self.coordinates[back])
        man = rel.abs().sum(-1)
        cheb = rel.abs().amax(-1)
        d = meta[:,6:].float().repeat_interleave(2,1)[:,None].expand_as(man)
        station = self.station(torch.stack((rel[...,0]/30, rel[...,1]/30, man/40,
                                            cheb/30, d/45, (d-man)/20, (d-cheb)/20), -1))
        stations = torch.cat((station[:,:,:2].amax(2), station[:,:,2:].amax(2)), -1)
        stocks = stock.float()
        resources = torch.cat((stocks/100, (stocks-meta[:,6:]).clamp(-20,40)/20,
                               loc[:,0]%2, (back>=4).float()[:,None]), 1)
        return self.readout(torch.cat((scene, stations, resources[:,None].expand(-1,2,-1)), -1)).mean(1)


class Decision(nn.Module):
    def __init__(self):
        super().__init__()
        self.geom = SpatialPair()
        self.enc = nn.Sequential(nn.Linear(266,72), nn.LayerNorm(72), nn.SiLU(), nn.Linear(72,48), nn.SiLU())
        self.pair = nn.Sequential(nn.Linear(96+9,64), nn.SiLU(), nn.Linear(64,9))
        self.move = nn.Sequential(nn.Linear(96+9,64), nn.SiLU(), nn.Linear(64,1))

    def forward(self, x, patch, meta, last, stock, allowed, legal, missing):
        g = self.geom(patch, meta, last, stock)
        h = self.enc(x.float())
        ctx = h.masked_fill(~legal[:,:,None], -1e4).amax(1)
        b = torch.arange(len(x), device=x.device)
        back = self.geom.reverse[last.long()].clamp_max(7)
        head = self.pair(torch.cat((ctx, h[b,back], g/4), 1)) + g
        head = head.masked_fill(~allowed, -1e4)
        # Прямой выход обучается по правильному ходу. Его входы целиком вычисляются по текущему снимку.
        direct = self.move(torch.cat((h, ctx[:,None].expand_as(h), g[:,None].expand(-1,8,-1)/4), -1)).squeeze(-1)
        return head, (missing+direct).masked_fill(~legal, -1e4)


FIELDS = ("x", "patch", "meta", "last", "stock", "allowed", "legal", "missing")


class Policy:
    def __init__(self, pack):
        self.early = early_model(pack["foundation"]["early"])
        self.late = Joint(pack["foundation"]["late"])
        self.late_w = pack["foundation"]["late"]["linear"].numpy()
        self.net = Decision().eval().requires_grad_(False)
        self.net.load_state_dict(pack["decision"])
        self.mode = pack["mode"]

    @torch.inference_mode()
    def directions(self, view):
        if phase(view.memory):
            values, ok = self.late.directions(view)
            q = values @ self.late_w
            return np.stack((q, q, q), 1), ok
        d = frame(self.early, view)
        q = d["missing"]
        if view.last_move is None or d["legal"].sum() == 1:
            return np.stack((q, q, q), 1), d["legal"]
        args = [torch.as_tensor(d[k])[None] for k in FIELDS]
        p, action = self.net(*args)
        marginal = p[0].softmax(0).numpy() @ d["options"]
        return np.stack((q, marginal, action[0].numpy()), 1), d["legal"]

    @torch.inference_mode()
    def choose(self, view):
        if self.mode[0] == 1 and not phase(view.memory):
            from .features import legacy
            values = self.early.features(view, (None, None))
            return MOVES[int(self.early(*legacy.tensor_batch([values])).argmax())]
        q, legal = self.directions(view)
        return MOVES[int(np.where(legal, q @ np.asarray(self.mode), -np.inf).argmax())]


def load(data):
    torch.set_num_threads(1)
    return Policy(torch.load(BytesIO(data), weights_only=True, map_location="cpu"))
