"""Предсказание пары с явно ограниченным входом. B доступна только аппарату движения."""
from dataclasses import replace
from io import BytesIO
import numpy as np
import torch
from torch import nn

from maze.grid import destination
from .features import legacy, MOVES, frame, phase
from .model import Decision, SpatialPair, Policy
from ..local_model import local_image


def restricted_inputs(view):
    # Фиксированная цель в текущей клетке устраняет зависимость от штатного направления к B.
    v = replace(view, dx=0, dy=0)
    legal = np.array([destination(v.memory, v.position, m) is not None for m in MOVES])
    past, bounds = legacy.memory_fields(v, (None,None), legal)
    order = [1,0] if v.health < v.fuel else [0,1]
    signals = past[:,order,8:].reshape(8,16).astype(np.float16)
    patch, meta = local_image(v)
    if v.last_move is None:
        allowed = np.zeros(9,bool); last=8
    else:
        from .features import DELTAS, REVERSE
        last=MOVES.index(v.last_move)
        if not v.moved:
            allowed=np.arange(9)==4
        else:
            feasible=bounds[REVERSE[last],order]
            allowed=np.all(feasible[np.arange(2),1-DELTAS],axis=1)
    return signals,patch,meta,np.int64(last),np.array([v.fuel,v.health],np.int16),allowed,legal


class Sensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.geom=SpatialPair()
        self.direction=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.predict=nn.Sequential(nn.Linear(73,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,9))

    def forward(self, signals, patch, meta, last, stocks, allowed, legal):
        geo=self.geom(patch,meta,last,stocks)
        h=self.direction(signals.float())
        back=self.geom.reverse[last.long()].clamp_max(7)
        context=h.masked_fill(~legal[:,:,None],-1e4).amax(1)
        part=h[torch.arange(len(h),device=h.device),back]
        logits=geo+self.predict(torch.cat((geo/4,context,part),1))
        return logits.masked_fill(~allowed,-1e4)


class StrictPolicy(Policy):
    def __init__(self,pack):
        super().__init__(pack)
        self.sensor=Sensor().eval().requires_grad_(False)
        self.sensor.load_state_dict(pack["sensor"])

    @torch.inference_mode()
    def directions(self,v):
        if phase(v.memory):
            x,ok=self.late.directions(v)
            return np.tile((x@self.late_w)[:,None],(1,3)),ok
        d=frame(self.early,v);ok=d["legal"];missing=d["missing"]
        if v.last_move is None or ok.sum()==1:return np.tile(missing[:,None],(1,3)),ok
        args=[torch.as_tensor(a)[None] for a in restricted_inputs(v)]
        p=self.sensor(*args)[0].softmax(0).numpy()
        assert np.array_equal(args[5][0].numpy(),d["allowed"])
        # Прямой выбор базового эксперимента. Предсказание пары из общей головы здесь не вычисляется.
        x=torch.from_numpy(d["x"])[None]
        h=self.net.enc(x)
        ctx=h.masked_fill(~args[-1][:,:,None],-1e4).amax(1)
        g=self.net.geom(*args[1:5])
        residual=self.net.move(torch.cat((h,ctx[:,None].expand_as(h),g[:,None].expand(-1,8,-1)/4),-1))[0,:,0]
        direct=missing+residual.numpy()
        return np.stack((missing,p@d["options"],direct),1),ok


def load(data):
    torch.set_num_threads(1)
    return StrictPolicy(torch.load(BytesIO(data),weights_only=True,map_location="cpu"))
