from io import BytesIO
import numpy as np
import torch

from .model import Decision, FIELDS, Policy
from .features import MOVES, frame, phase


@torch.inference_mode()
def outputs(net, d):
    p, direct=net(*(torch.as_tensor(d[k])[None] for k in FIELDS))
    post=p.softmax(1)[0]
    options=torch.as_tensor(d["options"])
    mean=(post@options).numpy()
    votes=(post@options.masked_fill(~torch.as_tensor(d["legal"])[None], -1e6).softmax(1)).numpy()
    return mean,direct[0].numpy(),votes


class ProbabilityPolicy(Policy):
    @torch.inference_mode()
    def choose(self, v):
        if phase(v.memory):
            x,ok=self.late.directions(v)
            q=x@self.late_w
        else:
            d=frame(self.early,v);ok=d["legal"]
            q=d["missing"] if v.last_move is None or ok.sum()==1 else outputs(self.net,d)[2]
        return MOVES[int(np.where(ok,q,-np.inf).argmax())]


def load(data):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(data),map_location="cpu",weights_only=True)
    cls=ProbabilityPolicy if pack.get("rule")=="probability" else Policy
    return cls(pack)


class Comparison(Policy):
    """Общий расчёт снимка для четырёх вариантов физического отбора на train."""
    def __init__(self, baseline, updated):
        super().__init__(baseline)
        self.updated=Decision().eval().requires_grad_(False)
        self.updated.load_state_dict(updated)

    @torch.inference_mode()
    def directions(self,v):
        if phase(v.memory):
            x,ok=self.late.directions(v)
            q=x@self.late_w
            return np.tile(q[:,None],(1,4)),ok
        d=frame(self.early,v);ok=d["legal"]
        if v.last_move is None or ok.sum()==1:
            return np.tile(d["missing"][:,None],(1,4)),ok
        mean,direct,_=outputs(self.net,d)
        old=np.stack((d["missing"],mean,direct),1)@np.asarray(self.mode)
        mean,direct,votes=outputs(self.updated,d)
        return np.stack((old,mean,direct,votes),1),ok
