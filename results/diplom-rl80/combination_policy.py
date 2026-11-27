"""Load a single frozen choice of early and late eight-action scoring heads."""
from io import BytesIO
import torch
from departure_policy import Departure
from exact_policy import ExactNet
from lookahead_policy import Lookahead
from protected_policy import Protected


class Combination:
    def __init__(self,pack,ablate=False):
        cls={"protected-tree-morph-1":Protected,"exact-mission-1":ExactNet,"partial-two-step-1":Lookahead}
        self.early=cls[pack["early_format"]](pack,ablate)
        self.late=None
        if pack["use_departure"] and pack["departure"].any():
            # Construct the trained late expert, then share its encoder with the early expert.
            # Both came from the same frozen joint400 and the pack stores its tensors only once.
            self.late=Departure(pack,ablate)
            self.late.b=self.early.b
        self.ablate=ablate
        self.previous=(None,None)

    def choose(self,v,cache=None):
        if v.last_move is None:
            self.previous=(None,None)
        old=(None,None) if self.ablate else self.previous
        key=v,old
        if cache is not None and key in cache:
            action=cache[key]
        else:
            if self.early.route(v,self.early.cfg):
                if self.late is None:
                    q=self.early.late_scores(v)
                else:
                    q,x=self.late.late_parts(v,old)
                    q=q+x@self.late.weights
            else:
                q=self.early.early_scores(v,old)
            action=self.early.early.MOVES[int(q.argmax())]
            if cache is not None:
                cache[key]=action
        self.previous=v.fuel_distance,v.repair_distance
        return action


def load(blob,ablate=False):
    torch.set_num_threads(1)
    p=torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if p["format"]!="combined-departure-1":
        raise ValueError("Unknown combined policy")
    return Combination(p,ablate)
