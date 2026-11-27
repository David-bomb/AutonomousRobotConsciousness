from io import BytesIO
import numpy as np
import torch
from ..belief.features import probabilities
from ..belief.joint_model import MOVES
from .gate_model import GateNet
from .gates import describe
from .model import image
from .suffix import likelihood
from .transfer_model import FullModel


class GateAgent:
    def __init__(self,pack):
        self.base=FullModel(pack["field_policy"])
        self.base_weights=pack["field_policy"]["linear"].numpy()
        self.gate=GateNet().eval().requires_grad_(False)
        self.gate.load_state_dict(pack["gate"])
        with np.load(BytesIO(pack["suffix_archive"].numpy().tobytes())) as archive:
            self.counts=archive["counts"]

    @torch.inference_mode()
    def directions(self,view):
        prev,ok=self.base.directions(view)
        z=np.zeros((8,11),np.float64)
        z[:,0]=prev@self.base_weights
        if not prev[:,2:].any():return z,ok
        board,meta=image(view)
        stocks=(view.fuel,view.health)
        x,ref,u=describe(board,meta,stocks)
        edges,q=self.gate(torch.from_numpy(x[None]))
        p=edges[0].sigmoid().numpy()*x[...,0]
        prob=q[0].sigmoid().numpy()
        _,mix=probabilities(view.memory,self.base.previous.tables)
        stats=likelihood(ref,meta,u,mix,self.counts)
        # Суммарная частота для одного первого хода по оставшимся допустимым выходам.
        joint=p*stats[:,:,0]
        signals=(stats[:,:,0].max(1),stats[:,:,0].sum(1),joint.max(1),joint.sum(1),p.max(1))
        z[:,1:4]=prob
        for i,s in enumerate(signals,4):
            z[:,i]=s/(s[ok].max()+1e-7)
        z[:,9]=prev[:,2]
        z[:,10]=prev[:,3]
        strength=np.clip((10-min(stocks-meta[6:8]))/5.,0.,1.)
        z[:,1:9]*=strength
        return z,ok


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=GateAgent(pack)
    w=pack["linear"].numpy()

    class Controller:
        def choose(self,view):
            x,ok=net.directions(view)
            return MOVES[int(np.where(ok,x@w,-1e12).argmax())]

    return Controller()
