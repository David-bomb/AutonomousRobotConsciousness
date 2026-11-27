"""Пространственный прогноз раннего движения и единый обученный выход на восемь команд."""
from io import BytesIO
import numpy as np
import torch
from maze.interface import Move
from ..belief.joint_model import Joint
from ..belief.paths import confidence
from ...policy import geometry
from .field_model import FieldNet
from .model import image

MOVES=tuple(Move)


class FullModel:
    def __init__(self,pack):
        self.previous=Joint(pack["foundation"])
        self.previous_weights=pack["foundation"]["linear"].numpy()
        self.field=FieldNet().eval().requires_grad_(False)
        self.field.load_state_dict(pack["field"])

    @torch.inference_mode()
    def directions(self,v):
        old,ok=self.previous.directions(v)
        out=np.zeros((8,8),np.float64)
        out[:,0]=old@self.previous_weights
        if ok.sum()==1 or old[:,1:5].any():
            return out,ok
        slack=np.array([v.fuel-v.fuel_distance,v.health-v.repair_distance])
        if slack.min()>10:return out,ok
        urgent=int(slack.argmin())
        intensity=np.clip((10.-slack.min())/4,0,1)
        out[:,1]=old[:,7:]@self.previous_weights[7:]
        board,meta=image(v)
        last=MOVES.index(v.last_move) if v.last_move is not None else 8
        scores,_,legal=self.field(torch.from_numpy(board[None]),torch.from_numpy(meta[None]),
                  torch.tensor([[v.fuel,v.health]]),torch.tensor([last]))
        assert np.array_equal(ok,legal[0].numpy())
        prob=scores[0].sigmoid().numpy()
        out[:,2]=intensity*prob[:,urgent]
        out[:,3]=intensity*prob[:,3]
        out[:,4]=intensity*prob[:,4]
        possibilities=confidence(v,self.previous.tables,48)
        out[:,5]=intensity*possibilities[:,1]
        goal=v.position[0]+v.dx,v.position[1]+v.dy
        _,raw=geometry(v.memory,v.position,goal)
        kind=1 if urgent==0 else 3
        exact=raw[:,:15].reshape(8,5,3)[:,kind:kind+2,0].min(1)
        d=(v.fuel_distance,v.repair_distance)[urgent]
        out[:,6]=intensity*(exact==d)
        out[:,7]=intensity*(possibilities[:,0]>0)
        return out,ok


def load(checkpoint):
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    torch.set_num_threads(1)
    net=FullModel(pack)
    w=pack["linear"].numpy()
    class Controller:
        def choose(self,v):
            x,ok=net.directions(v)
            return MOVES[int(np.where(ok,x@w,-1e12).argmax())]
    return Controller()
