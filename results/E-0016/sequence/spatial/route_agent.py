from io import BytesIO
import numpy as np
import torch
from .landmarks import landmarks
from .landmark_model import LandmarkNet,FIRST
from .model import image,MOVES
from .transfer_model import FullModel


class RouteAgent:
    def __init__(self,pack):
        self.base=FullModel(pack["foundation"])
        self.w=pack["foundation"]["linear"].numpy()
        self.routes=LandmarkNet().eval().requires_grad_(False)
        self.routes.load_state_dict(pack["routes"])

    @torch.inference_mode()
    def directions(self,v):
        prev,ok=self.base.directions(v)
        z=np.zeros((8,10),np.float64)
        z[:,0]=prev@self.w
        if ok.sum()<2:return z,ok
        stock=np.array([v.fuel,v.health],np.int16)
        slack=stock-np.array([v.fuel_distance,v.repair_distance])
        if slack.min()>12:return z,ok
        board,meta=image(v)
        pair,act=landmarks(board,meta)
        if pair[0,5,2] and round(pair[0,5,1]*127)<=stock.min():return z,ok
        known=min(pair[0,1:3,6].max(),pair[0,3:5,6].max())
        if known>=1 and not prev[:,2:].any():return z,ok
        last=torch.tensor([MOVES.index(v.last_move) if v.last_move else 8])
        q,rs,dl,sg,legal=self.routes(torch.from_numpy(pair[None]),torch.from_numpy(act[None]),
            torch.from_numpy(meta[None]),torch.from_numpy(stock[None]),last)
        assert np.array_equal(legal[0].numpy(),ok)
        confidence=q[0].sigmoid().numpy()
        probs=rs[0].sigmoid().numpy()
        z[:,1:4]=confidence
        z[:,4]=probs.max(1)
        u=int(slack.argmin());first=np.array(FIRST)
        z[:,5]=probs[:,(first<3) if u==0 else (first>=3)].max(1)
        # Какая конкретная станция вероятнее даёт текущую минимальную длину.
        # Сравнение использует только распределения, предсказанные для имеющегося снимка.
        indices=[1+2*u,2+2*u]
        near=dl[0,0,indices].softmax(-1)[:,min(127,int(meta[6+u]))]
        near=(near+1e-8)/(near.sum()+2e-8)
        slope=sg[0,:,2*u:2*u+2].softmax(-1)
        z[:,6]=(slope[:,:,0]*near[None]).sum(1).numpy()
        if slack[u] in (0,1):
            safe_step=slope[:,:,:int(slack[u])+1].sum(2)
            z[:,7]=(safe_step*near[None]).sum(1).numpy()
            z[:,8]=np.clip(q[0,:,0].numpy()-q[0,ok,0].numpy().max(),-6,0)
        z[:,9]=np.clip(q[0,:,1].numpy()-q[0,ok,1].numpy().max(),-6,0)
        z[:,1:7]*=np.clip((13.-slack.min())/7,0,1)
        return z,ok


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=RouteAgent(pack);w=pack["linear"].numpy()

    class Controller:
        def choose(self,v):
            z,ok=net.directions(v)
            return MOVES[int(np.where(ok,z@w,-1e12).argmax())]

    return Controller()
