from io import BytesIO
import numpy as np
import torch
from .gates import describe
from .local_model import local_image
from .model import image,MOVES
from .terminal_model import Signals
from .transfer_model import FullModel


class RiskModel:
    def __init__(self,pack):
        self.base=FullModel(pack["foundation"])
        self.ranking=pack["foundation"]["linear"].numpy()
        self.nets=Signals(pack)

    @torch.inference_mode()
    def evidence(self,gates,patch,meta,stock,last,raw):
        """Вероятность оплатить следующий шаг до достижимого сервиса по распределению предсказанных знаков."""
        _,g=self.nets.gate(gates.float())
        sig,_,_=self.nets.local(patch,meta,stock,last)
        slack=stock-meta[:,6:]
        allowed=torch.arange(-1,2)[None,None,None,:] <= slack[:,None,:,None]-1
        chance=(sig.softmax(-1)[:,:,:2,:]*allowed).sum(-1).amin(2)
        risk=1-chance
        learned=1-g[:,:,1].sigmoid()
        down=1-g[:,:,0].sigmoid()
        different=torch.stack((risk,learned,(torch.minimum(risk,learned)-.25).relu(),
                               ((risk*learned).sqrt()-.4).relu(),(risk*down-.3).relu()),2)
        minimum=slack.amin(1)
        return torch.cat((raw[:,:,None],-different*(minimum==0)[:,None,None],
                          -different*(minimum==1)[:,None,None]),2)

    def directions(self,v):
        prev,legal=self.base.directions(v)
        score=prev@self.ranking
        z=np.zeros((8,11),np.float64)
        z[:,0]=score
        if legal.sum()<2 or min(v.fuel-v.fuel_distance,v.health-v.repair_distance) not in (0,1):
            return z,legal
        board,meta=image(v)
        stock=np.array([v.fuel,v.health],np.int16)
        g,_,_=describe(board,meta,stock)
        patch,_=local_image(v)
        last=np.asarray(MOVES.index(v.last_move) if v.last_move else 8)
        v0=self.evidence(*(torch.from_numpy(a[None]) for a in (g,patch,meta,stock,last,score.astype(np.float32))))
        z[:,1:]=v0[0,:,1:].numpy()
        return z,legal


def load(checkpoint):
    torch.set_num_threads(1)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    net=RiskModel(pack)
    w=pack["linear"].numpy()

    class Controller:
        def choose(self,v):
            z,ok=net.directions(v)
            return MOVES[int(np.where(ok,z@w,-1e12).argmax())]

    return Controller()
