"""Односнимковая оценка восьми ходов с запасами на продолжении разведки."""
from io import BytesIO
import numpy as np
import torch
from maze.grid import destination
from maze.interface import Move
from ..alt_model import ExitNet,exits
from ..model import encode
from .features import features
from .itineraries import measures
from .safety_net import Safety

MOVES=tuple(Move)


class Joint:
    def __init__(self,pack):
        self.net=ExitNet().eval().requires_grad_(False)
        self.net.load_state_dict(pack["base"])
        self.safety=Safety().eval().requires_grad_(False)
        self.safety.load_state_dict(pack["safety"])
        self.prior=self.net.wall_prior.numpy().tobytes()
        self.tables=pack["tables"].numpy().tobytes()

    @torch.inference_mode()
    def directions(self,v):
        ok=np.array([destination(v.window,(1,1),m) is not None for m in MOVES])
        out=np.zeros((8,10),np.float64)
        if ok.sum()==1:
            return out,ok
        x,mask=encode(v,self.prior)
        assert np.array_equal(mask,ok)
        ex=exits(v,self.prior)
        xt,et,kt=(torch.from_numpy(val[None]) for val in (x,ex,ok))
        raw=self.net.full(xt,et,kt)[0][0].numpy()
        out[:,0]=(raw-raw[ok].max())/max(raw[ok].std(),.5)
        def evidence(col,ds,valid):
            if valid.any():out[valid,col]=np.exp(-.6*(ds[valid]-ds[valid].min()))
        known=x[:,255:295].reshape(8,5,8)
        lengths=np.where(known[:,:,6]>.5,known[:,:,4]*75,10000).min(1)
        goal=ok&(lengths<10000)
        front=ok&(x[:,448]>.5)
        evidence(1,lengths,goal)
        evidence(2,x[:,424]*50,front&~goal.any())
        if goal.any() or front.any():
            return out,ok
        sites=x[:,:110].reshape(8,5,22)
        connected=[(sites[ok,j:j+2,11]<.5).any() for j in (1,3)]
        current=v.memory[v.position[1]][v.position[0]]
        if current in "FR":
            index=int((current=="R")!=(v.health<v.fuel))
            connected[index]=True
        if any(connected):
            trips=measures(v,self.prior)
            # Числовые длины при трёх предположениях о неизвестной дороге.
            # Для каждой гарантии возврата сохраняем первое представление,
            # в котором присутствует хотя бы одно посильное продолжение до B.
            for category,col in ((1,3),(0,4)):
                for mode in (2,1,0):
                    ds=trips[:,mode,category,0]
                    available=ok&(ds<300)
                    if available.any():
                        evidence(col if all(connected) else col+2,ds,available)
                        break
        slack=np.array([v.fuel-v.fuel_distance,v.health-v.repair_distance])
        if slack.min()<=10:
            walk=features(v,self.tables)
            g,r=self.safety(xt,et,torch.from_numpy(walk[None]),kt)
            if v.health<v.fuel:slack=slack[::-1]
            urgent=int(slack.argmin())
            strength=np.clip((10.-slack.min())/5.,0.,1.)
            out[:,7]=strength*g[0,:,urgent].softmax(-1)[:,0].numpy()
            out[:,8]=strength*r[0].sigmoid().numpy()
            out[:,9]=np.clip((5.-slack.min())/3.,0.,1.)*walk[:,urgent*6+2]
        return out,ok


def load(checkpoint):
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    torch.set_num_threads(1)
    network=Joint(pack)
    w=pack["linear"].numpy()
    class Controller:
        def choose(self,v):
            x,ok=network.directions(v)
            return MOVES[int(np.where(ok,x@w,-1e12).argmax())]
    return Controller()
