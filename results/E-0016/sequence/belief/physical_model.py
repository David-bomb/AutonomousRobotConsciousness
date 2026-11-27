"""Обучаемый последний слой над оценками восьми направлений одного текущего снимка."""
from io import BytesIO
import numpy as np
import torch
from maze.grid import destination
from maze.interface import Move
from ..alt_model import ExitNet,exits
from ..model import encode
from .features import features

MOVES=tuple(Move)


class Model:
    def __init__(self,base,tables):
        self.net=ExitNet().eval().requires_grad_(False)
        self.net.load_state_dict(base)
        self.prior=self.net.wall_prior.numpy().tobytes()
        self.tables=tables.numpy().tobytes()

    @torch.inference_mode()
    def directions(self,v):
        ok=np.array([destination(v.window,(1,1),m) is not None for m in MOVES])
        matrix=np.zeros((8,5),np.float64)
        # У единственного легального действия сравнивать направления не требуется.
        if ok.sum()==1:
            return matrix,ok
        x,mask=encode(v,self.prior)
        assert np.array_equal(ok,mask)
        e=exits(v,self.prior)
        raw=self.net.full(torch.from_numpy(x[None]),torch.from_numpy(e[None]),torch.from_numpy(ok[None]))[0][0].numpy()
        matrix[:,0]=(raw-raw[ok].max())/max(raw[ok].std(),.5)
        known=x[:,255:295].reshape(8,5,8)
        routes=np.where(known[:,:,6]>.5,known[:,:,4]*75,10000).min(1)
        possible=ok&(routes<10000)
        front=ok&(x[:,448]>.5)
        def relative(column,length,eligible):
            if eligible.any():
                matrix[eligible,column]=np.exp(-.6*(length[eligible]-length[eligible].min()))
        relative(1,routes,possible)
        relative(2,x[:,424]*50,front&~possible.any())
        slack=np.array([v.fuel-v.fuel_distance,v.health-v.repair_distance])
        if not (possible.any() or front.any()) and slack.min()<=8:
            by_action=features(v,self.tables)
            if v.health<v.fuel:
                slack=slack[::-1]
            urgency=int(slack.argmin())
            likelihood=by_action[:,urgency*6+2]
            for column,margin in ((3,2),(4,8)):
                matrix[:,column]=likelihood*np.clip((margin+1-slack.min())/2,0,1)
        return matrix,ok


def load(checkpoint):
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    torch.set_num_threads(1)
    model=Model(pack["base"],pack["tables"])
    weight=pack["linear"].numpy()
    class Controller:
        def choose(self,v):
            features,legal=model.directions(v)
            scores=features@weight
            return MOVES[int(np.argmax(np.where(legal,scores,-1e12)))]
    return Controller()
