"""A fixed action network using observed-road probabilities as per-action input features."""
from io import BytesIO
import site
import sys
import numpy as np
import torch
from service_window_policy import Mixture,coverage

# The installed numerical geometry routines use SciPy from this Python's user site.
sys.path.append(site.getusersitepackages())
from maze.grid import locations,destination
from maze.mandatory_supply import MandatoryProof
from maze.proven_approach import certified


class VoteMixture(Mixture):
    def __init__(self,pack,ablate=False):
        super().__init__(pack,ablate)
        self.residual=pack["residual"].numpy()

    def early_parts(self,v,old):
        raw=super().early_scores(v,old)
        legal=np.array([destination(v.window,(1,1),m) is not None for m in self.early.MOVES])
        q=(raw-raw[legal].max())/max(.5,float(raw[legal].std()))
        parts=np.zeros((8,10))
        if legal.sum()<2:return q,parts
        readings=(v.fuel_distance,v.repair_distance)
        stocks=(v.fuel,v.health)
        active=[i for i in (0,1) if readings[i]>0]
        if not active:return q,parts
        kind=min(active,key=lambda i:stocks[i]-readings[i])
        slack=stocks[kind]-readings[kind]
        if slack>10:return q,parts
        x,y=v.position
        prior=(x-v.last_move.value[0],y-v.last_move.value[1]) if v.moved else None
        sites=tuple(tuple(locations(v.memory,c)) for c in "FR")
        sure,possible=certified(v.memory,v.position,readings[kind],old[kind],prior,sites[kind])
        ok=np.array([m in possible for m in self.early.MOVES])&legal
        certainly=np.array([m in sure for m in self.early.MOVES])&legal
        ok|=certainly
        if not ok.any():ok=legal.copy()
        if certainly.any():opinion=certainly.astype(float)
        elif ok.sum()==1:opinion=ok.astype(float)
        else:
            # Only numerical likelihoods for the eight directions are requested here.
            # This freshly initialized encoder sees the snapshot and previous readings.
            votes=MandatoryProof().direction_votes(v,readings,old,prior,sites)[kind]
            opinion=np.array([float(votes.get(m,0.)) for m in self.early.MOVES])
            opinion[~ok]=0.
            opinion/=max(float(opinion.max()),1.e-12)
        first=max(coverage(v.memory))<9
        parts[:,0]=opinion*(slack<=1)
        parts[:,1]=opinion*(1<slack<=4)
        parts[:,2]=opinion*(4<slack<=10)
        parts[:,3]=certainly*(slack<=4)
        parts[:,4]=ok*(slack<=1)
        parts[:,5]=opinion*first*(slack<=5)
        parts[:,6]=opinion*(not first)*(slack<=5)
        confidence=np.sort(q[legal])[-2]
        parts[:,7]=opinion*(confidence>-.9)
        parts[:,8]=opinion*(old[kind] is not None and old[kind]<readings[kind])
        parts[:,9]=opinion*(old[kind] is not None and old[kind]>readings[kind])
        return q,parts

    def early_scores(self,v,prior):
        q,fields=self.early_parts(v,prior)
        return q+fields@self.residual


def load(data,ablate=False):
    pack=torch.load(BytesIO(data),weights_only=True,map_location="cpu")
    if pack["format"]!="service-vote-mixture-1":raise ValueError("Unexpected model format")
    torch.set_num_threads(1)
    return VoteMixture(pack,ablate)
