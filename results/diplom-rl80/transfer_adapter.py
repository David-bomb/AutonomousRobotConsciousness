"""Frozen transfer of two trained frontier weights onto the controller with two actual readings."""
from importlib import import_module
from io import BytesIO

import numpy as np
import torch


def load(checkpoint:bytes, ablate=False):
    prev=import_module("results.E-0016.prev-two.adapter")
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    if pack["format"]!="prev-two-frontier-1":raise ValueError("Checkpoint format mismatch")
    core=prev.PrevNet("plain")
    core.load_state_dict(pack["base"])
    core.eval().requires_grad_(False)
    torch.set_num_threads(1)
    coefficients=pack["frontier"].numpy()
    if coefficients.shape!=(2,):raise ValueError("Expected two frontier weights")

    class Controller:
        def __init__(self):
            self.previous=(None,None)

        @torch.inference_mode()
        def choose(self,v,cache=None):
            if v.last_move is None:self.previous=(None,None)
            registers=(None,None) if ablate else self.previous
            key=v,registers
            if cache is None or key not in cache:
                obs,grad,past,allowed,legal=core.features(v,registers)
                logits=core(*prev.tensor_batch([(obs,grad,past,allowed,legal)]))[0].numpy()
                logits=(logits-logits[legal].max())/max(float(logits[legal].std()),.5)
                routes=obs[:,1:41].reshape(8,5,8)
                has=legal & (routes[:,:,6]>0).any(1)
                if has.any():
                    length=np.where(routes[:,:,6]>0,routes[:,:,4]*75,10000).min(1)
                    weight=coefficients[0]
                else:
                    has=legal & (obs[:,1+160+24+9]>0)
                    length=obs[:,1+160+9]*50
                    weight=coefficients[1]
                if has.any():
                    response=np.zeros(8,np.float32)
                    response[has]=np.exp(-np.maximum(length[has]-length[has].min(),0)*.60)
                    logits=logits.astype(np.float64)+weight*response.astype(np.float64)
                move=prev.MOVES[int(logits.argmax())]
                if cache is not None:cache[key]=move
            else:move=cache[key]
            self.previous=(v.fuel_distance,v.repair_distance)
            return move
    return Controller()
