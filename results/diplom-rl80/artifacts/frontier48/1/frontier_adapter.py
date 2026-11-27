"""Deployment of the current-snapshot policy. No previous readings or training data."""
from importlib import import_module
from io import BytesIO

import numpy as np
import torch


def load(checkpoint: bytes):
    sequence=import_module("results.E-0016.sequence.model")
    moves=tuple(import_module("maze.interface").Move)
    pack=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    if pack["format"]!="known-frontier-1":
        raise ValueError("Unexpected checkpoint format")
    net=sequence.SequenceNet("plain").eval()
    net.load_state_dict(pack["base"])
    net.requires_grad_(False)
    torch.set_num_threads(1)
    prior=net.wall_prior.numpy().tobytes()
    weights=pack["weights"].numpy()
    if weights.shape!=(9,) or weights[2:].any():
        raise ValueError("This adapter describes the two active trained features")

    class Controller:
        @torch.inference_mode()
        def choose(self,v):
            x,legal=sequence.encode(v,prior)
            if legal.sum()==1:
                return moves[int(legal.argmax())]
            logits=net(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))[0].numpy()
            logits=(logits-logits[legal].max())/max(float(logits[legal].std()),.5)
            known=x[:,255:415].reshape(8,4,5,8)[:,0]
            has=legal & (known[:,:,6]>0).any(1)
            if has.any():
                length=np.where(known[:,:,6]>0,known[:,:,4]*75,10000).min(1)
                coefficient=weights[0]
            else:
                # Minimum explored length to a new observation frontier plus remaining
                # straight-line displacement, with feasible confirmed return to both services.
                survey=x[:,415:]
                has=legal & (survey[:,24+9]>0)
                length=survey[:,9]*50
                coefficient=weights[1]
            if has.any():
                score=np.zeros(8,np.float32)
                score[has]=np.exp(-np.maximum(length[has]-length[has].min(),0)*.60)
                logits=logits.astype(np.float64)+coefficient*score.astype(np.float64)
            return moves[int(logits.argmax())]
    return Controller()
