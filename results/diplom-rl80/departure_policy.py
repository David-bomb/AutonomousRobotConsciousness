"""Choose a physical move towards an excursion with enough supply across plausible roads."""
import ctypes
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from protected_policy import Protected
from renewable_policy import targets


class Departure(Protected):
    def __init__(self, pack, ablate=False):
        super().__init__(pack, ablate)
        self.weights = pack["departure"].numpy()
        self.cap = int(pack["departure_samples"])
        root = Path(__file__).resolve().parent
        integer = ctypes.c_int
        ints = np.ctypeslib.ndpointer(np.int32)
        self.known = ctypes.CDLL(str(root / "proper_frontier.dll")).proper
        self.known.argtypes = [ctypes.c_char_p, integer, integer, integer, ints, integer, integer, ints]
        self.known.restype = None
        self.excursion = ctypes.CDLL(str(root / "departure.dll")).departures
        self.excursion.argtypes = [ctypes.c_char_p, integer, integer, integer, integer, ints, ints, ints,
                                  integer, integer, integer, np.ctypeslib.ndpointer(np.float64)]
        self.excursion.restype = None

    def late_parts(self, v, past):
        direction, legal = self.b.directions(v)
        q = direction @ self.linear
        q = (q-q[legal].max()) / max(.5, float(q[legal].std()))
        q[~legal] = -1.e6
        fields = np.zeros((8, 12))
        if legal.sum() < 2:
            return q, fields
        x,y = v.position
        w,h = len(v.memory[0]),len(v.memory)
        pos = y*w+x
        ids = targets(v.memory, (x+v.dx,y+v.dy))
        mem = "".join(v.memory).encode("ascii")
        masks = np.zeros(7, np.int32)
        self.known(mem,w,h,pos,ids,v.fuel,v.health,masks)
        if masks[0] or masks[1] or not masks[4]:
            return q, fields
        oldpos = pos
        if v.moved:
            mx,my = v.last_move.value
            oldpos -= mx+w*my
        now = np.asarray([v.fuel_distance,v.repair_distance],np.int32)
        old = np.asarray([-1 if z is None else z for z in past],np.int32)
        probability = np.zeros((8,6))
        self.excursion(mem,w,h,pos,oldpos,ids,now,old,v.fuel,v.health,self.cap,probability)
        parent = int(q.argmax())
        for j in range(6):
            advantage = probability[:,j] - probability[parent,j]
            fields[:,j] = advantage
            best = int(np.where(legal, probability[:,j], -2).argmax())
            threshold = .12 if j < 3 else .06
            if advantage[best] > threshold:
                fields[best,j+6] = q[parent]-q[best]+1.
        return q, fields

    def choose(self, v, cache=None):
        if v.last_move is None:
            self.previous = (None,None)
        prior = (None,None) if self.ablate else self.previous
        key = v,prior
        if cache is None or key not in cache:
            if self.route(v,self.cfg) and self.weights.any():
                q,x = self.late_parts(v,prior)
                q = q+x@self.weights
            elif self.route(v,self.cfg):
                q = super().late_scores(v)
            else:
                q = self.early_scores(v,prior)
            result = self.early.MOVES[int(q.argmax())]
            if cache is not None:
                cache[key] = result
        else:
            result = cache[key]
        self.previous = v.fuel_distance,v.repair_distance
        return result


def load(blob, ablate=False):
    pack = torch.load(BytesIO(blob),weights_only=True,map_location="cpu")
    if pack["format"] != "departure-1":
        raise ValueError("Unknown policy format")
    return Departure(pack, ablate)
