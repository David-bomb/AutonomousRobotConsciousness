import ctypes as ct
from pathlib import Path
import numpy as np
from .features import probabilities

LIB=ct.CDLL(str(Path(__file__).with_name("path_samples.dll")))
LIB.paths.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_char,
        np.ctypeslib.ndpointer(np.float32),ct.c_int,np.ctypeslib.ndpointer(np.float32)]
LIB.paths.restype=None


def confidence(v,table,n=48):
    k,d1,d2=min(((b"F",v.fuel_distance,v.repair_distance),(b"R",v.repair_distance,v.fuel_distance)),
                key=lambda choice:(v.fuel if choice[0]==b"F" else v.health)-choice[1])
    out=np.zeros((8,5),np.float32)
    p,_=probabilities(v.memory,table)
    h,w=p.shape
    LIB.paths("".join(v.memory).encode(),w,h,v.position[1]*w+v.position[0],d1,d2,k,p,n,out)
    return out
