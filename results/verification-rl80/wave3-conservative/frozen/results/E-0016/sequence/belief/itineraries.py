import ctypes as ct
from pathlib import Path
import numpy as np
from ...policy import geometry
from ...rich_policy import estimated_memory

LIB=ct.CDLL(str(Path(__file__).with_name("itineraries.dll")))
LIB.itineraries.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,
          np.ctypeslib.ndpointer(np.int32),ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.float32)]
LIB.itineraries.restype=None


def measures(v,prior):
    x,y=v.position
    goal=x+v.dx,y+v.dy
    sites,_=geometry(v.memory,v.position,goal)
    est=estimated_memory(v.memory,goal,prior)
    w=len(v.memory[0])
    ids=(sites[:,1]*w+sites[:,0]).astype(np.int32)
    raw=np.empty((8,3,3,4),np.float32)
    LIB.itineraries("".join(v.memory).encode(),"".join(est).encode(),w,len(v.memory),
                    y*w+x,ids,v.fuel,v.health,raw)
    return raw
