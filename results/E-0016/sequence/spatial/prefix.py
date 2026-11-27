import ctypes as ct
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/"prefix_bank.dll"))
LIB.load_prefix.argtypes=[np.ctypeslib.ndpointer(np.uint32),ct.c_int]
LIB.load_prefix.restype=ct.c_void_p
LIB.drop_prefix.argtypes=[ct.c_void_p]
LIB.drop_prefix.restype=None
LIB.score_prefix.argtypes=[ct.c_void_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,
                           ct.c_char,np.ctypeslib.ndpointer(np.float32)]
LIB.score_prefix.restype=None


class Prefix:
    def __init__(self,rows):
        self.handle=LIB.load_prefix(np.ascontiguousarray(rows,np.uint32),len(rows))

    def __del__(self):
        if getattr(self,"handle",None):LIB.drop_prefix(self.handle)

    def score(self,v):
        kind,d,second=min(((b"F",v.fuel_distance,v.repair_distance),(b"R",v.repair_distance,v.fuel_distance)),
                          key=lambda t:(v.fuel if t[0]==b"F" else v.health)-t[1])
        out=np.zeros((8,4),np.float32)
        w=len(v.memory[0])
        LIB.score_prefix(self.handle,"".join(v.memory).encode(),w,len(v.memory),v.position[1]*w+v.position[0],d,second,kind,out)
        return out
