import ctypes as ct
from functools import lru_cache
from pathlib import Path
import numpy as np
from ...rich_policy import context_codes,contexts

HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/"walks.dll"))
LIB.walks.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_int,ct.c_char,
                    np.ctypeslib.ndpointer(np.float32),np.ctypeslib.ndpointer(np.float32)]
LIB.walks.restype=None


@lru_cache(maxsize=2048)
def probabilities(memory,table):
    field=np.asarray([list(r.encode("ascii")) for r in memory],np.uint8)
    priors=np.frombuffer(table,np.float32).reshape(4,648)
    codes=context_codes(field)
    known=field!=ord("?")
    road=field!=ord("#")
    # Семейство не наблюдается. Четыре возможных предположения взвешены по
    # типам ранее открытых клеток в том же текущем снимке.
    base=priors[:,contexts(*field.shape)*81].clip(.001,.999)
    observed=np.where(road[None],np.log(base),np.log1p(-base))
    ll=observed[:,known].sum(1)
    weights=np.exp(np.clip((ll-ll.max())*.25,-70,0))
    weights/=weights.sum()
    predicted=(priors[:,codes]*weights[:,None,None]).sum(0)
    predicted[known]=road[known]
    return np.ascontiguousarray(predicted,np.float32),weights


def features(v,table):
    prob,mix=probabilities(v.memory,table)
    h,w=prob.shape
    raw="".join(v.memory).encode("ascii")
    p=v.position[1]*w+v.position[0]
    arr=[]
    for kind,d,d2 in ((b"F",v.fuel_distance,v.repair_distance),(b"R",v.repair_distance,v.fuel_distance)):
        out=np.zeros((8,6),np.float32)
        LIB.walks(raw,w,h,p,d,d2,kind,prob,out)
        arr.append(out)
    if v.health<v.fuel:
        arr.reverse()
    return np.concatenate((*arr,np.broadcast_to(mix,(8,4))),1)
