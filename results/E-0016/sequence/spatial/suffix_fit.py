import ctypes as ct
import numpy as np
from ...data import digest, rows, save
from ..belief.prior import FAMILIES
from .model import HERE


def main():
    source=HERE/"large.jsonl"
    data=rows(source)
    native=ct.CDLL(str(HERE/"suffix_counts.dll"))
    native.learn_suffix.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,np.ctypeslib.ndpointer(np.int32)]
    native.learn_suffix.restype=None
    counts=np.zeros((4,2,4,145,73,66),np.int32)
    for i,row in enumerate(data,1):
        j=FAMILIES.index(row["family"])
        assert max(row["width"],row["height"]) <= 37
        native.learn_suffix("".join(row["grid"]).encode(),row["width"],row["height"],counts[j])
        if i%2000==0:print("suffix maps",i,flush=True)
    path=HERE/"suffix-counts.npz"
    with path.open("xb") as f:np.savez_compressed(f,counts=counts)
    save(HERE/"suffix.json",dict(maps=len(data),geometries=digest(source),families=FAMILIES,
        total_prefixes=int(counts[...,65].sum()),observed_continuations=int(counts[...,:65].sum()),
        tensor_sha256=digest(path),implementation=digest(HERE/"suffix_counts.cpp"),
        rule="two points, displacement in the entry frame, parity, family, exact residual length"))


if __name__=="__main__":main()
