import ctypes as ct
import os
from ...data import rows,save,digest
from .prefix import HERE,LIB


def main():
    path=HERE/"extra.jsonl"
    reg=rows(path)[:3000]
    LIB.create_counts.argtypes=[]
    LIB.create_counts.restype=ct.c_void_p
    LIB.train_prefix.argtypes=[ct.c_void_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_uint32]
    LIB.train_prefix.restype=None
    LIB.write_prefix.argtypes=[ct.c_void_p,ct.c_char_p]
    LIB.write_prefix.restype=ct.c_int
    bank=LIB.create_counts()
    for i,r in enumerate(reg,1):
        LIB.train_prefix(bank,"".join(r["grid"]).encode(),r["width"],r["height"],int(r["geometry"][:8],16))
        if i%500==0:print(i,flush=True)
    out=HERE/"prefix-counts.bin"
    count=LIB.write_prefix(bank,os.fsencode(str(out)))
    assert count>0 and out.stat().st_size==16*count
    save(HERE/"prefix.json",dict(train_sha256=digest(path),maps=len(reg),range=[0,3000],max_length=64,
         max_prefix=12,entries=count,sha256=digest(out),cpp=digest(HERE/"prefix_bank.cpp"),dll=digest(HERE/"prefix_bank.dll")))


if __name__=="__main__":main()
