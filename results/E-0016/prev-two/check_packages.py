"""Проверка замороженных родителей, исходников и опубликованных результатов комплектов."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
from pathlib import Path
import sys
import torch
from .common import HERE,PARENT,SEQ,DEV,digest,read,write
from .early_bank import LATE
from . import cautious_policy,path_policy


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(1)
    early=torch.load(HERE/"weights/wave1.pt",weights_only=True,map_location="cpu")
    late=torch.load(LATE,weights_only=True,map_location="cpu")
    root=PARENT.parents[1]
    for f,factory in (("conservative",cautious_policy.load),("geodesic",path_policy.load)):
        base=HERE/f;path=base/"frozen.pt";lock=read(base/"selection.json");sha=digest(path)
        assert sha==lock["sha256"] and all(digest(HERE/k)==v for k,v in lock["sources"].items())
        pack=torch.load(path,weights_only=True,map_location="cpu")
        assert same(pack["early"],early) and same(pack["late"],late) and pack["threshold"]==lock["threshold"]
        ctl=factory(path.read_bytes());assert ctl.previous==(None,None)
        obj=read(base/"audit.json");assert obj["sha256"]==sha
        deps=[PARENT/"geometry.dll",SEQ/"sequence.dll",SEQ/"alternatives.dll",SEQ/"belief/walks.dll",
              SEQ/"belief/itineraries.dll",HERE/"cautious_geometry.dll"]
        if f=="geodesic":deps.append(HERE/"path_counts.dll")
        for module in list(sys.modules.values()):
            if not getattr(module,"__file__",None):continue
            p=Path(module.__file__).resolve()
            if p.suffix==".py" and (p.is_relative_to(root/"maze") or p.is_relative_to(PARENT)):
                deps.append(p)
        write(base/"package.json",dict(weights_sha256=sha,identical_early_tensors=True,early_parent=digest(HERE/"weights/wave1.pt"),
              identical_late_tensors=True,late_parent=digest(LATE),adapter=factory.__module__+".load",physical_dev=digest(DEV),
              outcomes={name:digest(base/name) for name in ("dev.json","dev-ablated.json","paired.json","audit.json")},
              loaded_sources={p.relative_to(root).as_posix():digest(p) for p in sorted(set(deps))}))
    print("identical frozen early and late parents, both packages",flush=True)


if __name__=="__main__":main()
