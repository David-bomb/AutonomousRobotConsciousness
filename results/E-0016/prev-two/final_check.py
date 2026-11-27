"""Независимая загрузка, выбор на каждом шаге без кеша, сверка всех 100 траекторий."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import sys
import torch
from .common import DEV,HERE,PARENT,SEQ,digest,read,rows,write
from .frozen_adapter import load
from .physical import episode

CONTROLLER=None


def setup(path,sha):
    global CONTROLLER
    torch.set_num_threads(1)
    CONTROLLER=load(path.read_bytes(),sha)


def check(job):
    row,expected=job
    trip=episode(row,CONTROLLER,"honest",False)
    assert trip==expected,row["name"]
    return trip["steps"]


def verify(tag):
    path=HERE/"weights"/(tag+".pt")
    mem=read(HERE/"dev"/(tag+"-honest.json"))
    sha=mem["weights_sha256"]
    assert sha==digest(path)
    with ProcessPoolExecutor(2,initializer=setup,initargs=(path,sha)) as workers:
        counts=[]
        for n in workers.map(check,zip(rows(DEV),mem["episodes"],strict=True)):
            counts.append(n)
            if len(counts)%25==0:print("uncached",len(counts),flush=True)
    modules=[HERE/"adapter.py",HERE/"frozen_adapter.py",SEQ/"model.py",SEQ/"sequence.dll",
             PARENT/"policy.py",PARENT/"critical_policy.py",PARENT/"rich_policy.py",PARENT/"geometry.dll"]
    if read(path.with_suffix(".json")).get("route_adapter_sha256"):modules.append(HERE/"route_adapter.py")
    if read(path.with_suffix(".json")).get("patch_adapter_sha256"):modules.append(HERE/"patch_adapter.py")
    root=PARENT.parents[1]
    write(HERE/"frozen.json",dict(tag=tag,weights=path.relative_to(HERE).as_posix(),sha256=sha,
         dev_result="dev/"+tag+"-honest.json",ablation="dev/"+tag+"-ablated.json",
         dev_sha256=digest(DEV),identical_uncached_episodes=len(counts),identical_actions=sum(counts),
         module="results.E-0016.prev-two.frozen_adapter",factory="load",device="cpu",
         sources={p.relative_to(root).as_posix():digest(p) for p in modules}))


if __name__=="__main__":verify(sys.argv[1])
