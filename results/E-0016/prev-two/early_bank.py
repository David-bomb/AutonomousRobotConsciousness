"""Четыре зарегистрированных новых учебных seed. Доступ только к train и dev."""
from concurrent.futures import ProcessPoolExecutor
from .common import HERE,PARENT,SEQ,DEV,TRAIN,digest,read,rows,write
from ..data import generate
from maze.datasets import geometry

FOLDER=HERE/"early"
LATE=SEQ/"runs/joint400/rl.pt"
LATE_SHA="ebf419fe5ed73a9204b8cc976bedbc1ca53aeffe615605fb443f20fb4f852f56"


def register(workers_count=2):
    FOLDER.mkdir(exist_ok=True)
    target=FOLDER/"registration.json"
    if target.exists():
        reg=read(target)
        for name,h in reg["files"].items():assert digest(FOLDER/name)==h
        return
    assert digest(LATE)==LATE_SHA
    occupied={r["geometry"] for f in (DEV,TRAIN,PARENT/"data/train.jsonl") for r in rows(f)}
    names=("fit1","fit2","select","hold")
    jobs=[("prev-two-early-"+n,9176401+i) for i,n in enumerate(names)]
    files={}
    with ProcessPoolExecutor(workers_count) as workers:
        for n,bank in zip(names,workers.map(generate,jobs),strict=True):
            for r in bank:
                assert r["geometry"]==geometry(r["grid"]) and r["geometry"] not in occupied
                occupied.add(r["geometry"])
            limits=(8,8,25,25) if n.startswith("fit") else ((8,)*4 if n=="select" else (12,)*4)
            chosen=[r for j,cap in enumerate(limits,1) for r in [x for x in bank if x["tier"]==f"H{j}"][:cap]]
            file=FOLDER/(n+".json")
            write(file,chosen);files[file.name]=digest(file)
            print("early bank",n,len(chosen),flush=True)
    write(target,dict(seeds=[s for _,s in jobs],files=files,plan=digest(HERE/"early-plan.md"),
                      wave1=digest(HERE/"weights/wave1.pt"),late=LATE_SHA,dev=digest(DEV)))


if __name__=="__main__":register()
