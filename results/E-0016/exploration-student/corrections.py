"""После самостоятельного префикса учитель размечает продолжение только на train."""
import os
for key in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from concurrent.futures import ProcessPoolExecutor
import torch
from maze.environment import Environment,Status
from maze.frontier_reserve import ReserveFrontier
from .adapter import Controller
from .bank import HERE,digest,level,read,save,train_order
from .collect import MOVES,drive

STUDENT=None


def start_worker(weight,sha):
    global STUDENT
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    STUDENT=Controller(weight,sha)


def collect(job):
    i,row,sha=job
    path=HERE/"correction-traces"/(row["name"]+".json")
    if path.exists():
        trace=read(path)
        assert trace["parent_sha256"]==sha
        return trace
    mode=(i//4+i%4)%4
    env=Environment(level(row))
    prefix=[]
    while env.status==Status.RUNNING and env.steps<72:
        if ((mode==0 and env.steps>=8) or (mode==1 and env.steps>=20) or
                (mode==2 and env.refuels+env.repairs) or (mode==3 and min(env.refuels,env.repairs)>0)):
            break
        move=STUDENT.choose(env.view)
        env.step(move)
        prefix.append(str(MOVES.index(move)))
    result=drive(env,ReserveFrontier())
    result["actions"]="".join(prefix)+result["actions"]
    result.update(map=row["name"],tier=row["tier"],geometry=row["geometry"],outcome=env.status.name,
                  steps=env.steps,refuels=env.refuels,repairs=env.repairs,prefix=len(prefix),
                  prefix_rule=mode,parent_sha256=sha)
    save(path,result)
    return result


def main():
    tags=["imitation12","imitation32"]
    def rank(tag):
        s=read(HERE/"dev"/(tag+".json"))["summary"]
        return s["wins"],min(s["tiers"].values())
    tag=max(tags,key=rank)
    weights=HERE/"weights"/(tag+".pt")
    sha=read(HERE/"weights"/(tag+".json"))["sha256"]
    n=read(HERE/"teacher-speed.json")["n"]
    batch=train_order()[n:n+64]
    traces=[]
    with ProcessPoolExecutor(2,initializer=start_worker,initargs=(weights,sha)) as workers:
        for trace in workers.map(collect,[(i,row,sha) for i,row in enumerate(batch)]):
            traces.append(trace)
            if len(traces)%16==0:
                print("corrections",len(traces),flush=True)
    save(HERE/"correction.json",dict(parent=tag,parent_sha256=sha,
         teacher_takeovers=sum(r["teacher_calls"]>0 for r in traces),
         wins=sum(r["outcome"]=="SUCCESS" for r in traces),
         files={r["map"]:digest(HERE/"correction-traces"/(r["map"]+".json")) for r in traces}))


if __name__=="__main__":
    main()
