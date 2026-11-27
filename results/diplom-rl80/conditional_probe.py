"""Compare distance-conditioned latent-road priors on held-back train observations."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from geodesic_probe import BASE,HASH,EXP,HERE,Environment,Status,data,moves,prev
from renewable_policy import targets
from service_window_policy import coverage
from renewable_train import digest,save

OUT=HERE/"conditional-search"


@lru_cache(maxsize=256)
def origin(memory):
    for y,row in enumerate(memory):
        if "A" in row:return y*len(row)+row.index("A")
    return -1


class Worlds:
    def __init__(self,particles):
        self.particles=particles
        ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
        self.lib=ctypes.CDLL(str(HERE/"conditional_worlds.dll"))
        self.lib.posterior.argtypes=[ctypes.c_char_p,ci,ci,ci,ci,ci,ints,ints,ints,ci,ci,ci,ci,ci,
                                     np.ctypeslib.ndpointer(np.float64)]
        self.lib.posterior.restype=None

    def evaluate(self,v,old):
        w=len(v.memory[0]);x,y=v.position
        prev_pos=y*w+x
        if v.moved:
            dx,dy=v.last_move.value;prev_pos-=dy*w+dx
        before=np.array([-1 if n is None else n for n in old],np.int32)
        now=np.array([v.fuel_distance,v.repair_distance],np.int32)
        ids=targets(v.memory,(x+v.dx,y+v.dy))
        out=np.empty((3,8,6))
        self.lib.posterior("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,prev_pos,
            y*w+x if v.last_move is None else origin(v.memory),ids,now,before,v.fuel,v.health,
            max(coverage(v.memory))<9,v.last_move is None,
            self.particles,out)
        if not np.isfinite(out).all():raise ValueError("Non-finite latent-road estimate")
        return out


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    source=HERE/"geodesic-search/sensor-probe.json"
    items=data.read(source)["observations"]
    wanted={}
    for e in items:
        group=wanted.setdefault(e["map"],{})
        if len(group)<4 and len(e["eligible"])>1 and e["slack"]<=7:group[e["step"]]=e
    save(OUT/"probe-protocol-v2.json",dict(source_sha256=digest(source),particles=192,
         rule="Up to first four eligible pre-service forks per held-back train map with multiple possible descents and slack <=7. Reuse the recorded observation of wave1. Compare against true next readings only after assigning scores.",
         cpp_sha256=digest(HERE/"conditional_worlds.cpp"),dll_sha256=digest(HERE/"conditional_worlds.dll"),
         macro_prior="Unobserved paths are simulated by locally constrained depth-first or branching trees and loop/room variants. Third likelihood includes resource-difference compatibility only if no station has a fully observed window. First attempt stopped because the ordinary memory omits the start label. Use the true current location on the first move; on subsequent moves omit all unobservable start-location terms."))
    world=Worlds(192);policy=prev.Controller(BASE,HASH)
    records=[]
    for r in data.rows(EXP/"sequence/train.jsonl")[1500:1600]:
        query=wanted.get(r["name"],{})
        if not query:continue
        env=Environment(data.level(r));policy.reset();old=(None,None)
        for t in range(max(query)+1):
            v=env.view
            if t in query:
                row=query[t];k=row["kind"]
                vals=world.evaluate(v,old)
                allowed=np.array([a in row["eligible"] for a in range(8)])
                choices=[int(np.where(allowed,vals[i,:,k],-1).argmax()) for i in range(3)]
                viability=[int(np.where(allowed,vals[i,:,3]+.15*vals[i,:,4],-1).argmax()) for i in range(3)]
                records.append(dict(map=r["name"],step=t,tier=r["tier"],actual=row["actual"],slack=row["slack"],
                     policy=row["model"],sensor=row["sensor"],likelihood=row["pred"],posterior=choices,mission=viability,
                     probabilities=vals[:,:,k].tolist(),scores=vals[:,:,3:].tolist()))
            m=policy.choose(v);old=(v.fuel_distance,v.repair_distance);env.step(m)
    def measure(rows):
        return dict(n=len(rows),**{k:sum(r[k] in r["actual"] for r in rows) for k in ("policy","sensor","likelihood")},
                    **{k:[sum(r[k][i] in r["actual"] for r in rows) for i in range(3)] for k in ("posterior","mission")})
    groups={"all":measure(records),"tight":measure([r for r in records if r["slack"]<=2]),
            "hard":measure([r for r in records if r["tier"]=="H4"])}
    save(OUT/"probe.json",dict(groups=groups,records=records));print(groups)


if __name__=="__main__":main()
