"""Verify the latent mission constraints on real training histories before measuring predictions."""
import ctypes
import numpy as np
import torch
from renewable_train import BASE,BASE_HASH,EXP,HERE,Environment,Status,data,digest,save
from geodesic_probe import prev
from exact_policy import Exact,args
from service_window_policy import coverage


def main():
    torch.set_num_threads(1);out=HERE/"exact-search";out.mkdir(exist_ok=True)
    protocol=out/"probe-protocol.json"
    save(protocol,dict(source={n:digest(HERE/n) for n in ("exact_mission.cpp","exact_mission.dll","mission_training_check.dll",
                         "exact_policy.py","exact_probe.py","origin_trace.py")},cap=2048,stop=30,
         rule="Before testing predictions, verify a valid latent mission for the actual start on independent train histories, irrespective of whether a first service was previously visited. The full-map train verifier is compiled separately. Compare estimated descents on the same held-back 286 training forks as the earlier morphology probe, including exact joint matches and a sample-count-dependent fallback."))
    ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
    check=ctypes.CDLL(str(HERE/"mission_training_check.dll")).true_mission_check
    check.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ci,ci,ci,ints,ci,ci,ints,ci,ci]
    check.restype=ctypes.c_double
    pilot=prev.Controller(BASE,BASE_HASH);checks=[]
    bank=data.rows(HERE/"handover-search/train.jsonl")
    for r in bank[100:]:
        env=Environment(data.level(r));pilot.reset();past=(None,None);seen=set()
        while env.status==Status.RUNNING and min(coverage(env.view.memory))<9:
            v=env.view;key=v,past
            if key in seen:break
            seen.add(key)
            if env.steps%4==0:
                mem,w,h,pos,prior,sites,d,old,f,hp,starts,n,initial=args(v,past)
                sx,sy=env.level.start;true=sy*w+sx
                only=starts[starts[:,0]==true].copy()
                assert len(only)==1,(r["name"],env.steps)
                prob=check(mem,"".join(env.level.grid).encode("ascii"),w,h,pos,sites,f,hp,only,1,initial)
                assert prob>0,(r["name"],env.steps,v.position,v.fuel,v.health,coverage(v.memory),only.tolist())
                checks.append(dict(map=r["name"],step=env.steps,any_service=bool(env.refuels or env.repairs),weight=prob))
            m=pilot.choose(v);past=v.fuel_distance,v.repair_distance;env.step(m)
    save(out/"true-check.json",dict(cases=len(checks),after_first=sum(r["any_service"] for r in checks),rows=checks))
    world=Exact(2048,30);wanted={}
    meta={(e["map"],e["step"]):e for e in data.read(HERE/"geodesic-search/sensor-probe.json")["observations"]}
    for e in data.read(HERE/"morph-search/probe.json")["records"]:wanted.setdefault(e["map"],{})[e["step"]]=e
    rows=[]
    for r in data.rows(EXP/"sequence/train.jsonl")[1500:1600]:
        targets=wanted.get(r["name"],{})
        if not targets:continue
        env=Environment(data.level(r));pilot.reset();past=(None,None)
        for i in range(max(targets)+1):
            v=env.view
            if i in targets:
                base=targets[i];m=meta[(r["name"],i)];k=m["kind"]
                est,count=world.get(v,past);legal=np.array([a in m["eligible"] for a in range(8)])
                prior_candidate=base["picks"][2]
                a=count[3]/(count[3]+14.);b=count[5]/(count[5]+9.)
                conditional=(1-a)*est[1,:,k]+a*est[4,:,k]
                joint=(1-b)*conditional+b*est[5,:,k]
                decisions=[]
                for scores in [est[j,:,k] for j in range(6)]+[conditional,joint]:
                    decisions.append(int(np.where(legal,scores,-2).argmax()) if scores[legal].max()>0 else prior_candidate)
                rows.append(dict(map=r["name"],tier=r["tier"],family=r["family"],step=i,actual=base["real"],
                    previous=prior_candidate,choices=decisions,counts=count.tolist()))
            m=pilot.choose(v);past=v.fuel_distance,v.repair_distance;env.step(m)
    def stats(rs):
        return dict(n=len(rs),prior=sum(r["previous"] in r["actual"] for r in rs),
            choices=[sum(r["choices"][j] in r["actual"] for r in rs) for j in range(8)],
            any_exact_mission=sum(r["counts"][5]>0 for r in rs),median_exact_mission=float(np.median([r["counts"][5] for r in rs])))
    groups={s:stats(rows if s=="all" else [r for r in rows if r["family"]==s]) for s in ("all","rooms","loops","branching","corridors")}
    save(out/"probe.json",dict(summary=groups,rows=rows));print(groups,flush=True)


if __name__=="__main__":main()
