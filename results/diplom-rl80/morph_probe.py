"""Paired check on held-back training forks, including ground-truth family only in reports."""
import numpy as np
import torch
from geodesic_probe import BASE,HASH,EXP,HERE,Environment,data,moves,prev
from renewable_train import digest,save
from morph_policy import Terrain


def main():
    torch.set_num_threads(1)
    out=HERE/"morph-search";out.mkdir(exist_ok=True)
    reference=HERE/"conditional-search/probe.json";doc=data.read(reference)
    meta={(e["map"],e["step"]):e for e in data.read(HERE/"geodesic-search/sensor-probe.json")["observations"]}
    targets={}
    for e in doc["records"]:targets.setdefault(e["map"],{})[e["step"]]=e
    save(out/"probe-protocol.json",dict(source=digest(reference),samples=384,code={n:digest(HERE/n) for n in
        ("morphology_worlds.cpp","morphology_worlds.dll","morph_policy.py","morph_probe.py","origin_trace.py")},
        method="Same 286 held-back training forks as the earlier conditional-world experiment. Build feasible underlying coarse trees, then exact-count rectangles or extra holes compatible with all observed floor/wall/landmark cells. Compare predicted service descent against true physical next readings. The true family is used only for reporting."))
    world=Terrain(384);pilot=prev.Controller(BASE,HASH);records=[]
    for r in data.rows(EXP/"sequence/train.jsonl")[1500:1600]:
        wanted=targets.get(r["name"],{})
        if not wanted:continue
        past=(None,None);env=Environment(data.level(r));pilot.reset()
        for i in range(max(wanted)+1):
            v=env.view
            if i in wanted:
                e=wanted[i];m=meta[(r["name"],i)];val,counts=world.get(v,past);k=m["kind"]
                legal=np.array([a in m["eligible"] for a in range(8)])
                picked=[]
                for j in range(6):
                    vec=val[j,:,k]
                    picked.append(int(np.where(legal,vec,-2).argmax()) if vec[legal].max()>0 else e["policy"])
                records.append(dict(map=r["name"],tier=r["tier"],family=r["family"],step=i,slack=e["slack"],real=e["actual"],
                        parent=e["policy"],old=e["posterior"][2],sensor=e["sensor"],picks=picked,counts=counts.tolist()))
            m=pilot.choose(v);past=(v.fuel_distance,v.repair_distance);env.step(m)
    def result(group):
        return dict(n=len(group),**{k:sum(r[k] in r["real"] for r in group) for k in ("parent","old","sensor")},
            new=[sum(r["picks"][j] in r["real"] for r in group) for j in range(6)],
            exact_particles=sum(int(np.sum(np.asarray(r["counts"])[:,2])) for r in group),
            forks_with_a_fully_matched_particle=sum(np.sum(np.asarray(r["counts"])[:,2])>0 for r in group).item())
    summary={s:result(records if s=="all" else [r for r in records if r["family"]==s]) for s in
             ("all","rooms","loops","branching","corridors")}
    save(out/"probe.json",dict(summary=summary,records=records));print(summary,flush=True)


if __name__=="__main__":main()
