"""Test first-entry distributions under the actual outcome of complete robot trips."""
import numpy as np
import torch
from boundary_policy import BoundaryNet,load
from early_trials import compare,physical
from exact_train import balanced
from renewable_train import EXP,HERE,Environment,data,digest,moves,save
from service_window_policy import load as old_load

OUT=HERE/"boundary-search"


class FullTrip:
    # Reuse the deterministic population evaluator from reset all the way to the exit.
    cfg=()

    def __init__(self,net):self.net=net

    def route(self,v,cfg):return False

    def early_parts(self,v,past):
        if self.net.route(v,self.net.cfg):return self.net.late_parts(v,past)
        return self.net.early_scores(v,past),np.zeros((8,12))


def probe(net,maps,events):
    parent=old_load((HERE/"service-window/fixed/model.pt").read_bytes());records=[]
    for event in events:
        if event["step"] is None:continue
        env=Environment(data.level(maps[event["map"]]));past=(None,None);cache={}
        for _ in range(event["step"]):
            v=env.view;action=parent.choose(v,cache);past=v.fuel_distance,v.repair_distance;env.step(action)
        q,x=net.late_parts(env.view,past);chosen=int(q.argmax())
        assert chosen==event["fork"]["chosen"]
        picks=(q[:,None]+8*x).argmax(0)
        records.append(dict(map=event["map"],win=event["win"],legal=event["fork"]["viable"],chosen=chosen,
                            picks=picks.tolist(),values=x.tolist()))
    counts={s:[sum(r["picks"][j] in r["legal"] for r in records if r["win"]==v) for j in range(12)]
            for s,v in (("matched",True),("first-loss",False))}
    save(OUT/"probe.json",dict(counts=counts,records=records));print(counts,flush=True)


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    parent=HERE/"protected-search/fixed/model.pt";bank=EXP/"sequence/spatial/large.jsonl"
    maps={r["name"]:r for f in ("return-search/bank.jsonl","handover-search/train.jsonl") for r in data.rows(HERE/f)}
    events=data.read(HERE/"late-errors/summary.json")["episodes"]
    fit=[maps[e["map"]] for e in events];select=balanced(data.rows(bank)[5600:5700],25)
    eye=np.eye(12);options=np.vstack((np.zeros(12),8*eye,3*eye[0]+5*eye[6],3*eye[3]+5*eye[9]))
    save(OUT/"protocol.json",dict(parent=digest(parent),fit=[r["name"] for r in fit],select=[r["name"] for r in select],
        options=options.tolist(),bank_sha256=digest(bank),source={n:digest(HERE/n) for n in ("boundary_train.py",
        "boundary_policy.py","boundary_walk.cpp","boundary_walk.dll","departure_policy.py","proper_frontier.dll","early_trials.py")},
        rule="When known renewable exploration is exhausted, propagate non-backtracking walks from the exit and remote services up to their first contact with known connected terrain. Compare maximum and sum of reachable branch probabilities across observable frontier cells and stock budgets at arrival. Measure choices at fixed training loss/matched forks, compare whole-terminal returns on that training cohort; top four nonzero settings and their mean against parent on a distinct balanced training hundred. First on ties. Freeze prior to complete dev and ablation. Population evaluator's prefix encompasses the full trip."))
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="boundary-walk-1",departure_samples=1,departure=torch.zeros(12,dtype=torch.float64),
                boundary=torch.zeros(12,dtype=torch.float64))
    net=BoundaryNet(pack);probe(net,maps,events);trial=FullTrip(net)
    a=compare(OUT,"fit",fit,options,trial)
    leads=options[np.argsort(-a[1:],kind="stable")[:4]+1];pool=np.vstack((options[0],leads,leads.mean(0)))
    b=compare(OUT,"select",select,pool,trial);theta=pool[b.argmax()]
    dst=OUT/"model.pt"
    if not dst.exists():
        tmp=dst.with_suffix(".tmp");torch.save(dict(pack,boundary=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(dst)
    save(OUT/"frozen.json",dict(model_sha256=digest(dst),fit=a.tolist(),select=b.tolist(),coefficients=theta.tolist(),
                                protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(dst.read_bytes(),ablate))


if __name__=="__main__":main()
