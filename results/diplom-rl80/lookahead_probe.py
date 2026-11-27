"""Check observation grouping and compare prospective actions on diagnosed training forks."""
from dataclasses import replace
import numpy as np
import torch
from renewable_train import HERE,Environment,data,digest,moves,save
from service_window_policy import load
from lookahead_policy import Future


def main():
    torch.set_num_threads(1);out=HERE/"lookahead-search";out.mkdir(exist_ok=True)
    bank=HERE/"handover-search/train.jsonl";rows=data.rows(bank)
    save(out/"probe-protocol.json",dict(bank=digest(bank),sha={n:digest(HERE/n) for n in ("sensor_lookahead.cpp",
        "sensor_lookahead.dll","lookahead_policy.py","lookahead_probe.py")},samples=256,
        rule="On the recorded first irreversible baseline action and matched viable controls, estimate one-step and two-step future values in simulated terrain. All worlds with the same future legal-move mask and same two distance changes must use the same next action. Verify adaptive value dominates the common blind next action; on fully known trees both values must agree. True input worlds/fork labels are only used in offline checks."))
    net=load((HERE/"service-window/fixed/model.pt").read_bytes());future=Future(256)
    full=0
    for r in rows:
        if r["family"] not in ("corridors","branching"):continue
        env=Environment(data.level(r));v=replace(env.view,memory=env.level.grid)
        z=future.estimate(v,(None,None))
        assert z[:,4:8].max()>0 and np.allclose(z[:,4:8],z[:,8:12]),r["name"]
        full+=1
        if full==12:break
    records=[]
    for r in rows:
        ref=data.read(HERE/"early-errors"/(r["name"]+".json"))
        if "fork" not in ref:continue
        f=ref["fork"];env=Environment(data.level(r));past=(None,None)
        for i in range(f["step"]):
            v=env.view;a=int(net.early_scores(v,past).argmax())
            past=v.fuel_distance,v.repair_distance;env.step(moves[a])
        v=env.view;q=net.early_scores(v,past)
        assert int(q.argmax())==f["chosen"] and list(past)==f["previous"]
        z=future.estimate(v,past);legal=np.array([a in f["legal"] for a in range(8)])
        choices=[int(np.where(legal,z[:,s]+np.clip(q-q[legal].max(),-1,0)*1.e-6,-2).argmax()) for s in range(16)]
        records.append(dict(map=r["name"],family=r["family"],tier=r["tier"],reason=ref["reason"],viable=f["viable"],
                       base=f["chosen"],decisions=choices,values=z.tolist()))
    def summarize(why):
        group=[r for r in records if r["reason"]==why]
        return dict(n=len(group),viable=[sum(r["decisions"][i] in r["viable"] for r in group) for i in range(16)])
    stats={reason:summarize(reason) for reason in ("first-loss","matched-viable")}
    save(out/"probe.json",dict(fully_observed_tree_checks=full,groups=stats,records=records));print(stats)


if __name__=="__main__":main()
