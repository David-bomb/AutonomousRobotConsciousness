"""Привязка успешных демонстраций и этапа обслуживания к уже собранным наблюдениям."""
from collections import Counter
import numpy as np
from maze.environment import Environment,Status
from ..data import read,rows,level,save
from ..policy import MOVES
from .bank import HERE


def main():
    records=read(HERE/"clean-400.json")["episodes"]
    lookup={r["name"]:r for r in rows(HERE/"train.jsonl")}
    phase=[];success=[];weight=[];target=[]
    for r in records:
        env=Environment(level(lookup[r["map"]]))
        seen=set()
        items=[]
        for t,code in enumerate(r["actions"]):
            v=env.view
            stage=int(env.refuels>0)+int(env.repairs>0)
            f,h=env.refuels,env.repairs
            env.step(MOVES[int(code)])
            event="F" if env.refuels>f else "R" if env.repairs>h else "B" if env.status==Status.SUCCESS else ""
            items.append((v not in seen,stage,v.health<v.fuel,event))
            seen.add(v)
        next_kind="B"
        labels=[]
        for keep,stage,swap,event in reversed(items):
            next_kind=event or next_kind
            if keep:
                labels.append((stage,0 if next_kind=="B" else 1 if (next_kind=="R")==swap else 2))
        labels.reverse()
        count=Counter(s for s,_ in labels)
        for s,k in labels:
            phase.append(s);target.append(k);success.append(env.status==Status.SUCCESS)
            weight.append(1/(len(count)*count[s]))
    assert len(phase)==read(HERE/"prepare-clean-400.json")["records"]
    path=HERE/"cache/clean-meta.npz"
    np.savez_compressed(path,phase=np.asarray(phase),target=np.asarray(target),
                        success=np.asarray(success),weight=np.asarray(weight,np.float32))
    save(HERE/"phase-counts.json",dict(total=len(phase),completed_states=sum(success),
         complete_by_stage=dict(Counter(s for s,ok in zip(phase,success) if ok))))


if __name__=="__main__":
    main()
