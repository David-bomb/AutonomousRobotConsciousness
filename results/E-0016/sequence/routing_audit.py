"""Разбор ошибок mission600 на train по доступному маршруту и состоянию запасов."""
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment,Status
from ..data import digest,level,rows,save
from ..policy import MOVES
from ..teacher import Teacher
from .bank import HERE
from .model import SequenceNet,encode


@torch.inference_mode()
def main():
    path=HERE/"runs/mission600/rl.pt"
    torch.set_num_threads(1)
    net=SequenceNet().eval()
    net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
    prior=net.wall_prior.numpy().tobytes()
    summary=Counter()
    records=[]
    for row in rows(HERE/"train.jsonl")[900:1000]:
        e=Environment(level(row))
        expert=Teacher(e.level)
        local=Counter()
        cache={}
        first=None
        while e.status==Status.RUNNING:
            v=e.view
            if v not in cache:
                x,mask=encode(v,prior)
                a=int(net(torch.from_numpy(x[None]),torch.from_numpy(mask[None])).argmax())
                route=x[:,255:295].reshape(8,5,8)
                lengths=np.where((route[:,:,6]>.5)&mask[:,None],route[:,:,4],10).min(1)
                safe=x[:,447:451].any(1)&mask
                cache[v]=(a,lengths,safe)
            a,lengths,safe=cache[v]
            confirmed=lengths.min()<10
            if confirmed:
                local["confirmed"]+=1
                local["confirmed_shortest"]+=int(lengths[a]==lengths.min())
                if lengths[a]>=10:
                    local["left_confirmed"]+=1
            elif safe.any():
                local["safe_frontier_available"]+=1
                local["used_direction_with_safe_frontier"]+=int(safe[a])
            if first is None:
                _,viable=expert.labels(e)
                if viable.any() and not viable[a]:
                    low=min(e.fuel-v.fuel_distance,e.health-v.repair_distance)
                    first=dict(step=e.steps+1,slack=low,known=bool(confirmed),safe=bool(safe.any()),
                               stage=int(e.refuels>0)+int(e.repairs>0))
                    local["first_loss"]+=1
                    local["first_loss_tight"]+=int(low<=3)
                    local["first_loss_with_confirmed"]+=int(confirmed)
                    local["first_loss_with_safe_frontier"]+=int(safe.any())
            e.step(MOVES[a])
        expert.remaining.cache_clear()
        local["maps_ever_confirmed"]+=int(local["confirmed"]>0)
        local["wins"]=int(e.status==Status.SUCCESS)
        local["lost_after_confirmed"]=int(local["confirmed"]>0 and e.status!=Status.SUCCESS)
        summary.update(local)
        records.append(dict(map=row["name"],outcome=e.status.name,first_irreversible=first,counts=dict(local)))
    save(HERE/"routing-audit.json",dict(weights_sha256=digest(path),train_range=[900,1000],
         train_sha256=digest(HERE/"train.jsonl"),summary=dict(summary),episodes=records))
    print(dict(summary),flush=True)


if __name__=="__main__":
    main()
