"""Верхние ориентиры с вмешательством наблюдателя на train, отдельно от оценки модели."""
from collections import Counter
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from ..adapter import Controller
from ..data import OUT,read,rows,digest,level,save
from ..policy import MOVES,features
from ..teacher import Teacher
from .bank import HERE


def main():
    path=OUT/"runs/recovery800/rl.pt"
    net=Controller(path,digest(path))
    bank=rows(HERE/"train.jsonl")[800:900]
    experiments={}
    for variant in ("original","exact_descent","avoid_first_loss"):
        records=[]
        for r in bank:
            env=Environment(level(r))
            expert=Teacher(env.level)
            cache={}
            interventions=0
            while env.status==Status.RUNNING:
                v=env.view
                if v not in cache:
                    x,mask=features(v)
                    with torch.inference_mode():
                        score=net.model(torch.from_numpy(x[None]),torch.from_numpy(mask[None]))[0].numpy()
                    a=base=int(score.argmax())
                    if variant=="avoid_first_loss":
                        best,viable=expert.labels(env)
                        if best.any() and not viable[a]:
                            a=max((i for i in range(8) if best[i]),key=lambda i:score[i])
                    elif variant=="exact_descent":
                        fuel=(v.fuel-v.fuel_distance)<=v.health-v.repair_distance
                        distance=v.fuel_distance if fuel else v.repair_distance
                        slack=(v.fuel if fuel else v.health)-distance
                        if 0<distance and slack<=3:
                            ds=expert.dist[:2] if fuel else expert.dist[2:4]
                            closer=[i for i,m in enumerate(MOVES) if mask[i] and
                                    min(field[destination(env.level.grid,env.position,m)] for field in ds)<distance]
                            if closer:
                                a=max(closer,key=lambda i:score[i])
                    cache[v]=(a,base)
                a,base=cache[v]
                interventions+=int(a!=base)
                env.step(MOVES[a])
            expert.remaining.cache_clear()
            records.append(dict(map=r["name"],outcome=env.status.name,steps=env.steps,refuels=env.refuels,
                                repairs=env.repairs,interventions=interventions))
        experiments[variant]=dict(episodes=records,outcomes=dict(Counter(r["outcome"] for r in records)),
                                  both=sum(bool(r["refuels"] and r["repairs"]) for r in records))
        print(variant,experiments[variant]["outcomes"],experiments[variant]["both"],flush=True)
    save(HERE/"train-interventions.json",dict(weights_sha256=digest(path),split_sha256=digest(HERE/"train.jsonl"),
         train_range=[800,900],slack=3,variants=experiments))


if __name__=="__main__":
    main()
