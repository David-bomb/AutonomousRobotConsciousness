from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import destination
from maze.levels import Level
from ...data import rows,level,save,digest
from ...policy import MOVES
from ...teacher import Teacher
from ..spatial.model import image,SpatialSensor
from .features import HERE,features
from .itineraries import measures


def main():
    table=np.load(HERE/"prior.npy").tobytes()
    prior=torch.load(HERE.parent/"runs/mission600/rl.pt",weights_only=True,map_location="cpu")["wall_prior"].numpy().tobytes()
    cases=rows(HERE.parent/"train.jsonl")[300:325]
    examples=positives=wrong=missed=0
    rng=np.random.default_rng(1619)
    for row in cases:
        env=Environment(level(row));teacher=Teacher(env.level)
        for _ in range(9):
            v=replace(env.view,memory=env.level.grid)
            evidence=features(v,table)
            observed=image(env.view)[0]
            for k,ds in enumerate((teacher.dist[:2],teacher.dist[2:4])):
                current=min(d[v.position] for d in ds)
                if current==0:continue
                index=1-k if v.health<v.fuel else k
                seen=False
                for a,m in enumerate(MOVES):
                    q=destination(v.memory,v.position,m)
                    if q is None:continue
                    dx,dy=m.value
                    bit=observed[1,v.position[1]+dy,v.position[0]+dx]
                    assert int(bit)&(1<<a)
                    correct=min(d[q] for d in ds)<current
                    positive=evidence[a,index*6+2]>1e-7
                    positives+=int(positive);wrong+=int(positive and not correct)
                    seen|=positive and correct
                missed+=int(not seen)
                examples+=1
            best,_=teacher.labels(env)
            if not best.any():break
            env.step(MOVES[int(rng.choice(np.flatnonzero(best)))])
            if env.status!=Status.RUNNING:break
        teacher.remaining.cache_clear()
    assert wrong==0 and missed==0
    corridor="#################\n#A.F.R....B.F.R.#\n#################"
    scenario=Level.parse("check\n100 100\n"+corridor)
    sim=Environment(scenario)
    memory=(scenario.grid[0],"#A.F.R??????F?R?#",scenario.grid[2])
    v=replace(sim.view,memory=memory)
    fields=measures(v,prior)
    assert fields[1,0,0,0]==9 and fields[1,0,2,0]==9
    save(HERE/"checks.json",dict(train_maps=len(cases),nonzero_readings=examples,positive_signals=positives,
         unsupported_signals=wrong,empty_true_direction=missed,frontier_full_route=float(fields[1,0,2,0]),
         sources={n:digest(HERE/n) for n in ("walks.cpp","walks.dll","itineraries.cpp","itineraries.dll")}))
    print(examples,positives,wrong,missed,fields[1,0,2,0],flush=True)


if __name__=="__main__":
    main()
