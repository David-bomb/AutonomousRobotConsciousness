"""Прогноз сокращения реального показания на первых необратимых учебных развилках."""
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment
from maze.grid import destination
from ...data import digest,level,read,rows,save
from ...policy import MOVES
from ...teacher import Teacher
from ..model import SequenceNet,encode
from .features import HERE,features


@torch.inference_mode()
def main():
    source=HERE.parent/"routing-audit.json"
    cases={r["map"]:r["first_irreversible"] for r in read(source)["episodes"] if r["first_irreversible"]}
    state=HERE.parent/"runs/mission600/rl.pt"
    net=SequenceNet().eval()
    net.load_state_dict(torch.load(state,weights_only=True,map_location="cpu"))
    torch.set_num_threads(1)
    prior=net.wall_prior.numpy().tobytes()
    tables=np.load(HERE/"prior.npy").tobytes()
    totals=Counter()
    saved=[]
    for row in rows(HERE.parent/"train.jsonl")[900:1000]:
        loss=cases.get(row["name"])
        if not loss:continue
        env=Environment(level(row))
        cache={}
        for i in range(loss["step"]):
            v=env.view
            if v not in cache:
                x,legal=encode(v,prior)
                g,grad=net.heads(torch.from_numpy(x[None]),torch.from_numpy(legal[None]))
                cache[v]=(g[0].numpy(),grad[0].softmax(-1).numpy(),legal)
            raw,grad,legal=cache[v]
            if i+1==loss["step"]:
                break
            env.step(MOVES[int(raw.argmax())])
        work=features(v,tables)
        s,group,d,offset=min((v.fuel-v.fuel_distance,0,v.fuel_distance,0),(v.health-v.repair_distance,1,v.repair_distance,2))
        k=1-group if v.health<v.fuel else group
        obs=work[:,k*6:k*6+3]
        teacher=Teacher(env.level)
        _,viable=teacher.labels(env)
        actual=np.array([bool(legal[a] and min(ds[destination(v.memory,v.position,m)] for ds in teacher.dist[offset:offset+2])<d)
                         for a,m in enumerate(MOVES)])
        teacher.remaining.cache_clear()
        modes={"network_sensor":grad[:,k,0],"walk_1":obs[:,0],"walk_sqrt":obs[:,1],"walk_squared":obs[:,2]}
        comparison={}
        for name,score in modes.items():
            a=int(np.argmax(np.where(legal,score+.000001*np.clip(raw,-10,10),-1e5)))
            totals[name+"_down"]+=int(actual[a])
            totals[name+"_viable"]+=int(viable[a])
            comparison[name]=a
        totals["states"]+=1
        totals["covered"]+=int(np.max(obs)>0)
        totals["has_downhill"]+=int(actual.any())
        saved.append(dict(map=row["name"],step=loss["step"],slack=s,chosen=comparison,viable=np.flatnonzero(viable).tolist(),
                          downhill=np.flatnonzero(actual).tolist(),mixture=work[0,-4:].tolist()))
    save(HERE/"hard-probe.json",dict(source_sha256=digest(source),weights_sha256=digest(state),summary=dict(totals),rows=saved))
    print(totals,flush=True)


if __name__=="__main__":
    main()
