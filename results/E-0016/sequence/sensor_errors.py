"""Прогноз на точных первых необратимых ходах замороженного пилота. Только train."""
from collections import Counter
import numpy as np
import torch
from maze.environment import Environment
from maze.grid import destination
from .model import SequenceNet,encode
from .bank import HERE
from ..adapter import Controller
from ..data import OUT,digest,level,rows,read,save
from ..policy import MOVES
from ..priors import fit_prior
from ..teacher import Teacher


def main():
    path=OUT/"runs/recovery800/rl.pt"
    net=Controller(path,digest(path))
    sensor=SequenceNet().eval()
    sensor.load_state_dict(torch.load(HERE/"sensor.pt",weights_only=True,map_location="cpu"),strict=False)
    train=rows(OUT/"data/train.jsonl")
    prior=fit_prior(train).tobytes()
    metadata={r["map"]:r for r in read(path.with_suffix(".one-change.json"))["episodes"]}
    counts=Counter()
    outcomes=[]
    for row in train[700:800]:
        r=metadata[row["name"]]
        if "alternatives" not in r:
            continue
        env=Environment(level(row))
        for _ in range(r["step"]-1):
            env.step(net.choose(env.view))
        v=env.view
        x,mask=encode(v,prior)
        with torch.inference_mode():
            _,g=sensor.sensor(torch.from_numpy(x[None]),torch.from_numpy(mask[None]))
        use_f=(v.fuel-v.fuel_distance)<=v.health-v.repair_distance
        slot=int((not use_f) if v.fuel<=v.health else use_f)
        conf=g[0,:,slot,:].softmax(-1)[:,0].numpy()
        conf[~mask]=-1
        a=int(conf.argmax())
        truth=Teacher(env.level)
        ds=truth.dist[:2] if use_f else truth.dist[2:4]
        down=[i for i,m in enumerate(MOVES) if mask[i] and min(d[destination(env.level.grid,v.position,m)] for d in ds)
              <(v.fuel_distance if use_f else v.repair_distance)]
        good={alt["move"] for alt in r["alternatives"]}
        counts["cases"]+=1
        counts["predicted_decreases"]+=int(a in down)
        counts["predicted_viable"]+=int(MOVES[a].name in good)
        counts["policy_decreases"]+=int(MOVES.index(net.choose(v)) in down)
        counts["any_downhill_viable"]+=int(any(MOVES[i].name in good for i in down))
        counts["uniform_downhill_expected"]+=len(down)/int(mask.sum())
        outcomes.append(dict(map=r["map"],step=r["step"],resource="fuel" if use_f else "health",
                             predicted=MOVES[a].name,downhill=[MOVES[i].name for i in down],viable=sorted(good)))
    save(HERE/"sensor-first-loss.json",dict(reference_sha256=digest(path),sensor_sha256=digest(HERE/"sensor.pt"),
         summary=counts,episodes=outcomes))
    print(dict(counts))


if __name__=="__main__":
    main()
