from collections import Counter
import numpy as np
import torch
from maze.environment import Environment
from ...data import digest,read,rows,level,save
from ...policy import MOVES
from ..model import SequenceNet,encode
from ..spatial.model import SpatialSensor,image
from .features import HERE
from .paths import confidence


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    cases={r["map"]:r for r in read(HERE/"hard-probe.json")["rows"]}
    base=SequenceNet().eval()
    base.load_state_dict(torch.load(HERE.parent/"runs/mission600/rl.pt",weights_only=True,map_location="cpu"))
    prior=base.wall_prior.numpy().tobytes()
    cnn=SpatialSensor().eval()
    cnn.load_state_dict(torch.load(HERE.parent/"spatial/adapt-400/s1800.pt",weights_only=True,map_location="cpu"))
    table=np.load(HERE/"prior.npy").tobytes()
    totals=Counter();logs=[]
    for row in rows(HERE.parent/"train.jsonl")[900:1000]:
        ref=cases.get(row["name"])
        if ref is None:continue
        env=Environment(level(row));cache={}
        while env.steps+1<ref["step"]:
            if env.view not in cache:
                x,ok=encode(env.view,prior)
                cache[env.view]=int(base(torch.from_numpy(x[None]),torch.from_numpy(ok[None])).argmax())
            env.step(MOVES[cache[env.view]])
        v=env.view;x,ok=encode(v,prior)
        raw=base(torch.from_numpy(x[None]),torch.from_numpy(ok[None]))[0].numpy()
        q=confidence(v,table,64)
        img,m=image(v)
        z,mask=cnn(torch.from_numpy(img[None]),torch.from_numpy(m[None]))
        assert np.array_equal(ok,mask[0].numpy())
        k=int(v.health-v.repair_distance<v.fuel-v.fuel_distance)
        p=z.softmax(-1)[0,:,k,0].numpy()
        guesses={}
        for name,score in (("prior",q[:,0]),("consistent",q[:,1]),("cnn",p),("together",q[:,1]+p)):
            a=int(np.where(ok,score+.000001*np.clip(raw,-20,20),-1e6).argmax())
            totals[name+"_down"]+=int(a in ref["downhill"])
            totals[name+"_viable"]+=int(a in ref["viable"])
            guesses[name]=a
        totals["covered"]+=int(q[:,1].sum()>0)
        logs.append(dict(map=row["name"],choices=guesses,acceptance=q[:,3].tolist()))
    save(HERE/"paths-probe.json",dict(reference=digest(HERE/"hard-probe.json"),states=len(logs),trials_per_direction=64,
         summary=dict(totals),records=logs))
    print(totals,flush=True)


if __name__=="__main__":
    main()
