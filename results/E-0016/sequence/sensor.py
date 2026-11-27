"""Отдельная проверка научения прогнозу изменения обоих штатных датчиков."""
import time
import numpy as np
import torch
from .bank import HERE
from .model import SequenceNet
from .train import dataset,diagnose,sensor_loss
from ..data import digest,save


def main():
    if (HERE/"sensor.pt").exists():
        raise FileExistsError(HERE/"sensor.pt")
    seed=1610
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=SequenceNet()
    for name,p in net.named_parameters():
        p.requires_grad_(name.startswith("sensor_"))
    data=dataset([("train",600),("partial",100)],"cpu")
    probe=dataset([("probe",100)],"cpu")
    before=diagnose(net,probe)
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),lr=.001)
    t=time.monotonic()
    for epoch in range(6):
        order=rng.permutation(len(data["x"]))
        for start in range(0,len(order),256):
            ids=torch.from_numpy(order[start:start+256])
            mask=data["legal"][ids]
            _,grad=net.sensor(data["x"][ids].float(),mask)
            loss=sensor_loss(grad,data["gradient"][ids],mask,data["slack"][ids])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
        print("sensor",epoch+1,round(time.monotonic()-t,1),flush=True)
    weights={k:v for k,v in net.state_dict().items() if k.startswith("sensor_")}
    torch.save(weights,HERE/"sensor.pt")
    after=diagnose(net,probe)
    save(HERE/"sensor.json",dict(seed=seed,epochs=6,device="cpu",updates=6*((len(order)+255)//256),
         before=before,after=after,seconds=time.monotonic()-t,weights_sha256=digest(HERE/"sensor.pt"),
         sources={f:digest(HERE/f) for f in ("sensor.py","model.py","train.py","prepare-train-600.json","prepare-probe-100.json","prepare-partial-100.json")}))
    print(before,after,flush=True)


if __name__=="__main__":
    main()
