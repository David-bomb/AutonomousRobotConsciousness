"""Обновление нелинейных оценок на ошибках самого студента до обоих окон."""
import os
for key in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from collections import defaultdict
import sys
import numpy as np
import torch
from torch.nn import functional as F
from .common import HERE,SEED,digest,read
from .early_bank import FOLDER
from .fit import loadnet,freeze


def fit(stage,threads=2):
    parent=("wave1","early600","early1200")[stage-1]
    tag=("early600","early1200","early1800")[stage-1]
    torch.set_num_threads(threads);torch.manual_seed(SEED+500+stage)
    net=loadnet(HERE/"weights"/(parent+".pt"));net.requires_grad_(False)
    for layer in (net.sign,net.edge,net.action,net.gains,net.risk):layer.requires_grad_(True)
    data=defaultdict(list);fingerprints={}
    for batch in ["fit1","fit2","fit3"][:stage]:
        source=FOLDER/(batch+"-samples.json");fingerprints[batch]=digest(source)
        for name,item in read(source)["records"].items():
            if not item["examples"]:continue
            path=FOLDER/batch/(name+".npz");assert digest(path)==item["sha256"]
            with np.load(path) as arrays:
                for k in arrays:data[k].append(arrays[k])
                for k in ("correction","later"):
                    if k not in arrays:data[k].append(np.zeros(item["examples"],bool))
            data["higher"].append(np.full(item["examples"],item["tier"] in ("H3","H4")))
    d={k:np.concatenate(v) for k,v in data.items()}
    rng=np.random.default_rng(SEED+510+stage)
    pools=[np.flatnonzero(v) for v in (d["viable"].any(1),d["hard"],d["hard"]&d["higher"],
                                        (d["teacher"]>=0)&d["higher"],d["higher"]&(d["slack"]<=6))]
    sizes=(44,28,24,16,16)
    if stage==3:
        changed=np.flatnonzero(d["correction"])
        repeated=np.flatnonzero(d["correction"]&d["later"])
        if not len(repeated):repeated=changed
        pools.extend((changed,repeated))
        sizes=(32,16,16,12,12,24,16)
    assert all(len(p) for p in pools)
    params=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.00020 if stage==1 else .00011)
    idx=torch.arange(128);logs=[]
    for step in range(1,601):
        ids=np.concatenate([rng.choice(pool,n) for pool,n in zip(pools,sizes)])
        b={k:torch.from_numpy(v[ids]) for k,v in d.items()}
        legal=b["legal"];viable=b["viable"]&legal
        scores,signs,risk=net.heads(b["core"].float(),b["grad"].float(),b["past"].float(),b["possible"],legal)
        logp=scores.log_softmax(1);urgent=b["urgent"]
        truth=b["sign"].gather(2,urgent[:,None,None].expand(-1,8,1)).squeeze(2)
        sign_loss=F.cross_entropy(signs[legal].reshape(-1,3),b["sign"][legal].flatten())
        risk_loss=F.binary_cross_entropy_with_logits(risk[legal],viable[legal].float())
        best=truth.masked_fill(~viable,5).amin(1,keepdim=True)
        close=(truth==best)&viable
        target=torch.where((b["slack"]<=6)[:,None],close,viable)
        valid=viable.any(1)
        target_loss=-logp.masked_fill(~target,-1e5).logsumexp(1)[valid].mean()
        teach=b["teacher"];teach_mask=teach>=0
        teacher_loss=-logp[idx[teach_mask],teach[teach_mask]].mean()
        if stage==3:
            intervention=b["correction"]&teach_mask
            teacher_loss=.5*teacher_loss+1.2*(-logp[idx[intervention],teach[intervention]].mean())
        parent_action=b["action"];keep=valid&target[idx,parent_action]
        anchor=F.kl_div(logp[keep],b["prior"][keep].softmax(1),reduction="batchmean") if keep.any() else 0.
        loss=1.25*sign_loss+.45*risk_loss+target_loss+.48*teacher_loss+1.3*anchor
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.4);opt.step()
        if step%100==0:logs.append([round(float(v.detach()),5) for v in (sign_loss,target_loss,teacher_loss)])
    freeze(net,tag,dict(parent=parent,steps=600,total=len(d["action"]),pools=[len(p) for p in pools],
                         samples=fingerprints,objective="all legal real sensor signs, viability, cold train teacher",logs=logs))
    print("early trained",tag,[len(p) for p in pools],flush=True)


if __name__=="__main__":fit(*map(int,sys.argv[1:]))
