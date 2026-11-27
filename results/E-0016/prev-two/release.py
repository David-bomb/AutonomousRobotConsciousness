"""Проверка выбранного post-RL и интервалы разности долей физических финишей."""
from math import comb
import numpy as np
from .common import HERE,SEED,read,write
from .compare import rank
from . import audit,final_check,phase_audit


def differences(ref,run):
    deltas=[]
    for a,b in zip(ref["episodes"],run["episodes"],strict=True):
        assert a["geometry"]==b["geometry"] and a["tier"]==b["tier"]
        deltas.append((a["tier"],int(b["outcome"]=="SUCCESS")-int(a["outcome"]=="SUCCESS")))
    rng=np.random.default_rng(SEED+401)
    bootstrap=np.zeros(10000)
    for tier in ("H1","H2","H3","H4"):
        group=[d for t,d in deltas if t==tier]
        assert len(group)==25
        bootstrap+=rng.choice(group,size=(10000,len(group))).sum(1)
    gains=sum(d>0 for _,d in deltas);losses=sum(d<0 for _,d in deltas)
    n=gains+losses
    return dict(gains=gains,losses=losses,difference_pp=gains-losses,
                paired_stratified_95_pp=np.quantile(bootstrap,[.025,.975]).tolist(),
                descriptive_exact_p=min(1,2*sum(comb(n,t) for t in range(min(gains,losses)+1))/(2**n)))


def main():
    tags=["wave1-pg","wave1","route-rl","post-teacher-rl"]
    tag=max(tags,key=rank)
    print("selected",tag,flush=True)
    audit.audit(tag)
    phase_audit.main(tag)
    x=read(HERE/"dev"/(tag+"-honest.json"))
    write(HERE/"paired.json",dict(selected=tag,bootstrap_replicates=10000,seed=SEED+401,
          versus_baseline=differences(read(HERE/"dev/mission600-baseline.json"),x),
          versus_ablation=differences(read(HERE/"dev"/(tag+"-ablated.json")),x)))
    final_check.verify(tag)


if __name__=="__main__":main()
