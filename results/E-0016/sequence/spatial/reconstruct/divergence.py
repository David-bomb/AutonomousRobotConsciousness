"""Разбор первой расходящейся команды на учебных стартах с изменившимся SUCCESS."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch

from maze.environment import Environment, Status
from ....data import level, digest, read, save
from ....teacher import Teacher
from .bank import HERE, RUN, maps
from .features import MOVES, frame, phase, sign_index
from .model import Policy, FIELDS


def initialize():
    global net
    torch.set_num_threads(1)
    net=Policy(torch.load(RUN/"marginal1.pt",map_location="cpu",weights_only=True))


@torch.inference_mode()
def probe(job):
    row, gain=job
    env=Environment(level(row));prev=None
    while env.status==Status.RUNNING:
        v=env.view
        if phase(v.memory):
            raise AssertionError("Два одинаковых префикса дошли до общей поздней сети")
        d=frame(net.early,v)
        current=np.array([v.fuel_distance,v.repair_distance],np.int16)
        label=sign_index(prev,current)
        missing=int(d["missing"].argmax())
        p,_=net.net(*(torch.as_tensor(d[k])[None] for k in FIELDS))
        q=p[0].softmax(-1).numpy()@d["options"] if label<9 else d["missing"]
        selected=int(np.where(d["legal"],q,-np.inf).argmax())
        if missing!=selected:
            teacher=Teacher(env.level);_,viable=teacher.labels(env);teacher.remaining.cache_clear()
            honest=int(d["options"][label].argmax())
            return dict(map=row["name"],tier=row["tier"],terminal_gain=gain,step=env.steps+1,
               slack=min(v.fuel-v.fuel_distance,v.health-v.repair_distance),pair=label,
               inferred=int(p[0].argmax()),previous=missing,marginal=selected,true_pair=honest,
               previous_safe=bool(viable[missing]),marginal_safe=bool(viable[selected]),true_safe=bool(viable[honest]))
        prev=current
        env.step(MOVES[missing])
    raise AssertionError("Разные терминалы при совпавших действиях")


def main():
    reg=read(HERE/"registration.json")["selection_order"]
    db=read(RUN/"selection.json");pairs=[]
    m,g=reg.index("missing"),reg.index("marginal1")
    for row,record in zip(maps("selection"),db["records"],strict=True):
        assert record["map"]==row["name"]
        a,b=[record["episodes"][i]["outcome"]=="SUCCESS" for i in (m,g)]
        if a!=b:pairs.append((row,int(b)-int(a)))
    with ProcessPoolExecutor(2,initializer=initialize) as pool:
        items=list(pool.map(probe,pairs))
    totals={}
    for k,tag in ((1,"gained"),(-1,"lost")):
        group=[r for r in items if r["terminal_gain"]==k]
        totals[tag]=dict(maps=len(group),median_step=float(np.median([r["step"] for r in group])) if group else None,
             both_viable=sum(r["previous_safe"] and r["marginal_safe"] for r in group),
             old_only=sum(r["previous_safe"] and not r["marginal_safe"] for r in group),
             new_only=sum(r["marginal_safe"] and not r["previous_safe"] for r in group),
             exact_pair=sum(r["inferred"]==r["pair"] for r in group),
             matches_true=sum(r["marginal"]==r["true_pair"] for r in group))
    save(RUN/"divergence-train.json",dict(selection=digest(RUN/"selection.json"),source=digest(HERE/"divergence.py"),
         comparison="marginal1 vs missing",groups=totals,rows=items))
    print(totals,flush=True)


if __name__=="__main__":main()
