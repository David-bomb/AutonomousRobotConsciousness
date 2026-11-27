"""Train-only paired measurements at early irreversible forks of the 76-win mixture."""
from collections import Counter
from hashlib import sha256
import site
import sys
import numpy as np
import torch
from renewable_train import HERE,Environment,Status,data,digest,moves,save
from service_window_policy import load,coverage
from maze.grid import destination,locations
# The legacy teacher uses the SciPy already installed in this interpreter's user site.
sys.path.append(site.getusersitepackages())
from maze.mandatory_supply import MandatoryProof
from maze.pilot_automaton import Mode
from maze.proven_approach import certified
from maze.route_shield import completion_status
from scripts.audit_automaton import Reachability

OUT=HERE/"early-errors"
MODEL=HERE/"service-window/fixed/model.pt"


def score_fork(env,v,old,oracle,model,chosen,position):
    ds=(v.fuel_distance,v.repair_distance)
    k=min((i for i in (0,1) if ds[i]>0),key=lambda i:(v.fuel,v.health)[i]-ds[i],default=0)
    legal=[a for a,m in enumerate(moves) if destination(v.memory,v.position,m) is not None]
    viable=[];descending=[]
    physical=oracle.services[k]
    for a in legal:
        q=destination(v.memory,v.position,moves[a])
        if oracle.possible(q,*oracle.after(q,v.fuel,v.health),env.limit-position-1):viable.append(a)
        if physical[q]<ds[k]:descending.append(a)
    x,y=v.position;goal=x+v.dx,y+v.dy
    prior=(x-v.last_move.value[0],y-v.last_move.value[1]) if v.moved else None
    sites=tuple(tuple(locations(v.memory,c)) for c in "FR")
    archive=MandatoryProof()
    if old[0] is not None:archive._fallback.last_lengths=dict(zip((Mode.FUEL,Mode.REPAIR),old))
    demonstration=moves.index(archive.choose(v))
    votes=MandatoryProof().direction_votes(v,ds,old,prior,sites)
    own=np.asarray([float(votes[k].get(m,0)) for m in moves])
    sure,possible=certified(v.memory,v.position,ds[k],old[k],prior,sites[k])
    best=max(legal,key=lambda a:(moves[a] in sure,own[a],a==chosen))
    statuses={moves[a].name:completion_status(v,goal,ds,prior,old,moves[a],limit=24)[0] for a in legal}
    non=[a for a in legal if statuses[moves[a].name]!="impossible"]
    shield=max(non or legal,key=lambda a:(a==chosen,moves[a] in sure,own[a]))
    late=int(model.late_scores(v).argmax())
    return dict(step=position,stock=[v.fuel,v.health],readings=ds,previous=old,urgent="FR"[k],
                known_windows=coverage(v.memory),legal=legal,viable=viable,down=descending,chosen=chosen,
                teacher=demonstration,votes=best,possible=[a for a in legal if moves[a] in possible],
                certain=[a for a in legal if moves[a] in sure],numeric_votes=own.tolist(),shield=shield,late=late,statuses=statuses)


def main():
    torch.set_num_threads(1)
    assert digest(MODEL)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    OUT.mkdir(exist_ok=True)
    model=load(MODEL.read_bytes())
    rows=data.rows(HERE/"handover-search/train.jsonl")
    save(OUT/"protocol.json",dict(model=digest(MODEL),bank=digest(HERE/"handover-search/train.jsonl"),
        n=len(rows),only_train=True,code=digest(HERE/"early_postmortem.py"),
        rule="Stop baseline prefix at fully open service windows or terminal. At first irreversible early action compare numerical votes, existential feasibility and cold teacher on identical observations with two previous real readings. On successful handoffs take first comparable tight fork at which the baseline chose a viable move. Full map labels are offline only."))
    results=[]
    for i,row in enumerate(rows):
        p=OUT/(row["name"]+".json")
        if p.exists():results.append(data.read(p));continue
        env=Environment(data.level(row));prev=(None,None);track=[];commands=bytearray();seen=set()
        while env.status==Status.RUNNING and min(coverage(env.view.memory))<9:
            v=env.view;key=v,prev
            if key in seen:break
            seen.add(key)
            a=int(model.early_scores(v,prev).argmax());track.append((v,prev,a))
            commands.append(a);prev=(v.fuel_distance,v.repair_distance);env.step(moves[a])
        reached=min(coverage(env.view.memory))==9
        doc=dict(map=row["name"],tier=row["tier"],gate=reached,steps=env.steps,outcome=env.status.name,actions_sha256=sha256(commands).hexdigest())
        oracle=Reachability(env.level)
        if not reached and env.status in (Status.EMPTY,Status.BROKEN):
            def alive(step):
                vv=track[step][0] if step<len(track) else env.view
                return oracle.possible(vv.position,vv.fuel,vv.health,env.limit-step)
            lo,hi=0,len(track)
            assert alive(lo) and not alive(hi)
            while hi-lo>1:
                mid=(lo+hi)//2
                if alive(mid):lo=mid
                else:hi=mid
            rec=score_fork(env,*track[lo][:2],oracle,model,track[lo][2],lo)
            assert rec["chosen"] not in rec["viable"] and rec["viable"]
            doc.update(reason="first-loss",fork=rec)
        elif reached:
            for t,(v,old,a) in enumerate(track):
                if min(v.fuel-v.fuel_distance,v.health-v.repair_distance)>2:continue
                edges=oracle.edges[v.position]
                if len(edges)<2:continue
                avail=[oracle.possible(q,*oracle.after(q,v.fuel,v.health),env.limit-t-1) for _,q in edges]
                if all(avail) or not any(avail):continue
                rec=score_fork(env,v,old,oracle,model,a,t)
                if a in rec["viable"]:
                    doc.update(reason="matched-viable",fork=rec);break
        save(p,doc);results.append(doc)
        if (i+1)%25==0:print(i+1,Counter(x.get("reason","other") for x in results),flush=True)
    groups={}
    for name in ("first-loss","matched-viable"):
        group=[x["fork"] for x in results if x.get("reason")==name]
        groups[name]=dict(n=len(group),**{k:sum(s[k] in s["viable"] for s in group)
                                          for k in ("chosen","teacher","votes","shield","late")},
                          any_sure=sum(bool(s["certain"]) for s in group))
    save(OUT/"summary.json",dict(groups=groups,maps=len(results),reached=sum(e["gate"] for e in results)))
    print(groups)


if __name__=="__main__":main()
