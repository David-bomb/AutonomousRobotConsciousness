"""First irreversible late action on independent training trips of the fixed mixture."""
import ctypes
import numpy as np
import torch
from renewable_train import HERE,Environment,Status,data,digest,moves,save
from renewable_policy import targets
from service_window_policy import load
from scripts.audit_automaton import Reachability
from maze.grid import destination

OUT=HERE/"late-errors"


def cohorts():
    labels={}
    for name in ("fit","select"):
        for sub in ("service-window/learned","handover-search"):
            for e in data.read(HERE/sub/(name+".json"))["episodes"]:
                if e["model"]!=0:continue
                hit=e["gate"] if "gate" in e else e["first_late"] is not None
                if hit:labels[e["map"]]=e["outcome"]=="SUCCESS"
    indexed={r["name"]:r for f in ("return-search/bank.jsonl","handover-search/train.jsonl") for r in data.rows(HERE/f)}
    selected=[]
    for t in ("H1","H2","H3","H4"):
        group=sorted(n for n in labels if indexed[n]["tier"]==t)
        failed=[n for n in group if not labels[n]];success=[n for n in group if labels[n]]
        selected.extend((indexed[n],labels[n]) for n in failed+success[:len(failed)])
    return selected


def masks(lib,v):
    w=len(v.memory[0]);x,y=v.position;out=np.zeros(7,np.int32)
    lib.proper("".join(v.memory).encode("ascii"),w,len(v.memory),y*w+x,targets(v.memory,(x+v.dx,y+v.dy)),v.fuel,v.health,out)
    return out


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    parent=HERE/"service-window/fixed/model.pt";model=load(parent.read_bytes())
    lib=ctypes.CDLL(str(HERE/"proper_frontier.dll"));ci=ctypes.c_int;ints=np.ctypeslib.ndpointer(np.int32)
    lib.proper.argtypes=[ctypes.c_char_p,ci,ci,ci,ints,ci,ci,ints];lib.proper.restype=None
    selected=cohorts()
    save(OUT/"protocol.json",dict(model=digest(parent),cohort=[[r["name"],win] for r,win in selected],source=digest(HERE/"late_errors.py"),
        method="Use recorded fixed-parent training outcomes on the two established independent banks. Include every failure after handover and an equal number of same-tier wins when available. Replay and binary-search first loss of true physical reachability. Stratify whether a known goal or a guaranteed return route to more exploration still exists. Compare viable directions at the exact same physical fork. For matched wins take first low-resource, multi-action late fork without a known goal or safe new window."))
    details=[]
    for r,expected in selected:
        p=OUT/(r["name"]+".json")
        if p.exists():details.append(data.read(p));continue
        env=Environment(data.level(r));hist=[];cmd=[]
        past=(None,None);gate=None;actions={}
        while env.status==Status.RUNNING:
            v=env.view
            if model.route(v,model.cfg) and gate is None:gate=env.steps
            hist.append((v,past))
            a=moves.index(model.choose(v,actions));cmd.append(a);past=v.fuel_distance,v.repair_distance;env.step(moves[a])
        assert expected==(env.status==Status.SUCCESS) and gate is not None
        oracle=Reachability(env.level)
        def can(t):
            view=hist[t][0] if t<len(hist) else env.view
            return oracle.possible(view.position,view.fuel,view.health,env.limit-t)
        if not expected and can(gate):
            if env.status==Status.LIMIT:
                reason="limit"
            else:reason="first-loss"
            lo,hi=gate,len(hist)
            while hi-lo>1:
                mid=(lo+hi)//2
                if can(mid):lo=mid
                else:hi=mid
            step=lo
        elif not expected:
            reason="already-lost-at-handover";step=gate
        else:
            reason="matched";step=None
            for j in range(gate,len(hist)):
                v=hist[j][0]
                if min(v.fuel,v.health)>35:continue
                if sum(destination(v.memory,v.position,m) is not None for m in moves)<2:continue
                ms=masks(lib,v)
                if not (ms[0] or ms[1]):step=j;break
        record=dict(map=r["name"],tier=r["tier"],family=r["family"],win=expected,gate=gate,steps=len(hist),
                    outcome=env.status.name,reason=reason,step=step)
        if step is not None:
            v,old=hist[step];a=cmd[step];ms=masks(lib,v);possible=[]
            for j,m in enumerate(moves):
                q=destination(v.memory,v.position,m)
                if q is not None and oracle.possible(q,*oracle.after(q,v.fuel,v.health),env.limit-step-1):possible.append(j)
            x,y=v.position
            record["fork"]=dict(chosen=a,viable=possible,known_goal=int(ms[0]),safe_window=int(ms[1]),
                  risk_window=int(ms[4]),fuel=v.fuel,hp=v.health,readings=[v.fuel_distance,v.repair_distance],prior=list(old),
                  exit_chebyshev=max(abs(v.dx),abs(v.dy)),known_cells=sum(len(row)-row.count("?") for row in v.memory),
                  alternatives=[i for i in range(8) if ms[4]&(1<<i)],late_scores=model.late_scores(v).tolist())
            if reason=="first-loss":assert a not in possible and possible
        save(p,record);details.append(record)
    from collections import Counter
    failures=[e for e in details if not e["win"]]
    result=dict(maps=len(details),late_failures=len(failures),reasons=dict(Counter(e["reason"] for e in failures)),
        known_goal=sum(bool(r["fork"]["known_goal"]) for r in failures),safe_window=sum(bool(r["fork"]["safe_window"]) for r in failures),
        risk_window=sum(bool(r["fork"]["risk_window"]) for r in failures),by_family=dict(Counter(r["family"] for r in failures)))
    save(OUT/"summary.json",dict(summary=result,episodes=details));print(result,flush=True)


if __name__=="__main__":main()
