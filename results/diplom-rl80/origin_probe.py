"""Check the inferred origin/elapsed bounds against hidden train truth after computing them."""
from collections import Counter
import math
import torch
from renewable_train import HERE,BASE,BASE_HASH,Environment,Status,data,digest,save
from geodesic_probe import prev
from service_window_policy import coverage
from origin_trace import origins

def main():
    torch.set_num_threads(1)
    out=HERE/"origin-search";out.mkdir(exist_ok=True)
    bank=HERE/"handover-search/train.jsonl"
    save(out/"trace-protocol.json",dict(bank=digest(bank),sources={n:digest(HERE/n) for n in ("origin_trace.py","origin_probe.py")},
         scope="First 100 training geometries. First 35 moves of wave1 before any fully observed F/R window. Reconstruct possible A and elapsed lower bounds only from the union of acquired 3x3 windows, position and last move. Compare with actual A and steps only afterwards."))
    states=[]
    net=prev.Controller(BASE,BASE_HASH)
    for row in data.rows(bank)[:100]:
        env=Environment(data.level(row));net.reset()
        while env.status==Status.RUNNING and env.steps<=35 and max(coverage(env.view.memory))<9:
            v=env.view
            allowed=origins(v)
            w=len(v.memory[0]);x,y=env.level.start
            real=y*w+x
            candidates={a:(low,straight) for a,low,straight in allowed}
            assert real in candidates and candidates[real][0]<=env.steps,(row["name"],env.steps,candidates.get(real))
            lo=min(q[0] for q in candidates.values())
            forced=[u for u,(cost,_) in candidates.items() if cost==lo]
            states.append(dict(map=row["name"],step=env.steps,n=len(allowed),width=w,truth=real,
                true_lower=candidates[real][0],best=lo,best_count=len(forced),true_in_best=real in forced))
            env.step(net.choose(v))
    def report(st):
        return dict(n=len(st),true_among_shortest=sum(r["true_in_best"] for r in st),
                    unique_shortest_correct=sum(r["true_in_best"] and r["best_count"]==1 for r in st),
                    true_time_exact=sum(r["true_lower"]==r["step"] for r in st),
                    lower_within_two=sum(r["step"]-r["true_lower"]<=2 for r in st))
    hist={k:report([r for r in states if lo<=r["step"]<=hi]) for k,lo,hi in (("all",0,35),("first_hops",1,5),("later",6,35))}
    save(out/"trace-probe.json",dict(groups=hist,rows=states))
    print(hist,flush=True)

if __name__=="__main__":main()
