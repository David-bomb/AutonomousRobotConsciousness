"""Полные учебные исходы по сложности и по сервисам, достигнутым к терминалу."""
import argparse
from collections import Counter
from .model import HERE
from ...data import read,save,digest


def summarize(items):
    finish=Counter(r["outcome"] for r in items)
    groups=Counter("both" if r["refuels"] and r["repairs"] else "one" if r["refuels"] or r["repairs"] else "before"
                   for r in items if r["outcome"]!="SUCCESS")
    return dict(count=len(items),outcomes=dict(finish),both=sum(bool(r["refuels"] and r["repairs"]) for r in items),
                loss_at_terminal=dict(groups))


def main(run):
    root=HERE.parent/"runs"/run
    source=root/("train.json" if run=="calibrated600" else "selection.json")
    data=read(source)
    rank=data["scores"] if "scores" in data else data["rank"]
    ix=max(range(len(rank)),key=lambda j:(rank[j]["wins"],rank[j]["minimum"]))
    raw=next(j for j,r in enumerate(rank) if all(i==0 for i in r["weights"][1:]))
    records=data["records"]
    scores={}
    for name,j in (("before",raw),("after",ix)):
        by_tier={}
        for t in ("H1","H2","H3","H4"):
            items=[(r["episodes"] if "episodes" in r else r["outcomes"])[j] for r in records if r["tier"]==t]
            by_tier[t]=summarize(items)
        scores[name]=dict(weights=rank[j]["weights"],groups=by_tier)
    save(root/"train-stages.json",dict(source=digest(source),scores=scores))
    print({n:{k:(v["outcomes"].get("SUCCESS",0),v["both"]) for k,v in r["groups"].items()} for n,r in scores.items()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("runs",nargs="+")
    for r in p.parse_args().runs:main(r)
