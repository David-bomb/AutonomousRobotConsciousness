"""Покартовые парные сравнения и сверка охвата E-0015."""
from collections import Counter
from statistics import median

from neural_rl.experiment import digest, read, save
from neural_rl.stress_hundreds import ROOT, SPLITS
from .train import OUT, REG


def paired(left, right):
    assert left.keys()==right.keys()
    return dict(gained=sorted(k for k in left if right[k] and not left[k]),
                lost=sorted(k for k in left if left[k] and not right[k]))


def finished(rows):
    states={r["map"]:r["outcome"]=="SUCCESS" for r in rows}
    assert len(states)==len(rows)
    return states


def fork_counts(rows):
    for r in rows:
        assert r["correct"]==sum(a==b for a,b in zip(r["choices"],r["expected"]))
        assert r["changed"]==(r["choices"][0]!=r["choices"][1])
    return dict(both=sum(r["correct"]==2 for r in rows),changed=sum(r["changed"] for r in rows),
                correct=sum(r["correct"] for r in rows),total=len(rows))


def main():
    reg=read(REG)
    assert all(digest(ROOT/p)==h for p,h in reg["base_sources"].items())
    assert all(digest(OUT/p)==h for p,h in reg["experiment_files"].items())
    totals={arm:Counter() for arm in reg["arms"]}
    seeds={}
    evidence={}
    for cfg in reg["configs"]:
        entry={}
        full={}
        for arm,tot in totals.items():
            folder=OUT/"runs"/cfg["id"]/arm
            ev=read(folder/"evaluation.json")
            tr=read(folder/"training.json")
            assert tr["sha256"]==ev["sha256"]==digest(folder/"weights.pt")
            assert tr["config"]==cfg
            evidence[folder.relative_to(OUT).as_posix()]=dict(weights=tr["sha256"],
                  training=digest(folder/"training.json"),evaluation=digest(folder/"evaluation.json"))
            full[arm]=ev
            probes=fork_counts(ev["forks"])
            identical=fork_counts(ev["indistinguishable"])
            assert identical==dict(both=0,changed=0,correct=25,total=25)
            entry[arm]=dict(forks=probes,validation=sum(finished(ev["validation"]).values()),
                            fork_finishes=sum(finished(ev["fork_episodes"]).values()),
                            median_margin_shift=median(r["margin_shift"] for r in ev["forks"]),
                            indistinguishable=identical,params=tr["params"],
                            archive_max_abs_difference=tr["archive_max_abs_difference"])
            if ev["uniform_read"] is not None:
                entry[arm]["uniform_forks"]=fork_counts(ev["uniform_read"])
                assert [r["pair"] for r in ev["forks"]]==[r["pair"] for r in ev["uniform_read"]]
                entry[arm]["decisions_changed_by_uniform_read"]=sum(
                    actual!=flat for a,b in zip(ev["forks"],ev["uniform_read"])
                    for actual,flat in zip(a["choices"],b["choices"]))
            wins={}
            for split in SPLITS:
                rows=[r for r in ev["stress"] if r["split"]==split]
                assert len(rows)==len(finished(rows))==100
                wins[split]=sum(finished(rows).values())
            entry[arm]["stress_wins"]=wins
            entry[arm]["visited_both"]=sum(bool(r["refuels"] and r["repairs"]) for r in ev["stress"])
            entry[arm]["outcomes"]=dict(Counter(r["outcome"] for r in ev["stress"]))
            assert {r["map"] for r in ev["stress"] if r["outcome"]=="SUCCESS"}<=ev["checked_paths"].keys()
            tot.update(both=probes["both"],correct=probes["correct"],changed=probes["changed"],
                       fork_finishes=entry[arm]["fork_finishes"],validation=entry[arm]["validation"],
                       visited_both=entry[arm]["visited_both"],stress=sum(wins.values()),
                       steps=sum(s["steps"] for s in tr["history"]),updates=sum(s["updates"] for s in tr["history"]))
            tot.update(wins)
        entry["paired"]={}
        for task in ("stress","validation","fork_episodes"):
            entry["paired"][task]=paired(finished(full["cnn"][task]),finished(full["query"][task]))
        entry["paired"]["forks"]=paired(*({str(r["pair"]):r["correct"]==2 for r in full[arm]["forks"]}
                                           for arm in ("cnn","query")))
        seeds[cfg["id"]]=entry
    assert len(seeds)==5 and all(t["steps"]==5*73728 and t["updates"]==5*1536 for t in totals.values())
    save(OUT/"comparison.json",dict(totals=totals,seeds=seeds,evidence=evidence,
         evaluation_code={p:digest(OUT/p) for p in ("evaluate_models.py","summarize.py")}))
    for name,r in seeds.items():
        print(name,[(a,r[a]["forks"],r[a]["stress_wins"],r[a]["validation"]) for a in totals])
    print(totals)


if __name__=="__main__":
    main()
