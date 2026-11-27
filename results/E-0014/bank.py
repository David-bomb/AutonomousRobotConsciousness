"""Сопоставление стартовых условий и уже сохранённых исходов test_id по имени задания."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sqlite3
from statistics import median
import zlib

from neural_rl.experiment import read, save
from neural_rl.stress_hundreds import SOURCE, SPLITS, bench

OUT = Path(__file__).resolve().parent


def classify(fuel, health, direct):
    return ("both" if max(fuel,health)<direct else "fuel" if fuel<direct
            else "health" if health<direct else "neither")


def main():
    stress = [json.loads(line) for split in SPLITS for line in bench().SETS[split].read_bytes().splitlines()]
    low, high = min(r["direct"] for r in stress), max(r["direct"] for r in stress)
    def bands(f,h,d):
        kind = classify(f,h,d)
        return [kind] + (["both_stress_length"] if kind == "both" and low<=d<=high else [])

    stocks = {}
    out = {"stress_direct_range":[low,high]}
    for split in ("train","test_id","stress"):
        bucket = defaultdict(list)
        order = Counter()
        if split=="stress":
            items = [(r["name"],r["fuel"],r["health"],r["direct"]) for r in stress]
        else:
            with sqlite3.connect((SOURCE/"corpus.sqlite").as_uri()+"?mode=ro",uri=True) as db:
                items = []
                for blob, in db.execute("SELECT data FROM geoms WHERE split=?", (split,)):
                    for r in json.loads(zlib.decompress(blob)):
                        name, levels = r["source"].splitlines()[:2]
                        fuel, health = map(int,levels.split())
                        items.append((name,fuel,health,r["direct"]))
        for name,f,h,d in items:
            labels = bands(f,h,d)
            for label in labels:
                bucket[label].append(d)
            if labels[0]=="both":
                order["fuel_lower" if f<h else "health_lower" if h<f else "equal"]+=1
            if split=="test_id":
                stocks[name]=labels
        out[split] = dict(count=len(items), direct_median=median(d for _,_,_,d in items),
                          groups={k:dict(n=len(v),median_direct=median(v)) for k,v in bucket.items()},
                          both_by_lower_initial_stock=dict(order))
    out["stress_optimum_median"]=median(r["optimum"] for r in stress)
    out["stress_first_witness_margin_median"]=median(r["first_witness_margin"] for r in stress)
    out["test_wins"]={}
    for family,meta in read(SOURCE/"frozen.json").items():
        seen=set()
        wins=Counter()
        for p in (SOURCE/"evaluation/test_id"/meta["id"]).glob("*.json.gz"):
            with gzip.open(p,"rt",encoding="utf-8") as stream:
                batch=json.load(stream)
            assert batch["weights_sha256"]==meta["sha256"]
            for r in batch["rows"]:
                assert r["map"] not in seen
                seen.add(r["map"])
                if r["outcome"]=="SUCCESS":
                    wins.update(stocks[r["map"]])
        assert seen==set(stocks) and len(seen)==24000
        out["test_wins"][family]={k:wins[k] for k in out["test_id"]["groups"]}
    save(OUT/"bank.json",out)
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
