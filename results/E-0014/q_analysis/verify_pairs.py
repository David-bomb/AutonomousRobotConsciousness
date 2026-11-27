"""Проверка пяти пар и построение таблицы по 1 200 общим валидационным поездкам."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIR = HERE/"reward_pair"
ARCHIVE = HERE.parents[2]/"results/E-0012"


def verify():
    lines = []
    totals = Counter()
    maps = None
    for seed in range(1,6):
        name = f"double-v03-s{seed}"
        d = json.loads((PAIR/f"{name}.json").read_text(encoding="utf-8"))
        original, altered = d["original"], d["without_coordinates"]
        for weights, expected in ((ARCHIVE/"runs"/name/"weights.pt",d["original_sha256"]),
                                   (PAIR/f"{name}-without-coordinate.pt",d["altered_sha256"])):
            assert hashlib.sha256(weights.read_bytes()).hexdigest()==expected
        keys = [r["map"] for r in original["rows"]]
        assert keys == [r["map"] for r in altered["rows"]] and len(set(keys))==1200
        if maps is None:
            maps = keys
        assert maps == keys
        if seed==2:
            assert d["repeated_control"]["max_abs_difference"]==0
            archive = {}
            for start in (1000,1100,1200):
                path = ARCHIVE/"evaluation/validation_id"/name/f"{start:05d}.json.gz"
                with gzip.open(path,"rt",encoding="utf-8") as f:
                    archive.update({r["map"]:{k:v for k,v in r.items() if k!="group"}
                                    for r in json.load(f)["rows"]})
            assert archive=={r["map"]:r for r in original["rows"]}
        values = []
        for condition, result in (("base",original),("altered",altered)):
            wins = sum(r["outcome"]=="SUCCESS" for r in result["rows"])
            assert wins==result["wins"]
            values.append(wins)
            totals[condition,"wins"]+=wins
        for group,n in (("opposed",890),("agree",4513)):
            for condition,result in (("base",original),("altered",altered)):
                stats = result["branches"][group]
                assert stats["n"]==n
                values.append(stats["correct"])
                totals[condition,group]+=stats["correct"]
        lines.append("| " + " | ".join(map(str,[seed,*values])) + " |")
    lines.append("| Сумма | " + " | ".join(str(totals[c,k]) for k in ("wins","opposed","agree")
                                         for c in ("base","altered")) + " |")
    return lines


if __name__=="__main__":
    print("\n".join(verify()))
