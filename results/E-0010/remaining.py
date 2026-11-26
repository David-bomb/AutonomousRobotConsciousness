"""Общие неудачи всех автоматов фиксированного снимка, с разбором одинаковых префиксов."""
import argparse
from collections import Counter
import json
from zipfile import ZipFile

from bench import HERE, MOVES, SETS, cases
from checkpoint import digest
from compare import PhysicalRoutes, first_loss, read, replay, transition


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot")
    args=p.parse_args()
    with ZipFile(HERE/"snapshots"/f"{args.snapshot}.zip") as archive:
        manifest=json.loads(archive.read("manifest.json"))
    histories={}
    for key,record in manifest["runs"].items():
        assert digest((HERE/f"{key}.jsonl").read_bytes())==record["sha256_lf"]
        histories[key]=read(key)
    wins=Counter()
    path=HERE/f"unreached-{args.snapshot}.jsonl"
    with path.open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            rows={k:d[name] for k,d in histories.items()}
            if any(r["status"]=="SUCCESS" for r in rows.values()):
                wins[split]+=1
                continue
            sequences={r["actions"]:r for r in rows.values()}
            common=sum(1 for _ in prefix_length(tuple(sequences)))
            sample=rows["streets"]
            routes=PhysicalRoutes(level)
            critical={}
            for seq,r in sequences.items():
                trace,env=replay(level,r)
                critical[seq]=first_loss(routes,trace,env)
            trace,env=replay(level,sample)
            possible=[m.name for m in MOVES if transition(routes,trace[0],m,env.limit-1)]
            actual=sorted({MOVES[int(seq[0])].name for seq in sequences})
            row=dict(map=name,split=split,tier=tier,family=family,common_prefix=common,
                     irreversible_in_common=critical[sample["actions"]]["step"]<=common,
                     start_moves=actual,start_viable=possible,distinct_trips=len(sequences),
                     any_both=any(c["refuels"] and c["repairs"] for c in critical.values()),
                     controllers={k:critical[r["actions"]] for k,r in rows.items()})
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(len(histories),"автоматов; хотя бы один финиш",[wins[s] for s in SETS])


def prefix_length(sequences):
    for moves in zip(*sequences):
        if len(set(moves))!=1:
            break
        yield moves[0]


if __name__=="__main__":
    main()
