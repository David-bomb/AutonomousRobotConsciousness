"""Полные физические поездки замороженных сетей на предложенной независимой сотне train."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
from . import cautious_eval
from .cautious_bank import bank as first_bank
from .common import HERE,PARENT,read,rows,write,digest
from .path_bank import bank as second_bank
from .release import differences


def main():
    folder=PARENT.parent/"verification-rl80";src=folder/"early-teacher100"
    receipt=read(src/"receipt.json");path=folder/"phase100-train/train.jsonl";maps=rows(path)
    assert digest(path)==receipt["bank_sha256"] and len(maps)==100
    old=set()
    for bank in (first_bank,second_bank):
        for split in ("fit","select","hold"):old.update(r["geometry"] for r in bank(split))
    assert not old.intersection(r["geometry"] for r in maps)
    out=HERE/"teacher100";out.mkdir(exist_ok=True)
    by={}
    for branch,external in (("wave1","fixed"),("teacher","hybrid")):
        episodes=[]
        for r in maps:
            a=read(src/"episodes"/(r["name"]+".json"))
            assert a["geometry"]==r["geometry"]
            e=a[external]
            episodes.append(dict(map=r["name"],tier=r["tier"],geometry=r["geometry"],outcome=e["outcome"],
                                 handover=e["first_open_windows"],both_at=e["first_both_services"],steps=e["steps"],
                                 actions=e["actions"],edits=None))
        by[branch]=episodes
    for branch in ("conservative","geodesic"):
        f=HERE/branch/"frozen.pt";sel=read(f.parent/"selection.json");assert digest(f)==sel["sha256"]
        with ProcessPoolExecutor(2,initializer=cautious_eval.init,initargs=(f,False)) as pool:
            records=[]
            for _,trips,_ in pool.map(cautious_eval.play,[(r,[sel["threshold"]],"dev") for r in maps]):
                records.append(trips[0])
        by[branch]=records
        write(out/(branch+".json"),dict(sha256=digest(f),bank=digest(path),episodes=records,summary=cautious_eval.summarize(records)))
        print("teacher100",branch,cautious_eval.summarize(records),flush=True)
    pairs={k:dict(versus_wave1=differences(dict(episodes=by["wave1"]),dict(episodes=v)),
                  versus_teacher=differences(dict(episodes=by["teacher"]),dict(episodes=v)))
           for k,v in by.items() if k in ("conservative","geodesic")}
    summaries={k:cautious_eval.summarize(v) for k,v in by.items()}
    assert summaries["wave1"]["wins"]==70 and summaries["teacher"]["wins"]==73,summaries
    write(out/"summary.json",dict(plan=digest(HERE/"external-train-plan.md"),bank=digest(path),
          demonstrations=digest(src/"demonstrations.jsonl"),receipt=digest(src/"receipt.json"),
          summaries=summaries,pairs=pairs,training_overlap=0))


if __name__=="__main__":main()
