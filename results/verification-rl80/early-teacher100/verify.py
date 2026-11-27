"""Проверить все действия и ранние метки, сопоставить обе ветви по одному ID карты."""
from collections import Counter
from hashlib import sha256
import json

from run import BANK,HERE,ORDER,Environment,Status,add_label,level,pack,read,save,sha,spec
from maze.datasets import geometry

LOCK="fb2e1825d00c0383a5052d66288b337737e219c8525ec26cbac26459d3cc4ba0"


def covered(view):
    seen={s:False for s in "FR"}
    for y,row in enumerate(view.memory):
        for x,symbol in enumerate(row):
            if symbol in seen:
                count=sum(view.memory[j][i]!="?" for j in range(max(0,y-1),min(y+2,len(view.memory)))
                          for i in range(max(0,x-1),min(x+2,len(row))))
                seen[symbol]|=count==9
    return all(seen.values())


def replay(lvl,record,hybrid):
    env=Environment(lvl)
    previous=(None,None)
    labels=sha256()
    actions=bytes(map(int,record["actions"]))
    assert sha256(actions).hexdigest()==record["actions_sha256"] and all(a<8 for a in actions)
    first_gate=0 if covered(env.view) else None
    first_both=None
    late_calls=teacher_decisions=0
    for a in actions:
        assert env.status==Status.RUNNING
        v=env.view
        if first_gate is None:
            if hybrid:
                add_label(labels,pack(v),previous,a)
                teacher_decisions+=1
        else:
            late_calls+=hybrid or min(v.fuel-v.fuel_distance,v.health-v.repair_distance)>=-101
        previous=(v.fuel_distance,v.repair_distance)
        env.step(ORDER[a])
        if first_gate is None and covered(env.view):
            first_gate=env.steps
        if first_both is None and env.refuels and env.repairs:
            first_both=env.steps
    assert env.status!=Status.RUNNING
    assert (env.status.name,env.steps,env.refuels,env.repairs,first_gate,first_both,late_calls,teacher_decisions)==tuple(
        record[k] for k in ("outcome","steps","refuels","repairs","first_open_windows","first_both_services","late_calls","teacher_steps"))
    if hybrid:
        assert labels.hexdigest()==record["teacher_labels_sha256"]
    elif first_gate is not None:
        assert len(actions)-first_gate>=late_calls


def aggregate(records):
    result={"n":len(records)}
    keys=("fixed","hybrid")
    for name in keys:
        runs=[r[name] for r in records]
        wins=[r for r in runs if r["outcome"]=="SUCCESS"]
        result[name]=dict(success=len(wins),both_services=sum(r["first_both_services"] is not None for r in runs),
                          windows=sum(r["first_open_windows"] is not None for r in runs),
                          switched=sum(r["late_calls"]>0 for r in runs),
                          steps=sum(r["steps"] for r in runs),outcomes=dict(Counter(r["outcome"] for r in runs)))
    for endpoint in ("finish","both_services"):
        a_only,b_only=[],[]
        both=neither=0
        for r in records:
            a,b=(r[k]["outcome"]=="SUCCESS" if endpoint=="finish" else r[k]["first_both_services"] is not None for k in keys)
            if a and b:both+=1
            elif a:a_only.append(r["id"])
            elif b:b_only.append(r["id"])
            else:neither+=1
        result[endpoint+"_pairs"]=dict(both=both,neither=neither,teacher_only=b_only,fixed_only=a_only)
    return result


def main():
    assert sha(HERE/"lock.json")==LOCK and read(HERE/"lock.json")["registration"]==spec()
    bank=list(map(json.loads,BANK.read_bytes().splitlines()))
    files=HERE/"episodes"
    assert {p.stem for p in files.iterdir()}=={v["name"] for v in bank} and len(bank)==100
    records=[]
    hashes={}
    for row in bank:
        name=row["name"]
        path=files/(name+".json")
        pair=read(path)
        lvl=level(row)
        assert (pair["id"],pair["tier"],pair["family"],pair["geometry"],pair["initial"],pair["initial_slack"])==(
            name,row["tier"],row["family"],geometry(lvl.grid),[lvl.fuel,lvl.health],row["min_start_slack"])
        for key in ("fixed","hybrid"):
            replay(lvl,pair[key],key=="hybrid")
        records.append(pair)
        hashes[name]=sha(path)
    overall=aggregate(records)
    result=dict(lock_sha256=LOCK,bank_sha256=sha(BANK),overall=overall,
                by_tier={t:aggregate([r for r in records if r["tier"]==t]) for t in ("H1","H2","H3","H4")},
                tight=aggregate([r for r in records if r["initial_slack"]<=2]),raw_sha256=hashes)
    if overall["hybrid"]["success"]>overall["fixed"]["success"]:
        gains=set(overall["finish_pairs"]["teacher_only"])
        demos=[]
        for r in records:
            h=r["hybrid"]
            n=h["teacher_steps"]
            demos.append(dict(id=r["id"],tier=r["tier"],geometry=r["geometry"],initial_slack=r["initial_slack"],
                              teacher_only_win=r["id"] in gains,teacher_steps=n,early_actions=h["actions"][:n],
                              actions_sha256=sha256(bytes(map(int,h["actions"][:n]))).hexdigest(),
                              labels_sha256=h["teacher_labels_sha256"]))
        payload=b"".join((json.dumps(d,sort_keys=True,separators=(",",":"))+"\n").encode() for d in demos)
        path=HERE/"demonstrations.jsonl"
        if path.exists():assert path.read_bytes()==payload
        else:
            with path.open("xb") as f:f.write(payload)
        result["demonstrations"]=dict(path=path.name,maps=len(demos),labels=sum(d["teacher_steps"] for d in demos),
                                      gain_maps=len(gains),sha256=sha(path))
    path=HERE/"receipt.json"
    if path.exists():assert read(path)==result
    else:save(path,result)
    print(json.dumps({k:overall[k] for k in ("n","fixed","hybrid")},ensure_ascii=False))


if __name__=="__main__":
    main()
