"""Проверка четырёх физических траекторий, двух границ передачи и оптимальности свидетелей."""
from collections import Counter
from itertools import combinations
import json
from statistics import median
import sys

from driver import (BANK_SHA, BRANCHES, HERE, MOVES, Environment, Status, as_level, boundary, full_state,
                    play, read, readings, registration, save, sha)
from scripts.audit_automaton import Reachability

LOCK="0688cd78e7163a74f232cde1b7bc44a230ee687b5308bfb1b78d79e587e860eb"
PREFIX={"wave":"own","teacher_wave":"teacher","teacher_frontier":"teacher","wave_exact":"own"}


def normalized(value):
    return json.loads(json.dumps(value))


def numbers(actions):
    assert set(actions)<=set("01234567")
    return bytes(map(int,actions))


def check_witness(env,witness,oracle):
    remaining=env.limit-env.steps
    can=lambda n:oracle.possible(env.position,env.fuel,env.health,n)
    if witness is None:
        assert not can(remaining)
    else:
        assert 0<len(witness)<=remaining and can(len(witness)) and not can(len(witness)-1)
        play(env,numbers(witness))
        assert env.status==Status.SUCCESS


def check_boundary(level,prefix,snap,oracle):
    env=Environment(level)
    previous=(None,None)
    for a in numbers(prefix):
        assert env.status==Status.RUNNING and not (env.refuels and env.repairs)
        previous=readings(env.view)
        env.step(MOVES[a])
    expected={k:v for k,v in snap.items() if k!="witness_suffix"}
    assert normalized(boundary(env,previous))==expected
    if snap["kind"]=="handoff":
        check_witness(env,snap["witness_suffix"],oracle)
    else:
        assert snap["witness_suffix"] is None and env.status!=Status.RUNNING


def inspect(row,item):
    level=as_level(row)
    assert (item["id"],item["tier"],item["geometry"],item["family"],item["initial"],item["limit"])==(
        row["name"],row["tier"],row["geometry"],row["family"],[level.fuel,level.health],Environment(level).limit)
    oracle=Reachability(level)
    optimum=item["optimal_actions"]
    assert len(optimum)==row["optimum"]
    check_witness(Environment(level),optimum,oracle)
    assert set(item["branches"])==set(BRANCHES)
    scores=item["branches"]
    t,o=item["teacher"],item["own"]
    teacher_prefix=optimum[:t["steps"]]
    own_prefix=scores["wave"]["actions"][:o["steps"]]
    check_boundary(level,teacher_prefix,t,oracle)
    check_boundary(level,own_prefix,o,oracle)
    assert t["kind"]=="early_goal" or (t["kind"]=="handoff" and t["witness_suffix"] is not None
                                       and len(t["witness_suffix"])==len(optimum)-len(teacher_prefix))
    for name in BRANCHES:
        s=scores[name]
        snap=item[PREFIX[name]]
        prefix=teacher_prefix if PREFIX[name]=="teacher" else own_prefix
        assert s["actions"].startswith(prefix)
        if snap["kind"]!="handoff":
            assert len(prefix)==s["steps"]
        else:
            assert snap["witness_suffix"] is not None or s["outcome"]!="SUCCESS"
        if name=="wave_exact":
            expected=own_prefix+o["witness_suffix"] if o["witness_suffix"] is not None else scores["wave"]["actions"]
            assert s["actions"]==expected
        env=Environment(level)
        actions=numbers(s["actions"])
        play(env,actions)
        assert env.status!=Status.RUNNING and normalized(full_state(env,actions))==s
    return item


def summarize(records):
    result={"maps":len(records),"arms":{},"pairs":{}}
    for name in BRANCHES:
        outcomes=Counter()
        stages=Counter()
        reached=available=finished=direct=0
        for r in records:
            s=r["branches"][name]
            snap=r[PREFIX[name]]
            hand=snap["kind"]=="handoff"
            win=s["outcome"]=="SUCCESS"
            outcomes[s["outcome"]]+=1
            stages[snap["kind"]]+=1
            reached+=hand
            available+=hand and snap["witness_suffix"] is not None
            finished+=hand and win
            direct+=not hand and win
        result["arms"][name]=dict(successes=outcomes["SUCCESS"],outcomes=dict(outcomes),stages=dict(stages),
                                  reached=reached,available=available,successful_continuations=finished,
                                  successes_without_handoff=direct,steps=sum(r["branches"][name]["steps"] for r in records))
    for a,b in combinations(BRANCHES,2):
        a_only,b_only=[],[]
        both=neither=0
        terminals={}
        for r in records:
            ra,rb=(r["branches"][arm]["outcome"] for arm in (a,b))
            wa,wb=ra=="SUCCESS",rb=="SUCCESS"
            if ra!=rb:
                terminals.setdefault(ra+"__"+rb,[]).append(r["id"])
            if wa and wb: both+=1
            elif wa: a_only.append(r["id"])
            elif wb: b_only.append(r["id"])
            else: neither+=1
        result["pairs"][a+"__"+b]=dict(both=both,neither=neither,first_only=a_only,second_only=b_only,
                                       different_terminal_outcomes=terminals)
    paired=[r for r in records if all(r[k]["kind"]=="handoff" for k in ("teacher","own"))]
    if paired:
        stats={}
        for key in ("teacher","own"):
            snaps=[r[key] for r in paired]
            stats[key]=dict(median_observed=median(s["observed"] for s in snaps),
                            median_handoff_step=median(s["steps"] for s in snaps),
                            median_fuel=median(s["observation"]["fuel"] for s in snaps),
                            median_health=median(s["observation"]["health"] for s in snaps))
        def same(field):
            return sum(r["teacher"]["observation"][field]==r["own"]["observation"][field] for r in paired)
        result["comparable_snapshots"]=dict(maps=len(paired),same_position=same("position"),same_memory=same("memory"),
                teacher_more_known=sum(r["teacher"]["observed"]>r["own"]["observed"] for r in paired),
                own_more_known=sum(r["teacher"]["observed"]<r["own"]["observed"] for r in paired),stats=stats)
        viable=[r for r in paired if r["own"]["witness_suffix"] is not None and r["teacher"]["witness_suffix"] is not None]
        result["both_prefixes_viable"]=dict(maps=len(viable),successes={
            arm:sum(r["branches"][arm]["outcome"]=="SUCCESS" for r in viable) for arm in BRANCHES})
    return result


def main():
    assert sha(HERE/"lock.json")==LOCK and sha(HERE/"train.jsonl")==BANK_SHA
    assert read(HERE/"lock.json")["registration"]==normalized(registration())
    rows=[json.loads(s) for s in (HERE/"train.jsonl").read_bytes().splitlines()]
    directory=HERE/"episodes"
    assert {p.stem for p in directory.iterdir()}=={r["name"] for r in rows}
    records=[]
    hashes={}
    for row in rows:
        p=directory/(row["name"]+".json")
        records.append(inspect(row,read(p)))
        hashes[row["name"]]=sha(p)
    assert len(set(hashes))==len(set(r["geometry"] for r in rows))==100
    assert Counter(r["tier"] for r in rows)==dict.fromkeys(("H1","H2","H3","H4"),25)
    data=dict(lock_sha256=LOCK,bank_sha256=BANK_SHA,all=summarize(records),
              by_tier={k:summarize([r for r in records if r["tier"]==k]) for k in ("H1","H2","H3","H4")},
              record_sha256=hashes)
    path=HERE/"receipt.json"
    if path.exists():
        assert read(path)==data
    else:
        save(path,data)
    print(json.dumps(data["all"]["arms"],ensure_ascii=False))


if __name__=="__main__":
    main()
