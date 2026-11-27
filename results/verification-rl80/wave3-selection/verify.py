"""Воспроизвести записи двух отборочных сотен без вызова сети и выбрать кандидата."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from run import HERE,SEEDS,SIGNS,Environment,Status,MOVES,banks,level,read,save,sha,spec

LOCK="120b41ef469782fef4ee6ee45500a9585486ffe8088830cfcccc93cb5b5178a3"
TIERS=("H1","H2","H3","H4")


def covers(view,positions):
    return all(any(all(view.memory[y+dy][x+dx]!="?" for dx in (-1,0,1) for dy in (-1,0,1))
                   for x,y in group) for group in positions)


def verify_row(row,r,seed,m):
    assert (r["id"],r["tier"],r["geometry"],r["method"],r["seed"],r["weights_sha256"],r["initial_registers"])==(
        row["name"],row["tier"],row["geometry"],m,seed,SIGNS[m],[None,None])
    env=Environment(level(row))
    assert r["initial"]==[env.fuel,env.health] and r["limit"]==env.limit
    data=bytes(map(int,r["actions"]))
    assert 0<len(data)<=env.limit and sha256(data).hexdigest()==r["actions_sha256"]
    assert r["routing"]==({"windows":9,"threshold":.9} if m=="B" else {"cfg":[9,9,-101]})
    pos=[[(x,y) for y,line in enumerate(env.view.memory) for x,c in enumerate(line) if c==s] for s in "FR"]
    assert all(0<x<len(env.view.memory[0])-1 and 0<y<len(env.view.memory)-1 for group in pos for x,y in group)
    assert all(len(group)==2 for group in pos)
    first_windows=0 if covers(env.view,pos) else None
    first_late=first_both=None
    flags=[]
    for i in data:
        assert env.status==Status.RUNNING and i<8
        v=env.view
        late=first_windows is not None and (m=="B" or min(v.fuel-v.fuel_distance,v.health-v.repair_distance)>=-101)
        flags.append("1" if late else "0")
        if late and first_late is None:
            first_late=env.steps+1
        env.step(MOVES[i])
        if first_windows is None and covers(env.view,pos):first_windows=env.steps
        if first_both is None and env.refuels and env.repairs:first_both=env.steps
    assert env.status!=Status.RUNNING and "".join(flags)==r["phases"]
    assert (env.steps,env.refuels,env.repairs,env.status.name,first_windows,first_late,first_both,flags.count("1"))==tuple(
        r[k] for k in ("steps","refuels","repairs","outcome","first_windows","first_late_command","first_both","late_actions"))


def phases(rows):
    late=[r for r in rows if r["first_late_command"] is not None]
    wins=sum(r["outcome"]=="SUCCESS" for r in rows)
    late_wins=sum(r["outcome"]=="SUCCESS" for r in late)
    return dict(maps=len(rows),success=wins,windows=sum(r["first_windows"] is not None for r in rows),
                late_entered=len(late),both_services=sum(r["first_both"] is not None for r in rows),
                failed_before_late=len(rows)-len(late)-(wins-late_wins),failed_after_late=len(late)-late_wins,
                success_before_late=wins-late_wins,success_after_late=late_wins,
                late_steps=sum(r["late_actions"] for r in rows),steps=sum(r["steps"] for r in rows),
                refuels=sum(r["refuels"] for r in rows),repairs=sum(r["repairs"] for r in rows),
                outcomes=dict(Counter(r["outcome"] for r in rows)))


def pairs(by_method):
    data={}
    for a,b in combinations(SIGNS,2):
        left,right=by_method[a],by_method[b]
        row={}
        for endpoint in ("finish","late","services"):
            first,second=[],[]
            both=neither=0
            for x,y in zip(left,right,strict=True):
                assert x["id"]==y["id"]
                def hit(r):
                    if endpoint=="finish":return r["outcome"]=="SUCCESS"
                    return r["first_late_command" if endpoint=="late" else "first_both"] is not None
                ha,hb=hit(x),hit(y)
                if ha and hb:both+=1
                elif ha:first.append(x["id"])
                elif hb:second.append(y["id"])
                else:neither+=1
            row[endpoint]=dict(both=both,neither=neither,first_only=first,second_only=second)
        data[a+b]=row
    return data


def main():
    assert sha(HERE/"lock.json")==LOCK and read(HERE/"lock.json")["registration"]==spec()
    case_sets=banks()
    directory=HERE/"rows"
    assert {p.name for p in directory.iterdir()}=={f"{s}-{m}-{i:02d}.json" for s in SEEDS for m in SIGNS for i in range(100)}
    results={}
    hashes={}
    for s in SEEDS:
        by_method={}
        for m in SIGNS:
            batch=[]
            for i,case in enumerate(case_sets[s]):
                p=directory/f"{s}-{m}-{i:02d}.json"
                r=read(p)
                verify_row(case,r,s,m)
                hashes[p.name]=sha(p)
                batch.append(r)
            by_method[m]=batch
        results[str(s)]=dict(methods={m:dict(**phases(rs),by_tier={t:phases([r for r in rs if r["tier"]==t]) for t in TIERS})
                                      for m,rs in by_method.items()},pairs=pairs(by_method))
    ranks={}
    for m in SIGNS:
        parts=[results[str(s)]["methods"][m] for s in SEEDS]
        a,b=(t["success"] for t in parts)
        ranks[m]=[min(a,b),a+b,sum(t["by_tier"]["H4"]["success"] for t in parts)]
    ordered=sorted(SIGNS,key=lambda m:(*[-v for v in ranks[m]],m))
    decision=dict(lock_sha256=LOCK,winner=ordered[0],weights_sha256=SIGNS[ordered[0]],ranking=ordered,keys=ranks,
                  sets={s:{m:v["success"] for m,v in rs["methods"].items()} for s,rs in results.items()})
    report=dict(**decision,episodes=len(hashes),details=results,files_sha256=hashes)
    for filename,values in (("receipt.json",report),("selection.json",decision)):
        path=HERE/filename
        if path.exists():assert read(path)==values
        else:save(path,values)
    print(decision)


if __name__=="__main__":
    main()
