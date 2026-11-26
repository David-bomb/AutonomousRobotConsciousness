"""Внешняя проверка предположения о переходе через место расхождения двух дорог."""
import json

from bench import HERE, MOVES, cases
from compare import read
from maze.environment import Environment, Status
from maze.grid import cell, locations, wave
from shared_evidence import analyze


def changed_rows():
    path=HERE/"street-trunk--street-trunk-reserve.jsonl"
    return {r["map"]:r for line in path.read_bytes().splitlines()
            if (r:=json.loads(line))["verdict"] in ("приобретён","потерян")}


def winner_stops(env,actions,marks):
    visits=[]
    for char in actions:
        env.step(MOVES[int(char)])
        if env.position in marks and all(p["point"] != env.position for p in visits):
            visits.append(dict(kind=cell(env.level.grid,*env.position),point=env.position,step=env.steps))
    assert env.status==Status.SUCCESS
    return visits


def main():
    cases_with_change=changed_rows()
    parent,child=read("street-trunk"),read("street-trunk-reserve")
    results=[]
    for split,name,_,_,level in cases():
        if name not in cases_with_change:
            continue
        record=cases_with_change[name]
        env=Environment(level)
        offset=record["first"]["step"]-1
        winner=child[name] if child[name]["status"]=="SUCCESS" else parent[name]
        assert parent[name]["actions"][:offset]==child[name]["actions"][:offset]
        for c in winner["actions"][:offset]:
            env.step(MOVES[int(c)])
        view=env.view
        common=analyze(env)["shared"]
        via_fork=view.fuel_distance+view.repair_distance-2*max(common.values(),default=0)
        tree_fits=(view.fuel_distance+via_fork<=view.health or
                   view.repair_distance+via_fork<=view.fuel)
        kinds={p:k for k in "FRB" for p in locations(level.grid,k)}
        marks=frozenset(kinds)
        physical={p:d for p,d,_ in wave(level.grid,[env.position])}
        nearest={k:{p for p,kind in kinds.items() if kind==k and physical.get(p)==reading}
                 for k,reading in zip("FR",(view.fuel_distance,view.repair_distance))}
        separations=[d for p in nearest["F"] for q,d,_ in wave(level.grid,[p]) if q in nearest["R"]]
        assert separations and min(separations)<=via_fork
        first_legs={p:d for p,d,_ in wave(level.grid,[env.position],stop_at=marks) if p in marks and d}
        pairs=[]
        for i,k in enumerate("FR"):
            for a in sorted(nearest[k]):
                first=first_legs.get(a)
                if first!=physical[a] or first>=((view.health,view.fuel)[i]) or first>((view.fuel,view.health)[i]):
                    continue
                for b,d,_ in wave(level.grid,[a],stop_at=marks):
                    if b in nearest["RF"[i]] and first+d<=(view.health,view.fuel)[i] and d<100:
                        pairs.append(dict(first=a,second=b,between=d))
        visits=winner_stops(env,winner["actions"][offset:],marks)
        results.append(dict(map=name,split=split,outcome=record["verdict"],first=record["first"],
                            longest_shared=max(common.values(),default=0),via_fork=via_fork,
                            tree_fits=tree_fits,shortest_between=min(separations),reachable_pairs=pairs,
                            nearest={k:sorted(v) for k,v in nearest.items()},winner_visits=visits))
    assert len(results)==9
    (HERE/"fork-distances.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results),
                                           encoding="utf8")
    print("Проверено физических расхождений",len(results),flush=True)


if __name__=="__main__":
    main()
