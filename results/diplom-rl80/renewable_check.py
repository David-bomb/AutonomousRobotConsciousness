"""Train-only unit check of action-conditional reachability with completely revealed synthetic inputs."""
import json
import numpy as np
from renewable_train import BASE,HERE,OUT,Controller,Environment,data,digest,moves,save
from renewable_policy import targets
from maze.grid import destination
from scripts.audit_automaton import Reachability


def main():
    rows=data.rows(HERE/"return-search/bank.jsonl")
    policy=Controller(BASE.read_bytes())
    results=[]
    for row in rows:
        env=Environment(data.level(row))
        v=env.view
        memory=env.level.grid
        w=len(memory[0]);x,y=v.position
        sites=targets(memory,env.level.goal)
        observed=np.empty((8,7))
        policy.dll.renewable("".join(memory).encode("ascii"),w,len(memory),y*w+x,sites,v.fuel,v.health,observed)
        oracle=Reachability(env.level)
        for i,m in enumerate(moves):
            q=destination(memory,(x,y),m)
            if q is None:continue
            stocks=oracle.after(q,env.fuel,env.health)
            actual=oracle.possible(q,*stocks,env.limit-1)
            estimate=observed[i,0]
            assert actual==(estimate<1.e7),(row["name"],m,actual,estimate)
            if actual:
                # Check the precise shortest physical length, including this first move.
                assert oracle.possible(q,*stocks,int(estimate)-1)
                if q!=env.level.goal:assert not oracle.possible(q,*stocks,int(estimate)-2)
            results.append([row["name"],m.name,actual,int(estimate) if actual else None])
    doc=dict(train_maps=len(rows),legal_actions=len(results),viable=sum(r[2] for r in results),
             source_sha256=digest(HERE/"renewable_features.cpp"),cases=results)
    save(OUT/"exact-check.json",doc)
    print(json.dumps({k:v for k,v in doc.items() if k!="cases"}))


if __name__=="__main__":main()
