"""Measure exactly what two authorized previous real readings certify in first-loss states on train."""
from collections import Counter
from importlib import import_module
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]/"University/Diplom"
sys.path.insert(0,str(PROJECT))
from maze.environment import Environment
from maze.grid import destination
from maze.interface import Move

data=import_module("results.E-0016.data")
frozen=import_module("results.E-0016.adapter")
teacher=import_module("results.E-0016.teacher")


def main():
    exp=PROJECT/"results/E-0016"
    weights=exp/"runs/recovery800/rl.pt"
    net=frozen.Controller(weights,sha256(weights.read_bytes()).hexdigest())
    labels=data.read(weights.with_suffix(".one-change.json"))
    target={r["map"]:r for r in labels["episodes"] if "alternatives" in r}
    rows=[]
    counts=Counter()
    for item in data.rows(exp/"data/train.jsonl")[700:800]:
        truth=target.get(item["name"])
        if not truth:continue
        env=Environment(data.level(item))
        observations=None
        for _ in range(truth["step"]-1):
            v=env.view
            observations=v.fuel_distance,v.repair_distance
            env.step(net.choose(v))
        v=env.view
        i=int(v.health-v.repair_distance<v.fuel-v.fuel_distance)
        reading=(v.fuel_distance,v.repair_distance)[i]
        allowed={x["move"] for x in truth["alternatives"]}
        rec=dict(map=item["name"],step=truth["step"],before=observations,current=reading,service="RF"[1-i],
                 prior_available=bool(observations is not None and v.moved))
        if rec["prior_available"]:
            old=observations[i]
            dx,dy=v.last_move.value
            u=(v.position[0]-dx,v.position[1]-dy)
            back=Move((-dx,-dy))
            assert destination(v.memory,v.position,back)==u
            rec["previous"]=old
            rec["reverse"]=back.name
            rec["certified_decrease"]=old==reading-1
            rec["certified_increase"]=old==reading+1
            rec["reverse_viable"]=back.name in allowed
            oracle=teacher.Teacher(env.level)
            ds=oracle.dist[2*i:2*i+2]
            assert min(d[u] for d in ds)==old
            # If the previous reading was larger, current legal neighbours that also
            # touch the previous position cannot decrease it by triangle inequality.
            from_old={q for m in Move if (q:=destination(v.memory,u,m)) is not None}
            impossible=set()
            for m in Move:
                q=destination(v.memory,v.position,m)
                if q is None:continue
                if (old>=reading and q==u) or (old>reading and q in from_old):
                    assert min(d[q] for d in ds)>=reading
                    impossible.add(m.name)
            chosen=net.choose(v).name
            rec.update(excluded=sorted(impossible),would_exclude_error=chosen in impossible)
            counts["known"]+=1
            counts["certified_back"]+=rec["certified_decrease"]
            counts["certified_back_and_viable"]+=rec["certified_decrease"] and rec["reverse_viable"]
            counts["excluded_bad"]+=rec["would_exclude_error"]
        rows.append(rec)
    result=dict(scope="E-0016 original train[700:800], first irreversible states of recovery800; independent of historical evaluation",
                model_sha256=sha256(weights.read_bytes()).hexdigest(),total=len(rows),summary=dict(counts),episodes=rows)
    out=HERE/"previous-measurements.json"
    encoded=(json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode()
    if out.exists() and out.read_bytes()!=encoded:raise ValueError("Previous audit differs")
    out.write_bytes(encoded)
    print(len(rows),dict(counts))


if __name__=="__main__":
    main()
