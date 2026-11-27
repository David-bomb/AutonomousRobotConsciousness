"""Train and evaluate a population of early scoring heads with a shared final policy."""
from hashlib import sha256
import numpy as np
from renewable_train import EXP,Environment,Status,data,digest,moves,save
from handover_train import complete


def compare(out, tag, rows, options, net):
    dest=out/tag
    dest.mkdir(exist_ok=True)
    entries=[]
    wins=np.zeros(len(options),int)
    for i,row in enumerate(rows):
        p=dest/(row["name"]+".json")
        if p.exists():
            record=data.read(p)
            assert record["options"]==options.tolist() and record["protocol"]==digest(out/"protocol.json")
            trials=record["episodes"]
        else:
            env=Environment(data.level(row))
            early,late,values={},{},{}
            trials=[]
            for j,coeff in enumerate(options):
                env.reset()
                past=(None,None);visited=set();steps=bytearray()
                while env.status==Status.RUNNING and not net.route(env.view,net.cfg):
                    v=env.view;key=v,past
                    if key in visited:break
                    visited.add(key)
                    if key not in early:
                        q,fields=net.early_parts(v,past)
                        early[key]=(q[None,:]+options@fields.T).argmax(1)
                    a=int(early[key][j]);steps.append(a)
                    past=(v.fuel_distance,v.repair_distance)
                    env.step(moves[a])
                gate=env.status==Status.RUNNING and net.route(env.view,net.cfg)
                if gate:outcome,_,_,_=complete(env,net,late,values)
                else:outcome=env.status.name if env.status!=Status.RUNNING else "exact_cycle"
                trials.append(dict(map=row["name"],tier=row["tier"],model=j,outcome=outcome,gate=gate,
                                   early_steps=len(steps),prefix_sha256=sha256(steps).hexdigest()))
            save(p,dict(options=options.tolist(),protocol=digest(out/"protocol.json"),episodes=trials))
        entries.extend(trials)
        wins += np.array([e["outcome"]=="SUCCESS" for e in trials])
        if (i+1)%12==0:print(tag,i+1,wins.tolist(),flush=True)
    save(out/(tag+".json"),dict(wins=wins.tolist(),options=options.tolist(),episodes=entries))
    return wins


def physical(out,tag,net):
    folder=out/tag;folder.mkdir(exist_ok=True)
    rows=data.rows(EXP/"data/dev.jsonl")
    key=digest(out/"model.pt")
    episodes=[]
    for i,row in enumerate(rows):
        p=folder/(row["name"]+".json")
        if p.exists():
            rec=data.read(p)
            assert rec["model_sha256"]==key
        else:
            env=Environment(data.level(row));cache={};commands=bytearray()
            while env.status==Status.RUNNING:
                m=net.choose(env.view,cache)
                commands.append(moves.index(m));env.step(m)
            rec=dict(map=row["name"],tier=row["tier"],geometry=row["geometry"],outcome=env.status.name,
                     steps=env.steps,refuels=env.refuels,repairs=env.repairs,model_sha256=key,
                     actions_sha256=sha256(commands).hexdigest())
            save(p,rec)
        episodes.append(rec)
        if (i+1)%25==0:print(tag,i+1,sum(e["outcome"]=="SUCCESS" for e in episodes),flush=True)
    save(out/(tag+".json"),dict(model_sha256=key,dev_sha256=digest(EXP/"data/dev.jsonl"),episodes=episodes,
                               wins=[sum(e["outcome"]=="SUCCESS" for e in episodes)]))
