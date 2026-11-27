"""Независимое физическое воспроизведение трёх dev-серий и проверка учебных вмешательств."""
from copy import copy
from collections import Counter
import torch
from maze.environment import Environment,Status
from scripts.audit_automaton import Reachability
from .adapter import MOVES
from .common import DEV,level,read,rows,write,digest
from .early_bank import FOLDER
from .early_policy import complete,load


def check_corrections():
    from .fit import loadnet
    from .common import HERE
    from .adapter import tensor_batch
    net=loadnet(HERE/"weights/early1800.pt").requires_grad_(False).eval()
    report=read(FOLDER/"fit3-samples.json");n=choices=0;using=Counter()
    avoided=Counter()
    for row in read(FOLDER/"fit3.json"):
        t=report["records"][row["name"]]
        if not t["edits"]:continue
        env=Environment(level(row));exact=Reachability(env.level)
        edits={e["step"]:e for e in t["edits"]}
        previous=(None,None)
        for step,c in enumerate(t["actions"]):
            v=env.view
            if step in edits:
                e=edits[step];assert e["chosen"]==int(c)
                with torch.inference_mode():
                    selected=int(net(*tensor_batch([net.features(v,previous)])).argmax())
                category="first" if step==t["edits"][0]["step"] else "later"
                avoided[category+"_n"]+=1
                avoided[category+"_viable"]+=int(selected in e["viable"])
                for a,want in ((e["own"],False),(e["chosen"],True)):
                    child=copy(env);child._memory=copy(env._memory)
                    child.step(MOVES[a]);assert child.moved
                    assert exact.possible(child.position,child.fuel,child.health,child.limit-child.steps)==want
                    choices+=1
                n+=1;using[e["source"]]+=1
            previous=v.fuel_distance,v.repair_distance
            env.step(MOVES[int(c)])
        assert env.status.name==t["outcome"] and complete(env.view.memory)==t["windows"]
    assert n==report["total"]
    return dict(interventions=n,alternatives_independently_checked=choices,source=dict(using),
                early1800_training_revisit=dict(avoided))


def check_dev(parent,trained,ablated,tag):
    maps=rows(DEV)
    result={}
    for key,source in (("parent",parent),("honest",trained),("ablated",ablated)):
        assert source["bank"]==digest(DEV)
        for row,t in zip(maps,source["episodes"],strict=True):
            assert row["geometry"]==t["geometry"]
            env=Environment(level(row));gate=None;both=None
            for c in t["actions"]:
                if gate is None and complete(env.view.memory):gate=env.steps
                env.step(MOVES[int(c)]);assert env.moved
                if both is None and env.refuels and env.repairs:both=env.steps
            assert (gate,both,env.steps,env.status.name)==(t["handover"],t["both_at"],t["steps"],t["outcome"])
            assert env.status!=Status.RUNNING
        result[key]=sum(t["steps"] for t in source["episodes"])
    path=FOLDER/(tag+"-joint400.pt");assert digest(path)==trained["sha256"]==ablated["sha256"]
    index={0,99}
    for outcome in (True,False):
        diff=[i for i,(a,b) in enumerate(zip(trained["episodes"],ablated["episodes"]))
              if a["actions"]!=b["actions"] and (a["outcome"]=="SUCCESS")==outcome]
        if diff:index.add(min(diff,key=lambda i:trained["episodes"][i]["steps"]))
    for mode,table in ((False,trained),(True,ablated)):
        ctl=load(path.read_bytes(),mode)
        for i in sorted(index):
            env=Environment(level(maps[i]));ctl.reset();previous=(None,None)
            for c in table["episodes"][i]["actions"]:
                assert ctl.previous==previous
                v=env.view;move=ctl.choose(v);assert move==MOVES[int(c)]
                ctl.previous=previous
                assert ctl.choose(v,{(v,previous):move})==move
                previous=v.fuel_distance,v.repair_distance
                assert ctl.previous==previous
                env.step(move)
    return dict(replayed_steps=result,physical_episodes=300,uncached_each_mode=[maps[i]["name"] for i in sorted(index)])


def main():
    torch.set_num_threads(1)
    tag=read(FOLDER/"selection.json")["chosen"]
    parent=read(FOLDER/"dev-wave1.json");train=read(FOLDER/("dev-"+tag+".json"));ablate=read(FOLDER/("dev-"+tag+"-ablated.json"))
    write(FOLDER/"audit.json",dict(selected=tag,corrections=check_corrections(),dev=check_dev(parent,train,ablate,tag)))
    print("early audit verified",flush=True)


if __name__=="__main__":main()
