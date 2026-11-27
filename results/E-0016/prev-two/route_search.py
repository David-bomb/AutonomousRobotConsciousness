"""Обучение линейного выхода за физический B из сохранённых состояний и со старта."""
import os
for k in ("MKL_NUM_THREADS","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[k]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import torch
from maze.environment import Environment
from . import terminal
from .adapter import MOVES
from .common import HERE,SEED,digest,group,level,read,write
from .fit import loadnet,freeze
from .route_adapter import RouteNet


def route_net(path):
    old=loadnet(path)
    from ..rich_policy import RichPolicy
    from ..critical_policy import CriticalPolicy
    net=RouteNet("rich" if isinstance(old.base.base,RichPolicy) else
                 "critical" if isinstance(old.base.base,CriticalPolicy) else "plain")
    state=old.state_dict()
    mix=state.pop("mix")
    missing,extra=net.load_state_dict(state,strict=False)
    assert set(missing)=={"mix","route_parent_mix"} and not extra
    with torch.no_grad():
        net.route_parent_mix.copy_(mix)
        net.mix[:9].copy_(mix)
    return net


def setup(path):
    torch.set_num_threads(1)
    terminal.NET=route_net(path).eval().requires_grad_(False)


def score(job):
    row,trace,options=job
    env=Environment(level(row));cache={};records=[]
    for weights in options:
        env.reset()
        previous=(None,None)
        if trace:
            for c in trace["prefix"]:
                previous=env.view.fuel_distance,env.view.repair_distance
                env.step(MOVES[int(c)])
            sn=trace["snapshot"];v=env.view
            assert list(previous)==sn["previous"] and tuple(sn["memory"])==v.memory
            assert (v.fuel,v.health,*v.position)==(sn["fuel"],sn["health"],*sn["position"])
        start_steps=env.steps
        n_unknown=sum(s.count("?") for s in env.view.memory)
        outcome,_=terminal.rollout(env,weights,cache,previous=previous)
        records.append(dict(**outcome,continuation_steps=env.steps-start_steps,
                            discovered=n_unknown-sum(s.count("?") for s in env.view.memory)))
    return row["name"],records


def run_pool(workers,cases,options):
    result={}
    for name,records in workers.map(score,[(r,t,options) for r,t in cases]):
        result[name]=records
        if len(result)%12==0: print("search exit",len(result),"/",len(cases),flush=True)
    totals=[sum(v[i]["win"] for v in result.values()) for i in range(len(options))]
    return result,totals


def train(parent):
    torch.set_num_threads(1)
    path=HERE/"weights"/(parent+".pt")
    net=route_net(path)
    rng=np.random.default_rng(SEED+202)
    index=group("extra")
    take=read(HERE/"takeovers.json")
    cases=[]
    for row in index[:100]:
        tr=read(HERE/"takeovers"/(row["name"]+".json"))
        if tr["reached"]: cases.append((row,tr))
    rng.shuffle(cases)
    full=[(r,None) for r in index[100:]]
    rng.shuffle(full)
    assert len(cases)>=48 and len(full)>=72
    initial=net.mix.detach().numpy().copy()
    options=[initial]
    # Пробуем раздельные масштабы, чтобы совместное изменение всех коэффициентов
    # не скрывало полезную поправку только к пути до B или только к разведке.
    for coefficients in ([-.9,0,18,0,0,0,0,0,0,0,0],
                         [-.9,-.65,18,2,4,.12,0,0,.1,1,0],
                         [-.7,0,10,.6,1,0,0,0,0,2,0],
                         [0,-.8,0,3,8,.4,0,0,.2,3,0],
                         [0,0,0,0,0,0,.8,1,0,3,4]):
        a=initial.copy()
        a[9:]=coefficients
        options.append(a)
    for i in range(14):
        a=initial.copy()
        a[9]=rng.uniform(-.98,-.35)
        a[10]=rng.uniform(-.95,.05)
        a[11]=rng.uniform(4,27)
        a[12]=rng.uniform(.6,7)
        a[13]=rng.uniform(1,12)
        a[14]=rng.uniform(-.15,1.2)
        a[15]=rng.uniform(0,4)
        a[16]=rng.uniform(0,3)
        a[17]=rng.uniform(-.15,1.4)
        a[18:20]=rng.uniform(0,6,size=2)
        options.append(a)
    with ProcessPoolExecutor(2,initializer=setup,initargs=(path,)) as workers:
        one,first=run_pool(workers,cases[:20]+full[:8],options)
        top=sorted(range(len(options)),key=lambda j:-first[j])[:5]
        shortlist=sorted(set([0,*top]))
        two,second=run_pool(workers,cases[20:40]+full[8:40],[options[i] for i in shortlist])
        winner=shortlist[max(range(len(shortlist)),key=lambda j:(second[j],first[shortlist[j]],-shortlist[j]))]
        # Независимый от выбора train-срез, диагностирует продолжения из тех же позиций.
        held,held_wins=run_pool(workers,cases[40:],[initial,options[winner]])
    with torch.no_grad():net.mix.copy_(torch.from_numpy(options[winner]))
    rows={row["name"]:tr for row,tr in cases[40:]}
    names=list(held)
    summary=dict(maps=len(names),physically_solvable=sum(rows[n]["shortest_finish"] is not None for n in names),
        super600_wins=sum(rows[n]["student"]["outcome"]=="SUCCESS" for n in names),
        teacher_wins=sum(rows[n]["teacher"]["outcome"]=="SUCCESS" for n in names),
        parent_wins=held_wins[0],trained_wins=held_wins[1])
    report=dict(parent=parent,takeover_sha256=digest(HERE/"takeovers.json"),options=[a.tolist() for a in options],
                first=one,first_wins=first,shortlist=shortlist,second=two,second_wins=second,selected=winner,
                held_continuations=held,held_summary=summary,plan_sha256=digest(HERE/"route-plan.md"))
    write(HERE/"terminal/route-search.json",report)
    freeze(net,"route-rl",dict(parent=parent,method="terminal SUCCESS from real post-service prefixes and full starts",
                              train_file="terminal/route-search.json",chosen=winner,held_summary=summary))
    print("route terminal selected",winner,summary,flush=True)


if __name__=="__main__":
    if len(sys.argv)>1: tag=sys.argv[1]
    else:
        from .compare import rank
        tag=max(["wave1-pg","wave1"],key=rank)
    train(tag)
