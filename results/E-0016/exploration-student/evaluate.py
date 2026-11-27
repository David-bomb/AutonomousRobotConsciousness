"""Оценщик фиксированных весов. Каждый выбранный ход исполняется штатной физикой."""
from collections import Counter
import time
from maze.environment import Environment,Status
from .adapter import Controller
from .bank import DEV,HERE,digest,level,read,rows,save
from .features import MOVES


def episode(controller,row):
    env=Environment(level(row))
    choices={}
    actions=[]
    first_f=first_r=first_both=None
    unknown_at_both=visits_at_both=None
    while env.status==Status.RUNNING:
        view=env.view
        if view not in choices:
            choices[view]=controller.choose(view)
        move=choices[view]
        env.step(move)
        assert env.moved
        actions.append(str(MOVES.index(move)))
        if first_f is None and env.refuels:
            first_f=env.steps
        if first_r is None and env.repairs:
            first_r=env.steps
        if first_both is None and first_f is not None and first_r is not None:
            first_both=env.steps
            unknown_at_both=sum(s.count("?") for s in env.view.memory)
            visits_at_both=env.refuels+env.repairs
    return dict(map=row["name"],tier=row["tier"],geometry=row["geometry"],outcome=env.status.name,
                steps=env.steps,actions="".join(actions),refuels=env.refuels,repairs=env.repairs,
                first_f=first_f,first_r=first_r,both_at=first_both,unique_observations=len(choices),
                post_both_steps=None if first_both is None else env.steps-first_both,
                post_both_discovered=None if first_both is None else unknown_at_both-sum(s.count("?") for s in env.view.memory),
                post_both_service_visits=None if first_both is None else env.refuels+env.repairs-visits_at_both)


def summary(trips):
    won=lambda r:r["outcome"]=="SUCCESS"
    wins=sum(map(won,trips))
    both=[r for r in trips if r["both_at"] is not None]
    stops=Counter(sum([r["refuels"]>0,r["repairs"]>0]) for r in trips if not won(r))
    return dict(wins=wins,tiers={t:sum(won(r) for r in trips if r["tier"]==t) for t in ("H1","H2","H3","H4")},
                outcomes=dict(Counter(r["outcome"] for r in trips)),both=len(both),
                wins_after_both=sum(map(won,both)),stopped_before_any=stops[0],
                stopped_after_one=stops[1],stopped_after_both=stops[2],
                post_both_steps=sum(r["post_both_steps"] for r in both),
                post_both_discovered=sum(r["post_both_discovered"] for r in both),
                post_both_service_visits=sum(r["post_both_service_visits"] for r in both),
                steps=sum(r["steps"] for r in trips))


def evaluate(tag,repeat=False):
    reg=read(HERE/"registration.json")
    freeze=read(HERE/"weights"/(tag+".json"))
    weights=HERE/"weights"/(tag+".pt")
    assert digest(DEV)==reg["dev_sha256"]
    for file,checksum in freeze["adapter"].items():
        assert digest(HERE/file)==checksum
    bank=rows(DEV)
    assert [r["name"] for r in bank]==reg["dev_names"] and len(bank)==100
    output=HERE/"dev"/(tag+".json")
    ctl=Controller(weights,freeze["sha256"])
    if output.exists() and not repeat:
        old=read(output)
        assert old["weights_sha256"]==digest(weights) and old["dev_sha256"]==digest(DEV)
        return old["summary"]
    start=time.perf_counter()
    trips=[episode(ctl,row) for row in bank]
    result=dict(weights_sha256=freeze["sha256"],dev_sha256=reg["dev_sha256"],summary=summary(trips),episodes=trips)
    if repeat:
        original=read(output)
        assert all(original[k]==v for k,v in result.items())
        save(HERE/"repeat.json",dict(tag=tag,weights_sha256=digest(weights),dev_file_sha256=digest(output),
                                    identical_episodes=100,identical_actions=sum(r["steps"] for r in trips)))
    else:
        save(output,dict(**result,seconds=time.perf_counter()-start))
    print("dev",tag,result["summary"],flush=True)
    return result["summary"]
