"""Полные физические поездки онлайн-учителя на свежем train."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import time

from maze.environment import Environment, Status
from maze.frontier_reserve import ReserveFrontier
from maze.interface import Move
from .bank import HERE, digest, level, read, register, save, train_order

MOVES = tuple(Move)


def internal(teacher):
    f = teacher._fallback
    return (teacher._unseen,teacher._service_streak,teacher.activity,f.mode,f.returning,
            f.service_offset,frozenset(f.last_lengths.items()))


def drive(env, teacher):
    seen = {}
    actions = []
    activities = Counter()
    cycle = None
    cycle_at = None
    calls = 0
    compute = 0.
    while env.status==Status.RUNNING:
        if cycle is None:
            key = env.view, internal(teacher)
            if key in seen:
                cycle_at = seen[key]
                cycle = actions[cycle_at:]
                assert cycle
            else:
                seen[key] = len(actions)
        if cycle is None:
            start = time.perf_counter()
            move = teacher.choose(env.view)
            compute += time.perf_counter()-start
            calls += 1
            activities[teacher.activity] += 1
            action = MOVES.index(move)
        else:
            action = cycle[(len(actions)-cycle_at)%len(cycle)]
        env.step(MOVES[action])
        assert env.moved
        actions.append(action)
    return dict(actions="".join(map(str,actions)),cycle_start=cycle_at,cycle_len=len(cycle or ()),
                teacher_calls=calls,teacher_seconds=compute,activities=dict(activities))


def demonstration(row):
    path = HERE/"teacher-traces"/(row["name"]+".json")
    if path.exists():
        return read(path)
    start = time.perf_counter()
    env = Environment(level(row))
    result = drive(env,ReserveFrontier())
    result.update(map=row["name"],tier=row["tier"],geometry=row["geometry"],outcome=env.status.name,
                  steps=env.steps,refuels=env.refuels,repairs=env.repairs,seconds=time.perf_counter()-start,
                  prefix=0)
    save(path,result)
    return result


def metrics(traces):
    elapsed = sum(r["seconds"] for r in traces)
    calls = sum(r["teacher_calls"] for r in traces)
    compute = sum(r["teacher_seconds"] for r in traces)
    return dict(n=len(traces),wins=sum(r["outcome"]=="SUCCESS" for r in traces),
                outcomes=dict(Counter(r["outcome"] for r in traces)),
                both=sum(min(r["refuels"],r["repairs"])>0 for r in traces),
                steps=sum(r["steps"] for r in traces),teacher_calls=calls,teacher_seconds=compute,
                calls_per_teacher_second=calls/compute,seconds_per_map=elapsed/len(traces))


def main():
    register()
    batch = train_order()
    start = time.perf_counter()
    with ProcessPoolExecutor(2) as workers:
        traces = list(workers.map(demonstration,batch[:16]))
        probe = metrics(traces)
        speed = probe["seconds_per_map"]
        n = 256 if speed<=6 else 128 if speed<=15 else 64 if speed<=30 else 32
        save(HERE/"teacher-speed.json",dict(probe=probe,n=n,names=[r["map"] for r in traces]))
        print("teacher probe",probe,"selected",n,flush=True)
        for trace in workers.map(demonstration,batch[16:n]):
            traces.append(trace)
            if len(traces)%32==0:
                print("teacher",len(traces),"/",n,"wins",sum(r["outcome"]=="SUCCESS" for r in traces),flush=True)
    save(HERE/"teacher.json",dict(summary=metrics(traces),wall_seconds=time.perf_counter()-start,
                                 files={r["map"]:digest(HERE/"teacher-traces"/(r["map"]+".json")) for r in traces}))


if __name__=="__main__":
    main()
