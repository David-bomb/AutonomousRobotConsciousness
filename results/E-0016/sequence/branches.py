"""На учебных развилках сравниваются полные физические продолжения каждого первого хода."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from pathlib import Path
import time
import numpy as np
import torch
from maze.environment import Environment, Status
from ..data import digest, level, read, rows, save
from ..policy import MOVES
from ..teacher import Teacher
from .bank import HERE
from .model import SequenceNet, encode


def fork(env):
    # Карта, статические длины и Observation неизменяемы. Для пишущейся памяти
    # клеток у каждого продолжения отдельный объект с тем же исходным кортежем строк.
    branch = copy(env)
    branch._memory = copy(env._memory)
    return branch


def init_worker(path):
    global network, prior
    torch.set_num_threads(1)
    state = torch.load(path,weights_only=True,map_location="cpu")
    network = SequenceNet().eval()
    network.load_state_dict(state)
    prior = network.wall_prior.numpy().tobytes()


@torch.inference_mode()
def outcomes(job):
    row,demo = job
    rng = np.random.default_rng(int(row["geometry"][:12],16))
    env = Environment(level(row))
    expert = Teacher(env.level)
    cache = {}
    def predict(views):
        fresh = [v for v in dict.fromkeys(views) if v not in cache]
        if fresh:
            x,mask = map(np.stack,zip(*(encode(v,prior) for v in fresh)))
            score = network(torch.from_numpy(x),torch.from_numpy(mask))
            prob = (score/.6).softmax(1).numpy().cumsum(1)
            for v,ok,s,p in zip(fresh,mask,score.numpy(),prob):
                cache[v] = (int(s.argmax()),p,np.flatnonzero(ok))
    candidates = {}
    counts = Counter()
    def offer(tag,commands):
        counts[tag] += 1
        if rng.integers(counts[tag]) == 0:
            candidates[tag] = (fork(env),commands)
    commands = ""
    seen = set()
    while env.status == Status.RUNNING and env.steps < 360 and env.view not in seen:
        v = env.view
        seen.add(v)
        predict([v])
        move,_,legal = cache[v]
        _,viable = expert.labels(env)
        if not viable.any():
            break
        if len(legal)>1:
            stage = "both" if env.refuels and env.repairs else "one" if env.refuels or env.repairs else "before"
            offer(stage,commands)
            if not viable[move]:
                offer("irreversible",commands)
            elif not viable[legal].all():
                offer("narrow",commands)
        commands += str(move)
        env.step(MOVES[move])
    if demo:
        env = Environment(env.level)
        for j,c in enumerate(demo):
            if env.refuels and env.repairs:
                # Только для выбора учебного состояния. Самостоятельные продолжения
                # начинаются после развилки без участия автора демонстрации.
                v = env.view
                if sum(v.window[y][x] not in "#?" for x,y in ((1,0),(0,1),(2,1),(1,2)))>1:
                    offer("demo_both",demo[:j])
            env.step(MOVES[int(c)])
    expert.remaining.cache_clear()
    examples = []
    active = []
    records = {}
    total_steps = 0
    for key,(start,cmd) in candidates.items():
        if start.view in records:
            examples[records[start.view]]["strata"].append(key)
            continue
        predict([start.view])
        _,_,legal = cache[start.view]
        records[start.view] = len(examples)
        trial = dict(prefix=cmd,strata=[key],legal=legal.tolist(),outcomes=[])
        examples.append(trial)
        for a in legal:
            for stochastic in (False,True):
                branch = fork(start)
                branch.step(MOVES[a])
                total_steps += 1
                info = dict(action=int(a),stochastic=stochastic)
                trial["outcomes"].append(info)
                active.append((branch,stochastic,info))
    while active:
        running = []
        predict([env.view for env,_,_ in active if env.status==Status.RUNNING])
        for env,stochastic,info in active:
            if env.status!=Status.RUNNING:
                info.update(outcome=env.status.name,final_step=env.steps)
                continue
            greedy,cdf,_ = cache[env.view]
            a = min(7,int(np.searchsorted(cdf,rng.random()*cdf[-1],side="right"))) if stochastic else greedy
            env.step(MOVES[a])
            total_steps += 1
            running.append((env,stochastic,info))
        active = running
    return dict(map=row["name"],geometry=row["geometry"],examples=examples,steps=total_steps)


def main(n):
    assert 0<n<=400 and n%4==0
    weights = HERE/"runs/mission600/rl.pt"
    name = f"branches-{n}"
    folder = HERE/"cache"/name
    folder.mkdir(parents=True,exist_ok=True)
    bank = rows(HERE/"train.jsonl")[:400]
    traces = {r["map"]:r["actions"] for r in read(HERE/"clean-400.json")["episodes"] if r["outcome"]=="SUCCESS"}
    # Каждый префикс из четвертей банка содержит поровну карт H1–H4.
    bank = [r for t in ("H1","H2","H3","H4") for r in [r for r in bank if r["tier"]==t][:n//4]]
    signature = dict(weights_sha256=digest(weights),source_sha256=digest(Path(__file__)),train_sha256=digest(HERE/"train.jsonl"),
                     demo_sha256=digest(HERE/"clean-400.json"),maps=n,rollin_cap=360,continuation="greedy and softmax T=.6 to terminal")
    if (folder/"config.json").exists():
        assert read(folder/"config.json")==signature
    else:
        save(folder/"config.json",signature)
    todo = [(r,traces.get(r["name"])) for r in bank if not (folder/(r["name"]+".json")).exists()]
    start = time.monotonic()
    with ProcessPoolExecutor(3,initializer=init_worker,initargs=(weights,)) as pool:
        for i,row in enumerate(pool.map(outcomes,todo,chunksize=1),1):
            save(folder/(row["map"]+".json"),row)
            if i%10==0:
                print("train maps",i,len(todo),round(time.monotonic()-start,1),flush=True)
    records = [read(folder/(r["name"]+".json")) for r in bank]
    states = [s for r in records for s in r["examples"]]
    acts = [a for s in states for a in s["outcomes"]]
    summary = dict(states=len(states),branches=len(acts),actual_steps=sum(r["steps"] for r in records),
                   outcomes=dict(Counter(a["outcome"] for a in acts)),
                   mixed=sum(0<sum(a["outcome"]=="SUCCESS" for a in s["outcomes"])<len(s["outcomes"]) for s in states))
    save(HERE/(name+".json"),dict(config=signature,summary=summary,records=records))
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("maps",type=int)
    main(parser.parse_args().maps)
