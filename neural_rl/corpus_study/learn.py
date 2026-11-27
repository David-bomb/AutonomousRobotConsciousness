"""Изолированный запуск обучения в E-0012. Физика и наблюдение принадлежат maze."""
import argparse
from collections import Counter
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import time
import zlib

import numpy as np
import torch

from maze.environment import Environment
from maze.levels import Level
from neural_rl import perception
from neural_rl.experiment import SEEDS, digest, read, save
from neural_rl.networks import Network
from neural_rl.rollout import evaluate
from neural_rl.training import train_q, train_ppo
from .corpus import DB
from .session import OUT, check_time, log

FAMILIES = ("double","dueling","rainbow","ppo")
CAPACITIES = (300,3000,30000,60000,3000,3000,3000,3000,3000,3000)
SCREEN = 60


@dataclass(frozen=True,slots=True)
class Entry:
    name: str
    stage: int
    index: int
    scenario: int


class Pool:
    """Сеть получает только Observation. В память загружаются текущие уровни."""
    def __init__(self, split, count, *, offset=0):
        self.db = sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True)
        self.split = split
        rows = self.db.execute("SELECT idx,size FROM geoms WHERE split=? AND idx>=? AND idx<? ORDER BY idx",
                               (split,offset,offset+count)).fetchall()
        if len(rows)!=count:
            raise ValueError(f"Ещё не готовы {split} [{offset},{offset+count})")
        self.cases = [Entry(f"{split}-{i:05}-{j}",1 if size<=15 else 2 if size<=21 else 3,i,j)
                      for i,size in rows for j in range(4)]
        self.template = lru_cache(maxsize=128)(self._template)
        self.records = lru_cache(maxsize=64)(self._records)

    def _records(self,index):
        data, = self.db.execute("SELECT data FROM geoms WHERE split=? AND idx=?",(self.split,index)).fetchone()
        return json.loads(zlib.decompress(data))

    def _template(self,case):
        record = self.records(case.index)[case.scenario]
        source = record["source"]
        if record["proof"]["status"]!="SUCCESS" or hashlib.sha256(source.encode()).hexdigest()!=record["sha256"]:
            raise ValueError("Неподтверждённый сценарий")
        level = Level.parse(source)
        if level.title!=case.name:
            raise ValueError("Неверное имя сценария")
        return Environment(level)

    def start(self,case):
        env=copy(self.template(case))
        env.reset()
        return env


class DirectNetwork(Network):
    def forward(self,data):
        check_time()
        return super().forward(data)


def config_for(family,variant,seed_no,ticks):
    conf = dict(id=f"{family}-v{variant:02}-s{seed_no}",family=family,variant=variant,
                seed=SEEDS[seed_no-1],geometry_count=CAPACITIES[variant-1],architecture=[8,64,1],
                lr=.0008,gamma=.97,lanes=24,ticks=ticks,horizon=1)
    if family=="ppo":
        conf.update(rollout=16,epochs=2,batch=96,clip=.2,entropy=.02,**{"lambda":.95})
    else:
        conf.update(batch=64,replay=16384,sync=64)
    if variant==5:
        conf["architecture"] = [8,64,2]
    elif variant==6:
        conf["architecture"] = [16,64,1]
    elif variant==7:
        conf["lr"] = .0003
    elif variant==8:
        conf["gamma"] = .99
    elif variant==9:
        conf["clip" if family=="ppo" else "sync"] = .12 if family=="ppo" else 32
    elif variant==10:
        conf["entropy" if family=="ppo" else "batch"] = .04 if family=="ppo" else 96
    return conf


class Chunk:
    def __init__(self,pool,cases):
        self.cases,self.start = cases,pool.start


def evaluate_rows(net,pool,chunk=32):
    rows=[]
    for start in range(0,len(pool.cases),chunk):
        check_time()
        rows.extend(evaluate(net,Chunk(pool,pool.cases[start:start+chunk]),1))
    return rows


def metrics(rows):
    counts=Counter(r["outcome"] for r in rows)
    rates={}
    for stage in (1,2,3):
        part=[r for r in rows if r["stage"]==stage]
        if part:
            rates[str(stage)] = sum(r["outcome"]=="SUCCESS" for r in part)/len(part)
    lengths=[r["steps"] for r in rows if r["outcome"]=="SUCCESS"]
    return dict(scenarios=len(rows),wins=counts["SUCCESS"],outcomes=dict(counts),by_stage=rates,
                worst=min(rates.values()),mean_success_steps=statistics.mean(lengths) if lengths else None)


def setup(seed):
    # Только в собственном процессе E-0012. Формат и код E-0011 воспроизводимы без изменений.
    perception.SIDE = 81
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")


def run(config,smoke=False):
    setup(config["seed"])
    run_id=config["id"]
    prefix = "smoke" if smoke else "runs"
    folder=OUT/prefix/run_id
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"run.json").is_file():
        old=read(folder/"run.json")
        if old["config"]!=config or digest(folder/"weights.pt")!=old["weights_sha256"]:
            raise ValueError("Не совпали сохранённая конфигурация или веса")
        return
    log("trial_started",id=run_id,technical=smoke,pid=os.getpid(),config=config)
    start=time.perf_counter()
    rng=random.Random(config["seed"])
    train,val=Pool("train",config["geometry_count"]),Pool("validation_id",SCREEN)
    if smoke:
        val.cases=[val.cases[0], val.cases[-1]]
        # Обе карты действительно прошли приёмку. Одна из них имеет сторону 41.
        for case in val.cases:
            env=val.start(case)
            encoded=perception.encode(env.view,False)
            assert encoded.image.shape==(81,81) and len(encoded.numbers)==29 and len(encoded.legal)==8
            assert encoded.image[env.position[1],env.position[0]] & 64
    net=DirectNetwork(config).cuda()
    initial=evaluate_rows(net,val)
    net.train()
    initial_params=[p.detach().clone() for p in net.parameters()]
    optimizer=torch.optim.Adam(net.parameters(),lr=config["lr"])
    train_start=time.perf_counter()
    if net.family=="ppo":
        history=train_ppo(net,optimizer,config,train,rng)
    else:
        history=train_q(net,optimizer,config,train,rng,np.random.default_rng(config["seed"]))
    torch.cuda.synchronize()
    train_seconds=time.perf_counter()-train_start
    delta=max((p.detach()-p0).abs().max().item() for p,p0 in zip(net.parameters(),initial_params))
    if not (delta>0 and math.isfinite(delta) and all(math.isfinite(s["mean_loss"]) and s["updates"]>0 for s in history)):
        raise RuntimeError("Не подтверждено конечное обновление весов")
    train_steps=sum(s["steps"] for s in history)
    if train_steps!=3*config["lanes"]*config["ticks"]:
        raise RuntimeError("Неполное обучение")
    rows=evaluate_rows(net,val)
    checkpoint=folder/"weights.pending"
    torch.save(net.state_dict(),checkpoint)
    checkpoint.replace(folder/"weights.pt")
    scenarios={name for s in history for name in s["maps"]}
    result=dict(config=config,training=history,train_steps=train_steps,updates=sum(s["updates"] for s in history),
                train_scenarios=len(scenarios),train_geometries=len({s.rsplit("-",1)[0] for s in scenarios}),
                train_seconds=round(train_seconds,3),seconds=round(time.perf_counter()-start,3),
                peak_gpu_mib=round(torch.cuda.max_memory_allocated()/2**20,2),parameter_delta=delta,
                initial=metrics(initial),screen=metrics(rows),weights_sha256=digest(folder/"weights.pt"))
    save(folder/"validation.json",dict(before=initial,after=rows))
    save(folder/"run.json",result)
    log("trial_completed",technical=smoke,id=run_id,train_steps=train_steps,updates=result["updates"],
        initial=result["initial"]["wins"],wins=result["screen"]["wins"],val=len(rows),
        seconds=result["seconds"],peak_gpu_mib=result["peak_gpu_mib"],weight_sha256=result["weights_sha256"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("family",choices=FAMILIES)
    parser.add_argument("variant",type=int)
    parser.add_argument("seed_no",type=int)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.smoke:
        config=config_for(args.family,1,1,32)
        config.update(lanes=8,batch=16)
    else:
        registry=read(OUT/"registration.json")
        config=next(c for c in registry["runs"] if c["family"]==args.family and c["variant"]==args.variant and c["seed"]==SEEDS[args.seed_no-1])
    run(config,args.smoke)
