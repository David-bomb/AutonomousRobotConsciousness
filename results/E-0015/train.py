"""Регистрация и обучение пар E-0015. Запуск python -m results.E-0015.train."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from neural_rl.corpus_study.learn import Pool, setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import encode
from neural_rl.stress_hundreds import SOURCE, ROOT, SPLITS, bench
from neural_rl.training import train_q
from .model import QueryNetwork

OUT = Path(__file__).resolve().parent
REG = OUT / "registration.json"
ARMS = ("cnn","query")


def register():
    base = read(SOURCE/"registration.json")
    configs = [c for c in base["runs"] if c["family"]=="double" and c["variant"]==3]
    assert len(configs)==5 and len({c["seed"] for c in configs})==5
    sources = base["source_sha256"]
    assert all(digest(ROOT/p)==h for p,h in sources.items())
    sets = {s:sha256(bench().SETS[s].read_bytes().replace(b"\r\n",b"\n")).hexdigest() for s in SPLITS}
    assert sets==read(ROOT/"results/E-0013/manifest.json")["sets_sha256_lf"]
    probe_hash=digest(SOURCE/"diagnostics/control.jsonl.gz")
    assert probe_hash==read(SOURCE/"diagnostics/verified.json")["sha256"]["control"]
    params=dict(configs=configs,arms=ARMS,input_canvas=base["input_canvas"],base_sources=sources,
                stress_maps=sets,control_sha256=probe_hash,train_db_sha256=digest(SOURCE/"corpus.sqlite"),
                experiment_files={p:digest(OUT/p) for p in ("model.py","train.py","protocol.md")},
                torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    if REG.exists():
        prior=read(REG)
        assert prior["configs"]==configs and prior["arms"]==list(ARMS)
        assert all(prior[k]==v for k,v in params.items() if k!="arms")
    else:
        save(REG,dict(**params,created_utc=datetime.now(timezone.utc).isoformat()))


def network(arm,config,side):
    return (Network(config) if arm=="cnn" else QueryNetwork(config,side)).cuda()


def train_one(job):
    arm,config=job
    reg=read(REG)
    folder=OUT/"runs"/config["id"]/arm
    folder.mkdir(exist_ok=True,parents=True)
    weights, record=folder/"weights.pt",folder/"training.json"
    if record.exists():
        saved=read(record)
        assert saved["config"]==config and saved["sha256"]==digest(weights)
        return
    setup(config["seed"])
    pool=Pool("train",config["geometry_count"])
    net=network(arm,config,reg["input_canvas"])
    # Отдельное создание CNN не меняет состояния генераторов обучения.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(config["seed"])
        control=Network(config).cuda()
        assert all(torch.equal(net.state_dict()[k],v) for k,v in control.state_dict().items())
        with torch.no_grad():
            data=batch([encode(pool.start(c).view,False) for c in pool.cases[:12]])
            assert torch.equal(net(data)[0],control(data)[0])
    del control
    initial={k:p.detach().clone() for k,p in net.named_parameters()}
    start=time.monotonic()
    history=train_q(net.train(),torch.optim.Adam(net.parameters(),lr=config["lr"]),config,pool,
                    random.Random(config["seed"]),np.random.default_rng(config["seed"]))
    delta={k:float((p-initial[k]).detach().abs().max()) for k,p in net.named_parameters()}
    assert all(math.isfinite(x) for x in delta.values()) and max(delta.values())>0
    assert all(math.isfinite(s["mean_loss"]) for s in history)
    assert [r["steps"] for r in history]==[24576]*3 and [r["updates"] for r in history]==[512]*3
    if arm=="query":
        assert delta["query.weight"]>0 and delta["combine.weight"]>0
    tmp=folder/"pending.pt"
    torch.save(net.state_dict(),tmp)
    tmp.replace(weights)
    archive=SOURCE/"runs"/config["id"]
    max_diff=None
    if arm=="cnn":
        assert read(archive/"run.json")["weights_sha256"]==digest(archive/"weights.pt")
        previous=torch.load(archive/"weights.pt",map_location="cuda",weights_only=True)
        max_diff=max(float((value.float()-previous[k].float()).abs().max()) for k,value in net.state_dict().items())
    save(record,dict(arm=arm,config=config,sha256=digest(weights),train_seconds=time.monotonic()-start,
                     params=sum(p.numel() for p in net.parameters()),max_abs_update=delta,
                     archive_max_abs_difference=max_diff,history=history))
    print(config["id"],arm,"trained",round(time.monotonic()-start,1),"s",flush=True)
    pool.db.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    register()
    jobs=[(arm,c) for c in read(REG)["configs"] for arm in ARMS]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(train_one,jobs))
