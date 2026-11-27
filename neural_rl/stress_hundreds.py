"""E-0013. Проверка сохранённых весов E-0012 на трёх исходных сотнях E-0010."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from importlib import import_module
from itertools import permutations
import json
from pathlib import Path
import sqlite3

from maze.datasets import Case, geometry
from maze.environment import Environment, Status
from maze.grid import locations
from maze.interface import Move
from neural_rl.experiment import digest, read, save
from neural_rl.perception import astar

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/E-0012"
OUT = ROOT / "results/E-0013"
SPLITS = ("I", "II", "III")
MOVES = tuple(Move)


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bench():
    return import_module("results.E-0010.bench")


def maps():
    return [(s, Case(name, SPLITS.index(s)+1, level)) for s, name, _, _, level in bench().cases() if s in SPLITS]


def witness(level):
    sites = locations(level.grid, "F") + locations(level.grid, "R")
    options = sorted(permutations(sites, 2), key=lambda pair: level.grid[pair[0][1]][pair[0][0]] == level.grid[pair[1][1]][pair[1][0]])
    for pair in options:
        env = Environment(level)
        actions = []
        for target in (*pair,level.goal):
            for move in astar(level.grid,env.position,target):
                env.step(move)
                actions.append(str(MOVES.index(move)))
                if env.status != Status.RUNNING:
                    break
            if env.status != Status.RUNNING:
                break
        if env.status == Status.SUCCESS:
            return dict(actions="".join(actions),steps=env.steps,refuels=env.refuels,repairs=env.repairs)
    raise ValueError(f"Не найден физический финиш для {level.title}")


def inventory():
    frozen = read(SOURCE/"frozen.json")
    registry = read(SOURCE/"registration.json")
    models = {}
    for config in registry["runs"]:
        name = config["id"]
        folder = SOURCE/"runs"/name
        run = read(folder/"run.json")
        assert config == run["config"]
        models[name] = dict(config=config, weights=(folder/"weights.pt").relative_to(ROOT).as_posix(),
                            sha256=run["weights_sha256"], kind="primary", frozen=frozen[config["family"]]["id"]==name)
    for family, base in frozen.items():
        folder = SOURCE/"diagnostics"/"training"/family
        run = read(folder/"run.json")
        assert run["base"] == base
        name = base["id"]
        models[name+"-diag"] = dict(config=models[name]["config"], weights=(folder/"weights.pt").relative_to(ROOT).as_posix(),
                                    sha256=run["sha256"], kind="diagnostic", frozen=False)
    assert len(models)==204 and len({m["sha256"] for m in models.values()})==204
    assert all(digest(ROOT/m["weights"])==m["sha256"] for m in models.values())
    return models


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/"models").mkdir(exist_ok=True)
    cases = maps()
    assert Counter(s for s,_ in cases)==dict.fromkeys(SPLITS,100)
    assert len({c.name for _,c in cases})==300
    locked = read(ROOT/"results/E-0009/frozen.json")["sha256_lf"]
    hashes = {}
    for s in SPLITS:
        path = bench().SETS[s]
        key = path.relative_to(ROOT).as_posix()
        hashes[s] = sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        assert hashes[s]==locked[key]
    evidence = {c.name:dict(split=s, geometry=geometry(c.level.grid), **witness(c.level)) for s,c in cases}
    walls = {r["geometry"] for r in evidence.values()}
    assert len(walls)==300
    with sqlite3.connect((SOURCE/"corpus.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        train_overlap = walls & {r[0] for r in db.execute("SELECT hash FROM geoms WHERE split='train'")}
    with gzip.open(SOURCE/"diagnostics/train.jsonl.gz","rt",encoding="utf-8") as stream:
        diag_overlap = walls & {json.loads(line)["geometry"] for line in stream}
    models = inventory()
    manifest = dict(source="E-0012",map_source="E-0010",sets_sha256_lf=hashes,
                    train_overlap=sorted(train_overlap),diagnostic_train_overlap=sorted(diag_overlap),
                    input_canvas=read(SOURCE/"registration.json")["input_canvas"],models=models)
    path = OUT/"manifest.json"
    if path.exists() and read(path)!=manifest:
        raise ValueError("Исходные карты или веса изменились с начала E-0013")
    save(OUT/"witnesses.json",evidence)
    save(path,manifest)
    print(f"Допущено {len(cases)} карт, зарегистрировано {len(models)} моделей",flush=True)


def worker(names):
    import torch
    from neural_rl import perception
    from neural_rl.networks import Network
    from neural_rl.rollout import MapPool, evaluate
    manifest = read(OUT/"manifest.json")
    perception.SIDE = manifest["input_canvas"]
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():
        raise RuntimeError("Требуется CUDA")
    cases = maps()
    pools = [MapPool([c for _,c in cases[i:i+50]]) for i in range(0,len(cases),50)]
    reference={c.name:s for s,c in cases}
    joblog = OUT/f"progress-{names[0]}.jsonl"
    for name in names:
        path = OUT/"models"/(name+".json")
        model = manifest["models"][name]
        if path.exists():
            saved = read(path)
            if saved["sha256"]!=model["sha256"] or {r["map"] for r in saved["rows"]}!=set(reference):
                raise ValueError(f"Не совпал сохранённый результат {name}")
            continue
        weights=ROOT/model["weights"]
        assert digest(weights)==model["sha256"]
        net=Network(model["config"]).cuda()
        net.load_state_dict(torch.load(weights,map_location="cuda",weights_only=True),strict=True)
        rows=[r for pool in pools for r in evaluate(net,pool,1)]
        assert len(rows)==len({r["map"] for r in rows})==len(reference)
        for r in rows:
            r["split"]=reference[r["map"]]
        wins={s:sum(r["outcome"]=="SUCCESS" and r["split"]==s for r in rows) for s in SPLITS}
        save(path,dict(model=name,sha256=model["sha256"],wins=wins,rows=rows))
        with joblog.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=stamp(),model=name,wins=wins))+"\n")


def run(workers):
    models=read(OUT/"manifest.json")["models"]
    first=sorted(models,key=lambda name: (not models[name]["frozen"],models[name]["kind"]!="diagnostic",name))
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(worker,[first[i::workers] for i in range(workers)]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
    else:
        run(args.workers)
