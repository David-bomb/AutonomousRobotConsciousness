"""Первичные архивы открываются только для чтения; SQLite с WAL разбирается на копии."""
from collections import Counter, defaultdict
import csv
import ctypes
import math
import os
import platform
from statistics import median
import struct
import subprocess
from tempfile import TemporaryDirectory
import winreg
import zlib
import json

import numpy as np
import torch
from maze.datasets import geometry
from .data import OUT, ROOT, CORPUS, STRESS, copy_corpus, digest, read, rows, save
from hashlib import sha256


def distribution(items):
    n=len(items)
    both=[r for r in items if max(r["fuel"],r["health"])<r["direct"]]
    interval=lambda k:[min(r[k] for r in items),median(r[k] for r in items),max(r[k] for r in items)]
    return dict(n=n,direct=interval("direct"),route=interval("route"),fuel=interval("fuel"),
                health=interval("health"),rectangles=sum(r["w"]!=r["h"] for r in items),
                sizes=dict(Counter(f"{r['w']}x{r['h']}" for r in items)),
                both=len(both),both_direct_28_84=sum(28<=r["direct"]<=84 for r in both),
                fuel_lower=sum(r["fuel"]<r["health"] for r in both),
                health_lower=sum(r["health"]<r["fuel"] for r in both))


def main():
    e13=ROOT/"results/E-0013"
    e15=ROOT/"results/E-0015"
    reg=read(e15/"registration.json")
    sources=list(reg["base_sources"])+["neural_rl/stress_hundreds.py","neural_rl/stress_hundreds_report.py",
        "results/E-0010/bench.py","results/E-0015/evaluate_models.py","results/E-0015/train.py",
        "results/E-0013/manifest.json","results/E-0013/witnesses.json","results/E-0012/diagnostics/control.jsonl.gz"]
    hashes={f:digest(ROOT/f) for f in sources}
    assert all(hashes[f]==v for f,v in reg["base_sources"].items())
    files={s:dict(path=p.relative_to(ROOT).as_posix(),raw=digest(p),
           lf=sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()) for s,p in zip(("I","II","III"),STRESS)}
    assert {k:v["lf"] for k,v in files.items()}==reg["stress_maps"]==read(e13/"manifest.json")["sets_sha256_lf"]
    archive={p.name:digest(p) for p in (CORPUS,CORPUS.with_suffix(".sqlite-wal")) if p.exists()}
    assert archive["corpus.sqlite"]==reg["train_db_sha256"]
    assert hashes["results/E-0012/diagnostics/control.jsonl.gz"]==reg["control_sha256"]
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version,compute_cap",
                                 "--format=csv,noheader"],text=True).strip()
    memory=ctypes.create_string_buffer(64)
    struct.pack_into("<I",memory,0,64)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(memory)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu=winreg.QueryValueEx(key,"ProcessorNameString")[0].strip()
    hardware=dict(cpu=cpu,logical_processors=os.cpu_count(),ram_bytes=struct.unpack_from("<Q",memory,8)[0],
                  gpu_name_vram_driver_capability=gpu,python=platform.python_version(),os=platform.platform(),
                  torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available())
    manifest=read(e13/"manifest.json")
    stress={}
    witness=read(e13/"witnesses.json")
    for f in STRESS:
        for r in rows(f):
            assert geometry(r["grid"])==r["geometry"]==witness[r["name"]]["geometry"]
            stress[r["name"]]=dict(w=r["width"],h=r["height"],fuel=r["fuel"],health=r["health"],
                                   direct=r["direct"],route=witness[r["name"]]["steps"])
    dist={"stress":distribution(list(stress.values()))}
    split_hashes={}
    with TemporaryDirectory() as folder:
        db=copy_corpus(folder)
        for split,limit in (("train",30000),("validation_id",60)):
            series=[]
            fingerprint=sha256()
            for idx,blob in db.execute("SELECT idx,data FROM geoms WHERE split=? AND idx<? ORDER BY idx",(split,limit)):
                for r in json.loads(zlib.decompress(blob)):
                    title,stock,*grid=r["source"].splitlines()
                    assert r["proof"]["status"]=="SUCCESS" and sha256(r["source"].encode()).hexdigest()==r["sha256"]
                    fingerprint.update(f"{title} {r['sha256']}\n".encode())
                    f,h=map(int,stock.split())
                    series.append(dict(w=len(grid[0]),h=len(grid),fuel=f,health=h,direct=r["direct"],route=r["proof"]["steps"]))
            assert len(series)==4*limit
            dist[split]=distribution(series)
            split_hashes[split]=fingerprint.hexdigest()
        db.close()
    totals=Counter()
    winners={}
    with (e13/"scores.csv").open(encoding="utf-8-sig") as stream:
        scores={r["model"]:r for r in csv.DictReader(stream)}
    assert len(scores)==len(manifest["models"])==204
    for name,model in manifest["models"].items():
        record=read(e13/"models"/(name+".json"))
        assert record["sha256"]==model["sha256"]
        rr=record["rows"]
        assert len(rr)==len(stress) and {r["map"] for r in rr}==stress.keys()
        for s in ("I","II","III"):
            group=[r for r in rr if r["split"]==s]
            assert len(group)==100 and sum(r["outcome"]=="SUCCESS" for r in group)==int(scores[name][s])==record["wins"][s]
        totals.update(r["outcome"] for r in rr)
        totals["both"]+=sum(bool(r["repairs"] and r["refuels"]) for r in rr)
        if sum(record["wins"].values()):
            winners[name]=record["wins"]
    traces=[r for f in (ROOT/"results/E-0014").glob("*-traces.json") for r in read(f)]
    assert len(traces)==1200
    losses=dict(n=len(traces),median_irreversible=median(r["loss"]["step"] for r in traces),
                by_step10=sum(r["loss"]["step"]<=10 for r in traces),before_service=sum(not r["loss"]["after_service"] for r in traces),
                median_stop=median(r["steps"] for r in traces))
    history=defaultdict(Counter)
    e15stress=Counter()
    val=Counter()
    for folder in sorted((e15/"runs").glob("*/*")):
        train=read(folder/"training.json")
        ev=read(folder/"evaluation.json")
        assert digest(folder/"weights.pt")==ev["sha256"]==train["sha256"]
        for stage in train["history"]:
            history[stage["stage"]].update(stage["outcomes"])
        assert len(ev["stress"])==300 and {r["map"] for r in ev["stress"]}==stress.keys()
        e15stress.update(r["outcome"] for r in ev["stress"])
        e15stress["both"]+=sum(bool(r["repairs"] and r["refuels"]) for r in ev["stress"])
        val[folder.name]+=sum(r["outcome"]=="SUCCESS" for r in ev["validation"])
    g=reg["configs"][0]["gamma"]
    discount=dict(gamma=g,effective_sum=1/(1-g),half_weight=math.log(.5)/math.log(g),
                  weights={t:g**(t-1) for t in (28,52,77,100)},terminal_at={t:8*g**(t-1) for t in (28,52,77,100)})
    save(OUT/"baseline.json",dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         hardware=hardware,sha256_sources=hashes,stress_sets=files,corpus_files=archive,
         split_name_source_sha256=split_hashes,distribution=dist,E13=dict(outcomes=totals,winners=winners),
         E14=losses,E15=dict(train=history,stress=e15stress,validation_wins=val),discount=discount))
    print("E-0013",totals,"E-0014",losses,"E-0015",history,e15stress,flush=True)


if __name__=="__main__":
    main()
