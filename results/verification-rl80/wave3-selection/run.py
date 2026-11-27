"""Четыре метода на двух новых сотнях. Регистрация отделена от оценки."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime,timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from maze.datasets import geometry
from maze.environment import Environment,Status
from maze.interface import Move,Observation
from maze.levels import Level

SEEDS=(9167001,9167002)
SIGNS={"A":"eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c",
       "B":"f9bd509a7d4778f4d0e33cd1ad3924f8eb8a0923086c8448b42ac7ebddee35bf",
       "C":"839140fa1346c675d550f6ee4ace6aaa5eb9f7a1d07c254a2be8a6e86f6f0bed",
       "D":"07859179f66199b6ecf95c5ef662f3bde4ad31b5956aa7ef8ae568446d958576"}
MOVES=tuple(Move)
KEYS=tuple(f.name for f in fields(Observation))


def read(p):
    return json.loads(p.read_bytes())


def save(path,data):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
        f.write("\n")


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def spec():
    banks=read(HERE/"bank-receipt.json")
    assert banks["seeds"]==list(SEEDS) and banks["mutual_overlap"]==0
    for seed in SEEDS:
        g=banks["sets"][str(seed)]
        assert g["maps"]==100 and g["overlap"]==0 and g["sha256"]==sha(HERE/f"{seed}.jsonl")
    for m,h in SIGNS.items():
        assert sha(HERE/"frozen/weights"/(m+".pt"))==h
    b=read(HERE/"frozen/origins/B.json")
    assert b["sha256"]==SIGNS["B"] and b["threshold"]==.9
    files=[p for p in (HERE/"frozen").rglob("*") if p.is_file() and p.suffix!=".pyc"]
    files.extend(HERE/p for p in ("preregistration.md","worker.py","run.py","banks.py","bank-receipt.json"))
    files.extend(ROOT/"maze"/(n+".py") for n in ("grid","levels","memory","environment","datasets","interface"))
    return dict(weights=SIGNS,seed_sha256={str(s):sha(HERE/f"{s}.jsonl") for s in SEEDS},
                runtime=dict(python=sys.version,numpy=version("numpy"),torch=version("torch")),
                source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})


def level(row):
    return Level.parse("\n".join([row["name"],f'{row["fuel"]} {row["health"]}',*row["grid"]]))


def banks():
    result={}
    used=set()
    for s in SEEDS:
        rows=list(map(json.loads,(HERE/f"{s}.jsonl").read_bytes().splitlines()))
        assert Counter(r["tier"] for r in rows)==dict.fromkeys(("H1","H2","H3","H4"),25)
        for r in rows:
            g=geometry(r["grid"])
            assert g==r["geometry"] and g not in used
            used.add(g)
        result[s]=rows
    assert len(used)==200
    return result


def packet(view):
    d={k:getattr(view,k) for k in KEYS}
    d["last_move"]=view.last_move.name if view.last_move else None
    return json.dumps(d,separators=(",",":")).encode()+b"\n"


def service_sites(view):
    return [[(x,y) for y,row in enumerate(view.memory) for x,s in enumerate(row) if s==kind] for kind in "FR"]


def windows(view,sites):
    return all(any(sum(len(seg)-seg.count("?") for r in view.memory[max(0,y-1):y+2]
                       for seg in [r[max(0,x-1):x+2]])==9 for x,y in group) for group in sites)


def episode(p,row,seed,m):
    env=Environment(level(row))
    sites=service_sites(env.view)
    gate=0 if windows(env.view,sites) else None
    early_late=None
    both=None
    acts=bytearray()
    flags=bytearray()
    p.stdin.write(b"start\n")
    p.stdin.flush()
    ready=json.loads(p.stdout.readline())
    assert ready["previous"]==[None,None] and ready["sha256"]==SIGNS[m]
    while env.status==Status.RUNNING:
        p.stdin.write(packet(env.view))
        p.stdin.flush()
        a,late=p.stdout.readline().decode().strip()
        index=int(a)
        assert index<8 and late in "01"
        acts.append(index)
        flags.append(int(late))
        if late=="1" and early_late is None:early_late=env.steps+1
        env.step(MOVES[index])
        if gate is None and windows(env.view,sites):gate=env.steps
        if both is None and env.refuels and env.repairs:both=env.steps
    p.stdin.write(b"end\n")
    p.stdin.flush()
    done=json.loads(p.stdout.readline())
    digest=sha256(acts).hexdigest()
    assert done==dict(steps=env.steps,sha256=digest)
    return dict(method=m,seed=seed,id=row["name"],geometry=row["geometry"],tier=row["tier"],initial=[row["fuel"],row["health"]],
                weights_sha256=ready["sha256"],initial_registers=ready["previous"],routing=ready["route"],
                outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,limit=env.limit,
                first_both=both,first_windows=gate,first_late_command=early_late,late_actions=sum(flags),
                actions="".join(map(str,acts)),actions_sha256=digest,phases="".join(map(str,flags)))


def play(method,group):
    settings={k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","TEMP","TMP","SYSTEMDRIVE","USERPROFILE",
                                         "LOCALAPPDATA","APPDATA") if k in os.environ}
    settings.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                    PYTHONNOUSERSITE="1",PYTHONHASHSEED="0")
    data=(HERE/"frozen/weights"/(method+".pt")).read_bytes()
    assert sha256(data).hexdigest()==SIGNS[method]
    with tempfile.TemporaryDirectory() as tmp, subprocess.Popen([sys.executable,"-B","-u",str(HERE/"worker.py"),method],
             cwd=tmp,env=settings,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as p:
        try:
            p.stdin.write(str(len(data)).encode()+b"\n"+data)
            p.stdin.flush()
            for i in range(100):
                for s in SEEDS:
                    record=episode(p,group[s][i],s,method)
                    save(HERE/"rows"/f"{s}-{method}-{i:02d}.json",record)
            p.stdin.close()
            assert p.wait()==0
        finally:
            if p.poll() is None:
                try:p.stdin.close()
                except BrokenPipeError:pass
                p.stdout.read()
                p.wait()


def main():
    reg=spec()
    lock=HERE/"lock.json"
    group=banks()
    if sys.argv[1]=="register":
        save(lock,dict(utc=datetime.now(timezone.utc).isoformat(),registration=reg))
        print(sha(lock))
        return
    assert sys.argv[1]=="run" and sha(lock)==sys.argv[2] and read(lock)["registration"]==reg
    (HERE/"rows").mkdir(exist_ok=False)
    with ThreadPoolExecutor(4) as pool:
        tasks=[pool.submit(play,m,group) for m in SIGNS]
        for task in tasks:task.result()
    assert spec()==reg


if __name__=="__main__":
    main()
