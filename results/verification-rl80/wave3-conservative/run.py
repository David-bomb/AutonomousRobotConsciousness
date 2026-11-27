"""Предрегистрация, один изолированный физический эпизод на карту и проверка журнала."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from evaluate import checked_lock,lock_data,receipt
from suite import ARCHIVE,ROOT,SPLITS,checked_suite,file_hash,object_hash,read,write_new
from maze.environment import Environment,Status
from maze.grid import destination
from maze.interface import Move,Observation
from maze.levels import Level

WEIGHTS="f9bd509a7d4778f4d0e33cd1ad3924f8eb8a0923086c8448b42ac7ebddee35bf"
ID="wave3-conservative/B"
OUT=HERE/"independent"
MOVES=tuple(Move)
KEYS=tuple(f.name for f in fields(Observation))


def packet(view):
    data={k:getattr(view,k) for k in KEYS}
    data["last_move"]=view.last_move.name if view.last_move else None
    return json.dumps(data,separators=(",",":")).encode()+b"\n"


def drive(level,weights):
    env=Environment(level)
    actions=bytearray()
    settings={k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","TEMP","TMP","SYSTEMDRIVE","USERPROFILE",
                                        "LOCALAPPDATA","APPDATA") if k in os.environ}
    settings.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                    PYTHONNOUSERSITE="1",PYTHONHASHSEED="0")
    with tempfile.TemporaryDirectory() as folder,subprocess.Popen(
            [sys.executable,"-B","-u",str(HERE/"worker.py")],cwd=folder,env=settings,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE) as worker:
        try:
            worker.stdin.write(f"{len(weights)}\n".encode()+weights)
            worker.stdin.flush()
            ready=json.loads(worker.stdout.readline())
            # Под Windows venv может запустить отдельный процесс-интерпретатор.
            assert type(ready["pid"]) is int and ready["pid"]>0
            assert ready==dict(ready=True,sha256=WEIGHTS,previous=[None,None],pid=ready["pid"])
            while env.status==Status.RUNNING:
                worker.stdin.write(packet(env.view))
                worker.stdin.flush()
                move=Move[worker.stdout.readline().decode("ascii").strip()]
                actions.append(MOVES.index(move))
                env.step(move)
            worker.stdin.close()
            closed=json.loads(worker.stdout.readline())
            assert worker.wait()==0
        finally:
            if worker.poll() is None:
                try:
                    worker.stdin.close()
                except BrokenPipeError:
                    pass
                worker.stdout.read()
                worker.wait()
    digest=sha256(actions).hexdigest()
    assert closed==dict(calls=env.steps,actions_sha256=digest)
    return dict(outcome=env.status.name,steps=env.steps,refuels=env.refuels,repairs=env.repairs,
                actions_sha256=digest,actions="".join(map(str,actions)),limit=env.limit,initial=[level.fuel,level.health],
                final=[env.fuel,env.health],final_position=list(env.position),initial_registers=ready["previous"],
                worker_pid=ready["pid"],checkpoint_sha256=ready["sha256"],policy_calls=closed["calls"])


def ledger(level,r):
    f,h=level.fuel,level.health
    limit=8*len(level.grid)*len(level.grid[0])
    p=level.start
    refuels=repairs=0
    status="RUNNING"
    seq=bytes(map(int,r["actions"]))
    assert len(seq)>0 and r["actions_sha256"]==sha256(seq).hexdigest()
    for steps,i in enumerate(seq,1):
        assert status=="RUNNING" and i<8
        q=destination(level.grid,p,MOVES[i])
        f-=1
        h-=1
        if q is not None:
            p=q
            symbol=level.grid[p[1]][p[0]]
            if symbol=="F":f,refuels=100,refuels+1
            if symbol=="R":h,repairs=100,repairs+1
        status=("SUCCESS" if p==level.goal else "EMPTY" if f==0 else "BROKEN" if h==0
                else "LIMIT" if steps==limit else "RUNNING")
    assert (r["outcome"],r["steps"],r["policy_calls"],r["refuels"],r["repairs"],r["limit"])==(
        status,len(seq),len(seq),refuels,repairs,limit) and status!="RUNNING"
    assert r["initial"]==[level.fuel,level.health] and r["final"]==[f,h] and r["final_position"]==list(p)
    assert r["initial_registers"]==[None,None] and r["checkpoint_sha256"]==WEIGHTS


def prepare():
    selection_dir=HERE.parent/"wave3-selection"
    director=read(HERE/"director-plan.json")
    selection=read(HERE/"selection.json")
    origin=read(HERE/"origin.json")
    assert file_hash(HERE/"selection.json")==director["selection_sha256"]
    assert selection==read(selection_dir/"selection.json") and selection["winner"]=="B"
    assert read(selection_dir/"receipt.json")["episodes"]==800
    assert WEIGHTS==selection["weights_sha256"]==director["sha256"]==origin["sha256"]
    assert origin["threshold"]==.9 and file_hash(HERE/"model.pt")==WEIGHTS
    assert file_hash(selection_dir/"lock.json")==selection["lock_sha256"]
    selected=read(selection_dir/"lock.json")["registration"]["source_sha256"]
    sources=[p for p in sorted((HERE/"frozen").rglob("*")) if p.is_file() and p.suffix in (".py",".dll",".cpp")]
    for p in sources:
        old=selection_dir/"frozen/diplom"/p.relative_to(HERE/"frozen")
        assert file_hash(p)==selected[old.relative_to(ROOT).as_posix()]
    folder=HERE/"frozen/results/E-0016/prev-two"
    for name,digest in origin["sources"].items():
        if (folder/name).is_file():
            assert file_hash(folder/name)==digest
    level=Level.parse("Синтетический запуск\n16 16\n#########\n#A.....B#\n#F.R.F.R#\n#########")
    data=(HERE/"model.pt").read_bytes()
    first,second=[drive(level,data) for _ in range(2)]
    for result in (first,second):
        ledger(level,result)
    assert first["actions"]==second["actions"]
    chain=dict(selection_sha256=director["selection_sha256"],source_count=len(sources),
               synthetic_restarts=2,synthetic_actions_sha256=first["actions_sha256"],
               synthetic_steps=first["steps"],threshold=origin["threshold"])
    write_new(HERE/"chain.json",chain)
    deps=sources+[HERE/name for name in ("run.py","worker.py","registration.md","chain.json","origin.json",
                                         "director-plan.json","selection.json")]
    lock=lock_data(folder/"cautious_policy.py",HERE/"model.pt",WEIGHTS,ARCHIVE,deps)
    write_new(HERE/"lock.json",lock)
    digest=object_hash(lock)
    write_new(HERE/"registered.json",dict(utc=datetime.now(timezone.utc).isoformat(),candidate=ID,
              weights_sha256=WEIGHTS,lock_sha256=digest,sets=list(SPLITS),episodes_per_map=1))
    print(digest)




def main():
    if sys.argv[1]=="register":
        prepare()
        return
    lock_sha=sys.argv[2]
    meta=read(HERE/"registered.json")
    lock=checked_lock(HERE/"lock.json",lock_sha)
    assert meta["lock_sha256"]==lock_sha and meta["candidate"]==ID and lock["checkpoint_sha256"]==WEIGHTS
    cases=checked_suite(ARCHIVE)
    if sys.argv[1]=="run":
        OUT.mkdir(exist_ok=False)
        (OUT/"rows").mkdir()
        weights=(HERE/"model.pt").read_bytes()
        assert sha256(weights).hexdigest()==WEIGHTS
        def work(index):
            c=cases[index]
            r=drive(c.level,weights)
            row=dict(split=c.split,id=c.name,geometry=c.geometry,scenario_sha256=c.scenario,**r)
            write_new(OUT/"rows"/f"{index:03d}.json",row)
            ledger(c.level,row)
        # Три сотни продвигаются одновременно. Итоговый порядок задан манифестом.
        order=[s*100+i for i in range(100) for s in range(3)]
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(work,order))
        checked_lock(HERE/"lock.json",lock_sha)
    else:
        assert sys.argv[1]=="verify"
    assert {p.name for p in (OUT/"rows").iterdir()}=={f"{i:03d}.json" for i in range(300)}
    rows=[read(OUT/"rows"/f"{i:03d}.json") for i in range(300)]
    for c,r in zip(cases,rows,strict=True):
        ledger(c.level,r)
    encoded=b"".join((json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n").encode() for r in rows)
    log=OUT/"episodes.jsonl"
    if sys.argv[1]=="run":
        with log.open("xb") as f:f.write(encoded)
    else:
        assert log.read_bytes()==encoded
    summary=receipt(cases,OUT,lock,lock_sha)
    groups={}
    for split in SPLITS:
        batch=[r for r in rows if r["split"]==split]
        groups[split]=dict(maps=len(batch),outcomes=dict(Counter(r["outcome"] for r in batch)),
                           steps=sum(r["steps"] for r in batch),refuels=sum(r["refuels"] for r in batch),
                           repairs=sum(r["repairs"] for r in batch))
    full=dict(**summary,candidate=ID,by_split=groups,empty_registers_checked=len(rows),
              unique_geometries=len({c.geometry for c in cases}),unique_ids=len({c.name for c in cases}),violations=[])
    if sys.argv[1]=="run":
        write_new(OUT/"summary.json",summary)
        write_new(HERE/"receipt.json",full)
    else:
        assert read(OUT/"summary.json")==summary and read(HERE/"receipt.json")==full
    print(summary["successes"])


if __name__=="__main__":
    main()
