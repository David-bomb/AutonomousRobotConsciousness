"""Самостоятельная поездка и три зарегистрированных вмешательства на новом train."""
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

from bank import HERE, ROOT, SEED
from maze.datasets import geometry
from maze.environment import Environment, Status
from maze.interface import Move, Observation
from maze.levels import Level
from oracle import Exact

MOVES=tuple(Move)
KEYS=[f.name for f in fields(Observation)]
FRONT=HERE.parent/"frontier48-dev"
BRANCHES=("wave","teacher_wave","teacher_frontier","wave_exact")
BANK_SHA="da9f168ef715811273125a951986ab23719e078088e0c96cec816bcfbafa6f3e"


def read(p):
    return json.loads(p.read_bytes())


def save(p,obj):
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,separators=(",",":"))
        f.write("\n")


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def registration():
    source=ROOT/"results/E-0016/prev-two"
    w=read(source/"weights/wave1.json")
    f=read(FRONT/"lock.json")
    assert sha(HERE/"wave1.pt")==w["sha256"]=="369c0d5616a07a19f6fa992d53bdaa5a3004c5c54344c427f2345272b2f56432"
    assert sha(HERE/"wave1_adapter.py")==w["adapter_sha256"]
    assert sha(source/"registration.json")==w["registration"]
    assert sha(FRONT/"frozen/model.pt")==f["checkpoint_sha256"]=="c921c6cfe26fee80b0ba5bd05946de47012a5d1da35d308092d2116350b2493b"
    assert sha(FRONT/"bridge.py")==f["adapter_sha256"]
    deps=[p for p in sorted((FRONT/"frozen").rglob("*")) if p.suffix in (".py",".cpp",".dll",".pt")]
    registered=read(source/"registration.json")["sources"]
    for p in deps:
        if p.name=="model.pt":
            continue
        assert sha(p)==f["dependencies"][p.relative_to(ROOT).as_posix()]
        name="results/E-0016/"+p.relative_to(FRONT/"frozen").as_posix()
        if name in registered:
            assert sha(p)==registered[name]
    assert sha(HERE/"train.jsonl")==BANK_SHA
    native=[ROOT/"maze"/(n+".py") for n in ("grid","environment","interface","levels","memory","datasets")]
    assert all(sha(p)==registered[p.relative_to(ROOT).as_posix()] for p in native if p.name!="datasets.py")
    files=[HERE/p for p in ("preregistration.md","bank.py","driver.py","worker.py","oracle.py","train.jsonl",
                            "wave1.pt","wave1_adapter.py")]
    files+=deps+native+[ROOT/"results/ARC-dot/source/ARC-stress-100/make_hard_maps.py",FRONT/"bridge.py",
                        ROOT/"scripts/audit_automaton.py",source/"registration.json",source/"weights/wave1.json"]
    return dict(seed=SEED,branches=BRANCHES,tiers=dict.fromkeys(("H1","H2","H3","H4"),25),
                tie_order=[m.name for m in MOVES],python=sys.version,torch=version("torch"),numpy=version("numpy"),
                sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})


def obs(view):
    packet={k:getattr(view,k) for k in KEYS}
    packet["last_move"]=view.last_move.name if view.last_move else None
    return packet


def readings(view):
    return view.fuel_distance,view.repair_distance


def stage(env):
    if env.status!=Status.RUNNING:
        if env.refuels and env.repairs:
            return "terminal_on_second_service"
        return "early_goal" if env.status==Status.SUCCESS else "failed_before_both"
    assert env.refuels and env.repairs
    return "handoff"


def boundary(env,previous):
    return dict(kind=stage(env),observation=obs(env.view),previous=previous,steps=env.steps,remaining=env.limit-env.steps,
                refuels=env.refuels,repairs=env.repairs,status=env.status.name,
                observed=sum(len(s)-s.count("?") for s in env.view.memory))


def full_state(env,actions):
    return dict(outcome=env.status.name,steps=env.steps,fuel=env.fuel,health=env.health,position=env.position,
                refuels=env.refuels,repairs=env.repairs,actions="".join(map(str,actions)),
                actions_sha256=sha256(bytes(actions)).hexdigest())


def play(env,actions):
    previous=(None,None)
    for a in actions:
        previous=readings(env.view)
        env.step(MOVES[a])
    return previous


class Robot:
    def __init__(self,mode,previous=(None,None)):
        self.mode,self.previous=mode,previous

    def __enter__(self):
        self.tmp=tempfile.TemporaryDirectory()
        settings={k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","TEMP","TMP","SYSTEMDRIVE",
                                            "USERPROFILE","LOCALAPPDATA","APPDATA") if k in os.environ}
        settings.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                        PYTHONNOUSERSITE="1",PYTHONHASHSEED="0")
        self.p=subprocess.Popen([sys.executable,"-u","-B",str(HERE/"worker.py"),self.mode],cwd=self.tmp.name,
                                env=settings,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        if self.mode=="wave":
            self.p.stdin.write(json.dumps(self.previous).encode()+b"\n")
            self.p.stdin.flush()
        return self

    def choose(self,view):
        self.p.stdin.write(json.dumps(obs(view),separators=(",",":")).encode()+b"\n")
        self.p.stdin.flush()
        name=self.p.stdout.readline().decode("ascii").strip()
        if name not in Move.__members__:
            raise RuntimeError(f"Процесс политики вернул {name!r}")
        return Move[name]

    def __exit__(self,etype,err,tb):
        try:
            self.p.stdin.close()
            if etype:
                self.p.kill()
            code=self.p.wait()
            if not etype and code:
                raise RuntimeError(f"Код завершения политики {code}")
        finally:
            self.p.stdout.close()
            self.tmp.cleanup()


def as_level(row):
    return Level.parse("\n".join([row["name"],f'{row["fuel"]} {row["health"]}',*row["grid"]]))


def replay(level,actions,snapshot):
    env=Environment(level)
    prev=play(env,actions)
    assert boundary(env,prev)==snapshot
    return env


def episode(row):
    level=as_level(row)
    assert geometry(level.grid)==row["geometry"]
    oracle=Exact(level)
    scores={}
    env=Environment(level)
    optimum=oracle.route(env)
    assert optimum is not None and len(optimum)==row["optimum"]
    cut=0
    last=(None,None)
    while env.status==Status.RUNNING and not (env.refuels and env.repairs):
        last=readings(env.view)
        env.step(MOVES[optimum[cut]])
        cut+=1
    tp=optimum[:cut]
    teacher=boundary(env,last)
    teacher["witness_suffix"]=None
    if teacher["kind"]=="handoff":
        w=oracle.route(env)
        assert w is not None and len(w)==len(optimum)-cut
        teacher["witness_suffix"]="".join(map(str,w))
    ts={k:v for k,v in teacher.items() if k!="witness_suffix"}
    env=Environment(level)
    whole=bytearray()
    own=None
    with Robot("wave") as robot:
        while env.status==Status.RUNNING:
            last=readings(env.view)
            move=robot.choose(env.view)
            env.step(move)
            whole.append(MOVES.index(move))
            if own is None and (env.status!=Status.RUNNING or (env.refuels and env.repairs)):
                own=boundary(env,last)
    scores["wave"]=full_state(env,whole)
    wp=whole[:own["steps"]]
    for mode,branch in (("wave","teacher_wave"),("frontier","teacher_frontier")):
        env=replay(level,tp,ts)
        actions=bytearray(tp)
        if env.status==Status.RUNNING:
            with Robot(mode,ts["previous"]) as robot:
                while env.status==Status.RUNNING:
                    move=robot.choose(env.view)
                    env.step(move)
                    actions.append(MOVES.index(move))
        scores[branch]=full_state(env,actions)
    env=replay(level,wp,own)
    own["witness_suffix"]=None
    rest=b""
    if env.status==Status.RUNNING:
        w=oracle.route(env)
        if w is None:
            rest=whole[len(wp):]
        else:
            own["witness_suffix"]="".join(map(str,w))
            rest=w
        play(env,rest)
        assert (env.status==Status.SUCCESS)==(w is not None)
    scores["wave_exact"]=full_state(env,wp+rest)
    assert all(s["outcome"]!="RUNNING" and len(s["actions"])==s["steps"] for s in scores.values())
    return dict(id=row["name"],tier=row["tier"],family=row["family"],geometry=row["geometry"],initial=[level.fuel,level.health],
                limit=env.limit,optimal_actions="".join(map(str,optimum)),teacher=teacher,own=own,branches=scores)


def main():
    reg=registration()
    lock=HERE/"lock.json"
    if sys.argv[1]=="register":
        save(lock,dict(utc=datetime.now(timezone.utc).isoformat(),registration=reg))
        print(sha(lock))
        return
    assert sys.argv[1]=="run" and sha(lock)==sys.argv[2] and read(lock)["registration"]==json.loads(json.dumps(reg))
    rows=[json.loads(s) for s in (HERE/"train.jsonl").read_bytes().splitlines()]
    assert len({r["geometry"] for r in rows})==len({r["name"] for r in rows})==100
    assert Counter(r["tier"] for r in rows)==reg["tiers"]
    out=HERE/"episodes"
    out.mkdir(exist_ok=False)
    def work(r):
        data=episode(r)
        save(out/(r["name"]+".json"),data)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work,rows))
    assert read(lock)["registration"]==json.loads(json.dumps(registration()))


if __name__=="__main__":
    main()
