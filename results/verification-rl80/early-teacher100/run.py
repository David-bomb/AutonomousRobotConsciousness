"""Два физических эпизода на карте train, общий поздний контроллер joint400."""
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
V=HERE.parent
ROOT=V.parents[1]
sys.path.insert(0,str(ROOT))
from maze.datasets import geometry
from maze.environment import Environment,Status
from maze.interface import Move,Observation
from maze.levels import Level

BANK=V/"phase100-train/train.jsonl"
BASE=V/"wave2-service-window"
OLD=V/"mandatory-proof-train"
LATE=ROOT/"results/E-0016/sequence/runs/joint400/rl.pt"
SIGN={"fixed":"eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c",
      "joint":"ebf419fe5ed73a9204b8cc976bedbc1ca53aeffe615605fb443f20fb4f852f56",
      "bank":"da9f168ef715811273125a951986ab23719e078088e0c96cec816bcfbafa6f3e"}
SYSTEM=Path(sys.base_prefix)/"python.exe"
ORDER=tuple(Move)
KEYS=tuple(f.name for f in fields(Observation))


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def save(p,obj):
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
        f.write("\n")


def spec():
    old=read(OLD/"lock.json")["registration"]
    assert sha(OLD/"lock.json")=="6f391b4f68d1995238e72ae99b0fce622fb7dfb51a97254014d375a20feb2efc"
    for k,h in old["sources_sha256_lf"].items():
        assert sha256((ROOT/k).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==h
    adapter=OLD/"adapter.py"
    assert sha(adapter)==old["verification_sha256"][adapter.relative_to(ROOT).as_posix()]
    fixed=read(BASE/"lock.json")
    assert sha(BASE/"frozen/model.pt")==fixed["checkpoint_sha256"]==SIGN["fixed"]
    assert sha(BASE/"bridge.py")==fixed["adapter_sha256"]
    deps=[]
    for k,h in fixed["dependencies"].items():
        p=ROOT/k
        if p.is_relative_to(BASE/"frozen"):
            assert sha(p)==h
            deps.append(p)
    assert sha(LATE)==read(BASE/"manifests/joint400.json")["weights_sha256"]==SIGN["joint"]
    assert sha(BANK)==SIGN["bank"]
    runtime=json.loads(subprocess.check_output([str(SYSTEM),"-B","-c","import json,sys,numpy,scipy;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__)))"]))
    assert runtime==old["runtime"]
    deps += [HERE/p for p in ("preregistration.md","run.py","neural.py")]
    deps += [OLD/"lock.json",adapter,V/"policy_worker.py",LATE,BANK,BASE/"lock.json",BASE/"frozen/model.pt",BASE/"bridge.py"]
    deps += [ROOT/"maze"/(n+".py") for n in ("grid","interface","datasets","levels","memory","environment")]
    return dict(sign=SIGN,teacher_runtime=runtime,torch=version("torch"),numpy=version("numpy"),python=sys.version,
                sources={p.relative_to(ROOT).as_posix():sha(p) for p in deps})


def level(row):
    return Level.parse("\n".join([row["name"],f'{row["fuel"]} {row["health"]}',*row["grid"]]))


def pack(view):
    packet={k:getattr(view,k) for k in KEYS}
    packet["last_move"]=view.last_move.name if view.last_move else None
    return json.dumps(packet,sort_keys=True,separators=(",",":")).encode()+b"\n"


class Brain:
    def __init__(self,name):
        self.name=name

    def __enter__(self):
        self.temp=tempfile.TemporaryDirectory()
        teacher=self.name=="teacher"
        exe=str(SYSTEM) if teacher else sys.executable
        code=(OLD/"adapter.py" if teacher else HERE/"neural.py").read_bytes()
        data=b"previous" if teacher else (LATE if self.name=="joint" else BASE/"frozen/model.pt").read_bytes()
        settings={k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","TEMP","TMP","SYSTEMDRIVE",
                                            "USERPROFILE","LOCALAPPDATA","APPDATA") if k in os.environ}
        settings.update(CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                        PYTHONNOUSERSITE="1",PYTHONHASHSEED="0")
        self.p=subprocess.Popen([exe,"-B","-u",str(V/"policy_worker.py")],cwd=self.temp.name,env=settings,
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        for blob in (code,data):
            self.p.stdin.write(str(len(blob)).encode()+b"\n"+blob)
        self.p.stdin.flush()
        return self

    def decide(self,packet):
        self.p.stdin.write(packet)
        self.p.stdin.flush()
        name=self.p.stdout.readline().decode("ascii").strip()
        if name not in Move.__members__:
            raise RuntimeError(f"Нет команды {self.name}: {name!r}")
        return Move[name]

    def __exit__(self,etype,err,tb):
        try:
            self.p.stdin.close()
            self.p.stdout.read()
            code=self.p.wait()
            if not etype and code:raise RuntimeError(f"Процесс {self.name}: {code}")
        finally:
            self.p.stdout.close()
            self.temp.cleanup()


def coverage(memory,sites):
    def complete(x,y):
        return all(memory[j][i]!="?" for j in range(y-1,y+2) for i in range(x-1,x+2))
    return all(any(complete(x,y) for x,y in group) for group in sites)


def service_sites(view):
    groups=[[(x,y) for y,row in enumerate(view.memory) for x,c in enumerate(row) if c==kind] for kind in "FR"]
    assert all(len(g)==2 for g in groups)
    assert all(0<x<len(view.memory[0])-1 and 0<y<len(view.memory)-1 for g in groups for x,y in g)
    return groups


def add_label(digest,packet,previous,index):
    digest.update(packet)
    digest.update(json.dumps(previous,separators=(",",":")).encode()+b"\n")
    digest.update(bytes([index]))


def drive(lvl,hybrid):
    env=Environment(lvl)
    sites=service_sites(env.view)
    first_gate=0 if coverage(env.view.memory,sites) else None
    first_both=None
    actions=bytearray()
    labels=sha256()
    previous=(None,None)
    late_count=0
    for name in (("teacher","joint") if hybrid else ("fixed",)):
        if env.status!=Status.RUNNING or (name=="teacher" and first_gate is not None):
            continue
        with Brain(name) as brain:
            while env.status==Status.RUNNING:
                v=env.view
                packet=pack(v)
                move=brain.decide(packet)
                i=ORDER.index(move)
                if name=="teacher":
                    assert first_gate is None
                    add_label(labels,packet,previous,i)
                elif name=="joint":
                    assert first_gate is not None
                    late_count+=1
                elif first_gate is not None and min(v.fuel-v.fuel_distance,v.health-v.repair_distance)>=-101:
                    late_count+=1
                previous=(v.fuel_distance,v.repair_distance)
                actions.append(i)
                env.step(move)
                if first_both is None and env.refuels and env.repairs:
                    first_both=env.steps
                if first_gate is None and coverage(env.view.memory,sites):
                    first_gate=env.steps
                    if name=="teacher":
                        break
    assert env.status!=Status.RUNNING and len(actions)==env.steps
    early_len=first_gate if first_gate is not None else env.steps
    return dict(outcome=env.status.name,steps=env.steps,actions="".join(map(str,actions)),actions_sha256=sha256(actions).hexdigest(),
                first_open_windows=first_gate,first_both_services=first_both,refuels=env.refuels,repairs=env.repairs,
                late_calls=late_count,teacher_steps=early_len if hybrid else 0,
                teacher_labels_sha256=labels.hexdigest() if hybrid else None)


def run(row):
    lvl=level(row)
    assert geometry(lvl.grid)==row["geometry"]
    initial=Environment(lvl).view
    slack=min(initial.fuel-initial.fuel_distance,initial.health-initial.repair_distance)
    assert slack==row["min_start_slack"]
    return dict(id=row["name"],tier=row["tier"],geometry=row["geometry"],family=row["family"],
                initial=[lvl.fuel,lvl.health],initial_slack=slack,fixed=drive(lvl,False),hybrid=drive(lvl,True))


def main():
    lock=HERE/"lock.json"
    reg=spec()
    if sys.argv[1]=="register":
        save(lock,dict(utc=datetime.now(timezone.utc).isoformat(),registration=reg))
        print(sha(lock))
        return
    assert sys.argv[1]=="run" and sha(lock)==sys.argv[2] and read(lock)["registration"]==reg
    rows=list(map(json.loads,BANK.read_bytes().splitlines()))
    assert len({r["geometry"] for r in rows})==len(rows)==100 and Counter(r["tier"] for r in rows)==dict.fromkeys(("H1","H2","H3","H4"),25)
    out=HERE/"episodes"
    out.mkdir(exist_ok=False)
    def work(r):
        save(out/(r["name"]+".json"),run(r))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work,rows))
    assert spec()==reg


if __name__=="__main__":
    main()
