"""Один чистый CPU-процесс. По каналу приходят байты модели и только Observation."""
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import torch
from maze.interface import Move,Observation

WEIGHTS="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
MOVES=tuple(Move)


def main():
    source=sys.stdin.buffer
    size=int(source.readline())
    checkpoint=source.read(size)
    digest=sha256(checkpoint).hexdigest()
    assert len(checkpoint)==size and digest==WEIGHTS
    frozen=HERE/"frozen"
    venv=Path(sys.prefix).resolve()
    allowed=(frozen,ROOT/"maze")
    bridge=HERE/"bridge.py"
    def guard(event,args):
        if event in ("open","ctypes.dlopen") and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(ROOT) and not p.is_relative_to(venv):
                if p!=bridge and not (p.suffix.lower() in {".py",".pyc",".pyd",".dll"} and
                                      any(p.is_relative_to(d) for d in allowed)):
                    raise PermissionError("Политике недоступны данные проекта")
        if event in ("socket.connect","subprocess.Popen","os.system"):
            raise PermissionError(event)
    sys.addaudithook(guard)
    api=import_module("results.verification-rl80.wave2-service-window.bridge")
    ctrl=api.load(checkpoint)
    previous=(None,None)
    assert ctrl.previous==previous and ctrl.cfg==[9,9,-101] and not ctrl.ablate
    assert torch.get_num_threads()==1 and all(p.device.type=="cpu" for p in ctrl.a.parameters())
    print(json.dumps(dict(ready=True,sha256=digest,previous=list(previous),pid=os.getpid())),flush=True)
    actions=sha256()
    steps=0
    for line in source:
        data=json.loads(line)
        for k in ("memory","window","position"):
            data[k]=tuple(data[k])
        if data["last_move"] is not None:
            data["last_move"]=Move[data["last_move"]]
        view=Observation(**data)
        assert (view.last_move is None)==(steps==0) and ctrl.previous==previous
        move=ctrl.choose(view)
        previous=(view.fuel_distance,view.repair_distance)
        assert ctrl.previous==previous and type(move) is Move
        actions.update(bytes([MOVES.index(move)]))
        steps+=1
        print(move.name,flush=True)
    print(json.dumps(dict(calls=steps,actions_sha256=actions.hexdigest())),flush=True)


if __name__=="__main__":
    main()
