"""Один чистый CPU-процесс. По каналу приходят байты модели и только Observation."""
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE/"frozen"),str(ROOT)]
import torch
from maze.interface import Move,Observation

WEIGHTS="f9bd509a7d4778f4d0e33cd1ad3924f8eb8a0923086c8448b42ac7ebddee35bf"
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
    def guard(event,args):
        if event in ("open","ctypes.dlopen") and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(ROOT) and not p.is_relative_to(venv):
                if not (p.suffix.lower() in {".py",".pyc",".pyd",".dll"} and
                                      any(p.is_relative_to(d) for d in allowed)):
                    raise PermissionError("Политике недоступны данные проекта")
        if event in ("socket.connect","subprocess.Popen","os.system"):
            raise PermissionError(event)
    sys.addaudithook(guard)
    api=import_module("results.E-0016.prev-two.cautious_policy")
    for name,m in sys.modules.copy().items():
        if name.startswith("results.E-0016.") and getattr(m,"__file__",None):
            assert Path(m.__file__).resolve().is_relative_to(frozen)
    ctrl=api.load(checkpoint)
    previous=(None,None)
    assert ctrl.previous==previous and ctrl.threshold==.9 and not ctrl.ablate
    networks=(ctrl.early,ctrl.correction,ctrl.late.net,ctrl.late.safety)
    assert torch.get_num_threads()==1 and not torch.cuda.is_initialized()
    assert all(not n.training and all(p.device.type=="cpu" and not p.requires_grad for p in n.parameters()) for n in networks)
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
