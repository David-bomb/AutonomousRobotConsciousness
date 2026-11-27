"""Изолированная политика одного метода. На каждой карте новый контроллер и пустой кэш."""
from functools import _lru_cache_wrapper
from hashlib import sha256
from importlib import import_module
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LAB=Path("C:/Users/David/pers_projects/Codex_research_lab/projects/diplom-rl80")
FROZEN=HERE/"frozen"
sys.path[:0]=[str(FROZEN/"diplom"),str(FROZEN/"lab"),str(ROOT),str(LAB)]
from maze.interface import Move,Observation
import torch

MODULES={"A":"service_window_policy","B":"results.E-0016.prev-two.cautious_policy","C":"tree_policy","D":"panel_policy"}


def guard(event,args):
    if event in ("open","ctypes.dlopen") and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if (path.is_relative_to(ROOT) or path.is_relative_to(LAB)) and not path.is_relative_to(Path(sys.prefix)):
            allow=path.is_relative_to(FROZEN) or path.is_relative_to(ROOT/"maze")
            if not allow or path.suffix.lower() not in (".dll",".py",".pyc",".pyd"):
                raise PermissionError("Файлы карт недоступны политике")
    if event in ("subprocess.Popen","socket.connect","os.system"):
        raise PermissionError(event)


def main(method):
    source=sys.stdin.buffer
    weights=source.read(int(source.readline()))
    sign=sha256(weights).hexdigest()
    sys.addaudithook(guard)
    torch.set_num_threads(1)
    api=import_module(MODULES[method])
    valid_sources=(FROZEN/"lab",FROZEN/"diplom")
    ctl=None
    for line in source:
        if line==b"start\n":
            assert ctl is None
            ctl=api.load(weights)
            assert ctl.previous==(None,None) and not ctl.ablate and torch.get_num_threads()==1
            if method=="B":
                assert abs(ctl.threshold-.9)<1.e-6
                late=lambda v:api.complete(v.memory)
                route={"windows":9,"threshold":ctl.threshold}
            else:
                late=lambda v:ctl.route(v,ctl.cfg)
                route={"cfg":ctl.cfg}
            wrappers=set()
            for name,m in sys.modules.copy().items():
                path=getattr(m,"__file__",None)
                if name.startswith("results.E-0016.") and path:
                    assert Path(path).resolve().is_relative_to(valid_sources[1])
                if name in ("tree_policy","panel_policy","service_window_policy","geodesic_probe","renewable_policy"):
                    assert Path(path).resolve().is_relative_to(valid_sources[0])
                if name.startswith(("maze.","results.E-0016.")) or (path and Path(path).is_relative_to(valid_sources[0])):
                    wrappers.update(x for x in vars(m).values() if isinstance(x,_lru_cache_wrapper))
            for fn in wrappers:fn.cache_clear()
            previous=(None,None)
            cache={}
            actions=sha256()
            steps=0
            print(json.dumps(dict(sha256=sign,previous=previous,route=route)),flush=True)
        elif line==b"end\n":
            assert ctl is not None
            print(json.dumps(dict(steps=steps,sha256=actions.hexdigest())),flush=True)
            ctl=None
            cache.clear()
        else:
            assert ctl is not None
            d=json.loads(line)
            for k in ("memory","window","position"):
                d[k]=tuple(d[k])
            if d["last_move"] is not None:d["last_move"]=Move[d["last_move"]]
            v=Observation(**d)
            assert (v.last_move is None)==(steps==0) and ctl.previous==previous
            flag=int(late(v))
            move=ctl.choose(v,cache)
            assert type(move) is Move
            previous=(v.fuel_distance,v.repair_distance)
            assert ctl.previous==previous
            index=tuple(Move).index(move)
            actions.update(bytes([index]))
            steps+=1
            print(f"{index}{flag}",flush=True)
    assert ctl is None


if __name__=="__main__":
    main(sys.argv[1])
