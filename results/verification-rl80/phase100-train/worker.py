"""Отдельный процесс нейросети. На входе два начальных регистра wave1 и затем только Observation."""
from importlib import import_module, util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from maze.interface import Move, Observation
import torch


def main(mode):
    torch.set_num_threads(1)
    source = sys.stdin.buffer
    if mode == "wave":
        # Относительные импорты исторического адаптера привязаны к тем же
        # замороженным файлам policy и sequence.model, что и у frontier48.
        spec = util.spec_from_file_location("results.verification-rl80.frontier48-dev.frozen.prev.adapter",
                                            HERE/"wave1_adapter.py")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctl = module.from_bytes((HERE/"wave1.pt").read_bytes())
        pair = json.loads(source.readline())
        assert len(pair)==2 and all(v is None or type(v) is int for v in pair)
        ctl.previous=tuple(pair)
    elif mode == "frontier":
        package = import_module("results.verification-rl80.frontier48-dev.bridge")
        ctl = package.load((HERE.parent/"frontier48-dev/frozen/model.pt").read_bytes())
    else:
        raise ValueError(mode)
    for line in source:
        obs = json.loads(line)
        for key in ("memory","window","position"):
            obs[key] = tuple(obs[key])
        if obs["last_move"] is not None:
            obs["last_move"] = Move[obs["last_move"]]
        view = Observation(**obs)
        move = ctl.choose(view)
        if mode=="wave":
            assert ctl.previous==(view.fuel_distance,view.repair_distance)
        print(move.name,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
