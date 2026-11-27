"""Фиксация последующих моделей с их собственным адаптером и точным повтором dev."""
import argparse
from importlib import import_module
from pathlib import Path
import sys
from ..data import OUT,ROOT,digest,read,rows,save
from ..dev import evaluate
from .bank import HERE

PACKAGES={
    "exit":("alt_adapter.py",("alt_model.py","alternatives.cpp","alternatives.dll")),
    "returns":("returns_adapter.py",("value_model.py",)),
    "terminal":("belief/physical_adapter.py",("belief/physical_model.py","belief/features.py",
         "belief/walks.cpp","belief/walks.dll","alt_model.py","alternatives.cpp","alternatives.dll")),
    "joint":("belief/joint_adapter.py",("belief/joint_model.py","belief/safety_net.py","belief/features.py",
        "belief/walks.cpp","belief/walks.dll","belief/itineraries.py","belief/itineraries.cpp","belief/itineraries.dll",
        "alt_model.py","alternatives.cpp","alternatives.dll")),
}
FIELD_SOURCES=PACKAGES["joint"][1]+("spatial/model.py","spatial/field_model.py","spatial/transfer_model.py",
    "spatial/region.cpp","spatial/region.dll","belief/paths.py","belief/path_samples.cpp","belief/path_samples.dll")
PACKAGES["field"]=("spatial/transfer_adapter.py",FIELD_SOURCES)
PACKAGES["calibration"]=("spatial/correction_adapter.py",FIELD_SOURCES+("spatial/correction_model.py",
    "spatial/prefix.py","spatial/prefix_bank.cpp","spatial/prefix_bank.dll"))
GATE_SOURCES=("spatial/gate_model.py","spatial/gates.py","spatial/gates.cpp","spatial/gates.dll")
PACKAGES["rooms"]=("spatial/rooms/adapter.py",FIELD_SOURCES+GATE_SOURCES+
                   ("spatial/rooms/model.py","spatial/rooms/cells.py"))
PACKAGES["reconstruct"]=("spatial/reconstruct/adapter.py",PACKAGES["joint"][1]+(
    "spatial/reconstruct/model.py","spatial/reconstruct/features.py","spatial/local_model.py","spatial/model.py",
    "spatial/region.cpp","spatial/region.dll","../prev-two/adapter.py","../prev-two/early_policy.py"))
PACKAGES["reconstruct-dagger"]=("spatial/reconstruct/dagger_adapter.py",PACKAGES["reconstruct"][1]+
                                ("spatial/reconstruct/dagger_model.py",))
PACKAGES["gates"]=("spatial/gate_adapter.py",FIELD_SOURCES+GATE_SOURCES+
                  ("spatial/gate_agent.py","spatial/suffix.py"))
PACKAGES["local"]=("spatial/local_adapter.py",PACKAGES["joint"][1]+("spatial/local_agent.py","spatial/local_model.py",
    "spatial/model.py","spatial/region.cpp","spatial/region.dll","spatial/prefix.py","spatial/prefix_bank.dll",
    "spatial/prefix_bank.cpp","belief/paths.py","belief/path_samples.cpp","belief/path_samples.dll"))
PACKAGES["paired"]=("spatial/terminal_adapter.py",FIELD_SOURCES+GATE_SOURCES+
                   ("spatial/terminal_model.py","spatial/local_model.py"))


def main(path,kind):
    adapter,deps=PACKAGES[kind]
    path=path.resolve()
    pack=path.read_bytes()
    dev=OUT/"data/dev.jsonl"
    prior=read(path.with_suffix(".physical.json"))
    assert prior["weights_sha256"]==digest(path) and prior["dev_sha256"]==digest(dev)
    mod=import_module("results.E-0016.sequence."+adapter[:-3].replace("/","."))
    summary,episodes=evaluate(mod.load(pack),rows(dev))
    assert summary==prior["summary"] and episodes==prior["episodes"]
    verifier=ROOT/"results/verification-rl80"
    sys.path.insert(0,str(verifier))
    from evaluate import episode
    from suite import inventory
    cases,_=inventory([dev],"validation")
    isolated=episode(cases[0],(HERE/adapter).read_bytes(),pack)
    assert all(isolated[k]==episodes[0][k] for k in ("outcome","steps","refuels","repairs"))
    shared=("policy.py","critical_policy.py","rich_policy.py","geometry.dll","geometry.cpp","dev.py")
    sources=[OUT/n for n in shared]+[HERE/n for n in ("model.py","geometry.cpp","sequence.dll",adapter,*deps)]
    record=dict(weights=path.relative_to(OUT).as_posix(),weights_sha256=digest(path),
                adapter=(HERE/adapter).relative_to(OUT).as_posix(),source_sha256={p.relative_to(OUT).as_posix():digest(p) for p in sources},
                dev_sha256=digest(dev),summary=summary,exact_repeats=len(episodes),isolated=isolated,
                worker_sha256=digest(verifier/"policy_worker.py"))
    save(path.with_suffix(".published.json"),record)
    print(record["weights_sha256"],summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    p.add_argument("kind",choices=PACKAGES)
    args=p.parse_args()
    main(args.path,args.kind)
