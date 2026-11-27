"""Публикация этапа через повтор dev и передачу весов отдельному оценочному процессу."""
import argparse
from pathlib import Path
import sys
from ..data import OUT, ROOT, digest, read, rows, save
from ..dev import evaluate
from .adapter import load
from .bank import HERE


def main(path):
    dev=OUT/"data/dev.jsonl"
    ref=read(path.with_suffix(".physical.json"))
    signature=digest(path)
    assert ref["weights_sha256"]==signature and ref["dev_sha256"]==digest(dev)
    summary,episodes=evaluate(load(path.read_bytes()),rows(dev))
    assert (summary,episodes)==(ref["summary"],ref["episodes"])
    verifier=ROOT/"results/verification-rl80"
    sys.path.insert(0,str(verifier))
    from evaluate import episode
    from suite import inventory
    cases,_=inventory([dev],"validation")
    isolated=episode(cases[0],(HERE/"adapter.py").read_bytes(),path.read_bytes())
    assert all(isolated[k]==episodes[0][k] for k in ("outcome","steps","refuels","repairs"))
    record=dict(weights=path.resolve().relative_to(OUT).as_posix(),weights_sha256=signature,dev_sha256=digest(dev),
                summary=summary,exact_repeats=len(episodes),isolated=isolated,
                adapter="sequence/adapter.py",worker_sha256=digest(verifier/"policy_worker.py"),
                source_sha256={n:digest(OUT/n) for n in ("sequence/adapter.py","sequence/model.py","sequence/guard_model.py",
                    "sequence/sequence.dll","sequence/geometry.cpp","policy.py","rich_policy.py","critical_policy.py",
                    "geometry.dll","geometry.cpp","dev.py")})
    save(path.with_suffix(".published.json"),record)
    print(record["weights"],signature,summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("weights",type=Path)
    main(p.parse_args().weights)
