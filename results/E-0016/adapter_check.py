"""Контроль передачи байтов весов независимому процессу на одной карте нового dev."""
import sys
from .data import OUT, ROOT, digest, read, save


def main():
    verifier=ROOT/"results/verification-rl80"
    sys.path.insert(0,str(verifier))
    from evaluate import episode
    from suite import inventory
    locked=read(OUT/"frozen.json")
    file=OUT/locked["weights"]
    assert digest(file)==locked["weights_sha256"]
    cases,_=inventory([OUT/"data/dev.jsonl"],"validation")
    case=cases[0]
    trial=episode(case,(OUT/locked["adapter"]).read_bytes(),file.read_bytes())
    reference=read(file.with_suffix(".physical.json"))["episodes"][0]
    assert all(trial[k]==reference[k] for k in ("outcome","steps","repairs","refuels"))
    save(OUT/"adapter-check.json",dict(weights_sha256=digest(file),adapter_sha256=digest(OUT/locked["adapter"]),
         independent_worker_sha256=digest(verifier/"policy_worker.py"),episode=trial))
    print(trial)


if __name__=="__main__":
    main()
