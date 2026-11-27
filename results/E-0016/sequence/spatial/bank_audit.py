"""Проверка регистрации геометрий до вырезания обучающих состояний."""
import json
import re
from hashlib import sha256
from maze.datasets import geometry
from ...data import OUT,digest,read,save
from .fresh import HERE,fingerprints


def main():
    used=fingerprints()
    archive_digest=sha256("\n".join(sorted(used)).encode()).hexdigest()
    count_before=len(used)
    hashes=re.compile(rb'"geometry"\s*:\s*"([0-9a-f]{64})"')
    groups=[]
    first=read(OUT/"data/manifest.json")
    extra=read(HERE/"fresh.json")["sets"]
    reward=read(HERE/"reward-bank.json")["groups"]
    files=[
        (OUT/"data/train.jsonl","первичное обучение",False,None),
        (OUT/"data/dev.jsonl","открытая dev",True,None),
        (OUT/"data/confirmation.jsonl","две сотни независимого оценщика",True,None),
        (HERE.parent/"train.jsonl","обучение второй волны",False,None),
        (HERE.parent/"probe.jsonl","вспомогательная проверка датчиков",False,None),
        (HERE/"extra.jsonl","ранние состояния, 3000 fit и 200 внутренних геометрий",False,extra["extra"]["sha256"]),
        (HERE/"selection.jsonl","две сотни отбора ранних политик по финишам",False,extra["selection"]["sha256"]),
        (HERE/"large.jsonl","12000 для предварительного обучения, включая 200 геометрий парных продолжений",False,read(HERE/"large.json")["sha256"]),
        *((HERE/f"reward-{name}.jsonl",role,False,reward[name]["sha256"]) for name,role in (
            ("screen","предварительный отбор LocalNet"),("selection","200 геометрий отбора LocalNet по финишам"),
            ("probe","100 геометрий проверки терминальных продолжений"))),
        (HERE/"gate-selection.jsonl","200 полных стартов отбора политики с выходами из известной дороги",False,read(HERE/"gate-selection.json")["sha256"])]
    for path,role,opaque,expected in files:
        sig=digest(path)
        if expected:assert sig==expected
        group=set()
        with path.open("rb") as f:
            for line in f:
                key=hashes.search(line).group(1).decode()
                if not opaque:
                    assert geometry(json.loads(line)["grid"])==key
                assert key not in group and key not in used
                group.add(key)
        groups.append(dict(path=path.relative_to(OUT).as_posix(),role=role,maps=len(group),sha256=sig,
                           geometries_sha256=sha256("\n".join(sorted(group)).encode()).hexdigest(),
                           overlap_with_previous=0,access="geometry field only" if opaque else "canonical walls recomputed"))
        used.update(group)
    assert count_before==first["excluded_count"] and archive_digest==first["excluded_sha256"]
    save(HERE/"bank-audit.json",dict(previous_excluded=count_before,archive_geometries_sha256=archive_digest,
         banks=groups,registered=sum(r["maps"] for r in groups),combined_count=len(used),
         canonicalization="walls without labels, four rotations and reflections",script=digest(HERE/"bank_audit.py")))
    print(count_before,[g["maps"] for g in groups],len(used),flush=True)


if __name__=="__main__":main()
