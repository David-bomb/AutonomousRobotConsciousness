"""Снимок исходников и контрольных сумм полностью записанных серий."""
from argparse import ArgumentParser
from hashlib import sha256
import json
import platform
from zipfile import ZIP_DEFLATED,ZipFile

import numpy
import scipy

from bench import HERE,ROOT,SETS
from ledger import PARENTS


def digest(data):
    return sha256(data.replace(b"\r\n",b"\n")).hexdigest()


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--include-runs", action="store_true",
                        help="приложить исходные наборы карт и журналы поездок, включая незаконченные серии")
    parser.add_argument("--include-analysis", action="store_true",
                        help="приложить также покартовые сравнения и результаты разбора траекторий")
    args=parser.parse_args()
    if not args.name.replace("-","").isalnum():
        parser.error("Имя снимка должно состоять из букв, цифр и дефиса")
    files=sorted((ROOT/"maze").glob("*.py"))+sorted(HERE.glob("*.py"))
    files.extend([ROOT/"scripts/audit_automaton.py",ROOT/"docs/experiments/E-0010-lineage.md",
                  ROOT/"tests/test_experimental_guards.py"])
    manifest=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
                  maps={str(p.relative_to(ROOT).as_posix()):digest(p.read_bytes()) for p in SETS.values()},
                  sources={},runs={})
    attachments = dict(manifest["maps"]) if args.include_runs else {}
    for key in ("reserve",*PARENTS):
        path=HERE/f"{key}.jsonl"
        if not path.exists():
            continue
        data=path.read_bytes()
        if args.include_runs:
            attachments[path.relative_to(ROOT).as_posix()] = digest(data)
            if len(data.splitlines()) < 336:
                manifest.setdefault("partial_runs", {})[key] = dict(records=len(data.splitlines()), sha256_lf=digest(data))
        if len(data.splitlines())==336:
            cases=[json.loads(l) for l in data.splitlines()]
            assert len({r["map"] for r in cases})==336
            manifest["runs"][key]=dict(parent=PARENTS.get(key),sha256_lf=digest(data),
                wins=[sum(r["split"]==s and r["status"]=="SUCCESS" for r in cases) for s in SETS])
    if args.include_analysis:
        for path in sorted(HERE.iterdir()):
            if path.is_file() and (path.suffix in {".json", ".jsonl", ".partial"} or path.name == "by-map.md"):
                rel = path.relative_to(ROOT).as_posix()
                if rel not in attachments:
                    attachments[rel] = digest(path.read_bytes())
    snapshots=HERE/"snapshots"
    snapshots.mkdir(exist_ok=True)
    with ZipFile(snapshots/f"{args.name}.zip","x",compression=ZIP_DEFLATED) as archive:
        for path in files:
            data=path.read_bytes()
            rel=path.relative_to(ROOT).as_posix()
            archive.writestr(rel,data)
            manifest["sources"][rel]=digest(data)
        if attachments:
            manifest["attachments"] = attachments
            for rel in attachments:
                archive.write(ROOT / rel, rel)
        archive.writestr("manifest.json",json.dumps(manifest,indent=2))
    with ZipFile(snapshots/f"{args.name}.zip") as archive:
        assert archive.testzip() is None
        for path,h in manifest["sources"].items():
            assert digest(archive.read(path))==h
        for path,h in attachments.items():
            assert digest(archive.read(path))==h
    print(args.name,len(files),"исходных файлов,",len(manifest["runs"]),"полных серий")


if __name__=="__main__":
    main()
