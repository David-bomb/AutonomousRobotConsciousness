"""Заранее закреплённые разные геометрии отбора для малого запаса и для полных последовательностей."""
from hashlib import sha256
import json
import re
from maze.datasets import geometry
from ...data import OUT,generate,read,save,digest
from .fresh import HERE,fingerprints


def main():
    registration=read(HERE/"bank-audit.json")
    keys=fingerprints()
    pattern=re.compile(rb'"geometry"\s*:\s*"([0-9a-f]{64})"')
    for item in registration["banks"]:
        path=OUT/item["path"]
        assert digest(path)==item["sha256"]
        with path.open("rb") as stream:
            for line in stream:keys.add(pattern.search(line).group(1).decode())
    assert len(keys)==registration["combined_count"]
    groups={}
    for name,seeds in (("risk-selection",[9164400,9164401]),("route-selection",[9164410,9164411])):
        dest=HERE/(name+".jsonl")
        before=len(keys)
        with dest.open("x",encoding="utf8",newline="\n") as f:
            for seed in seeds:
                for row in generate((name,seed)):
                    key=geometry(row["grid"])
                    assert row["geometry"]==key and key not in keys
                    keys.add(key)
                    f.write(json.dumps(row,separators=(",",":"))+"\n")
        assert len(keys)-before==200
        groups[name]=dict(maps=200,seeds=seeds,sha256=digest(dest),overlap=0)
    save(HERE/"decisions-bank.json",dict(initial=digest(HERE/"bank-audit.json"),groups=groups,combined=len(keys),
        geometries_sha256=sha256("\n".join(sorted(keys)).encode()).hexdigest()))


if __name__=="__main__":main()
