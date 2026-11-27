"""Дополнительные независимые учебные геометрии и отдельная диагностика датчиков."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from ..data import OUT, digest, excluded, generate, geometry, rows, save

HERE=Path(__file__).resolve().parent
SEEDS={"train":list(range(9161100,9161116)),"probe":[9161200]}


def main():
    used=excluded()
    for split in ("train","dev","confirmation"):
        used.update(r["geometry"] for r in rows(OUT/f"data/{split}.jsonl"))
    reg=dict(seeds=SEEDS,excluded=len(used),excluded_sha256=sha256("\n".join(sorted(used)).encode()).hexdigest(),splits={})
    with ProcessPoolExecutor(3) as workers:
        for split,seeds in SEEDS.items():
            path=HERE/f"{split}.jsonl"
            if path.exists():
                batch=rows(path)
            else:
                batch=[r for part in workers.map(generate,[(f"sequence-{split}",s) for s in seeds]) for r in part]
            assert len(batch)==100*len(seeds)
            for r in batch:
                key=geometry(r["grid"])
                assert key==r["geometry"] and key not in used
                used.add(key)
            if not path.exists():
                with path.open("x",encoding="utf8",newline="\n") as out:
                    for r in batch:
                        out.write(json.dumps(r,separators=(",",":"))+"\n")
            reg["splits"][split]=dict(n=len(batch),sha256=digest(path))
    manifest=HERE/"bank.json"
    if manifest.exists():
        assert json.loads(manifest.read_bytes())==reg
    else:
        save(manifest,reg)
    print(reg["splits"],flush=True)


if __name__=="__main__":
    main()
