"""Выбор единственного RL-чекпойнта по физической dev. Без чтения прежних сотен."""
from .data import OUT,digest,read,rows,save
from .dev import evaluate
from .frozen_adapter import load


def main():
    # Более ранняя запись выигрывает при равных двух компонентах правила.
    names=("pilot200/rl","recovery800/rl","critical800/rl","rich800/rl1","rich800/rl2")
    candidates=[]
    for name in names:
        path=OUT/"runs"/(name+".pt")
        row=read(path.with_suffix(".physical.json"))
        assert row["weights_sha256"]==digest(path)
        summary=row["summary"]
        candidates.append(dict(name=name,sha256=row["weights_sha256"],wins=summary["wins"],
                               min_tier=min(summary["by_tier"].values())))
    chosen=max(candidates,key=lambda r:(r["wins"],r["min_tier"]))
    path=OUT/"runs"/(chosen["name"]+".pt")
    prior=read(path.with_suffix(".physical.json"))
    controller=load(path.read_bytes())
    summary,episodes=evaluate(controller,rows(OUT/"data/dev.jsonl"))
    assert summary==prior["summary"] and episodes==prior["episodes"]
    record=dict(rule="max(dev SUCCESS, min(H1,H2,H3,H4), earliest)",candidates=candidates,
                weights=path.relative_to(OUT).as_posix(),weights_sha256=digest(path),
                adapter="frozen_adapter.py",dev_sha256=digest(OUT/"data/dev.jsonl"),physical_dev=summary,
                repeated_identical_episodes=len(episodes),confirmation_trigger=summary["wins"]>=85 and chosen["min_tier"]>=18,
                source_sha256={n:digest(OUT/n) for n in ("adapter.py","frozen_adapter.py","policy.py","critical_policy.py",
                    "rich_policy.py","geometry.cpp","geometry.dll","dev.py","freeze.py")})
    final=OUT/"frozen.json"
    if final.exists():
        assert read(final)==record
    else:
        save(final,record)
    print(record["weights"],record["weights_sha256"],summary,flush=True)


if __name__=="__main__":
    main()
