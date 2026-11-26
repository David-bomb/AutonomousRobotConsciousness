"""Был ли общий отрезок двух истинных кратчайших дорог в точке изменения решения."""
from collections import Counter
from functools import lru_cache
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.grid import destination


def analyze(env):
    fields=[env._lengths[k] for k in "FR"]
    def downhill(p):
        options=[]
        for field in fields:
            options.append({m:q for m in MOVES if (q:=destination(env.level.grid,p,m)) is not None
                            and p in field and field[q]==field[p]-1})
        return options
    @lru_cache(maxsize=None)
    def shared(p):
        f,r=downhill(p)
        return max((1+shared(f[m]) for m in f.keys()&r.keys()),default=0)
    f,r=downhill(env.position)
    depths={m.name:1+shared(f[m]) for m in f.keys()&r.keys()}
    return dict(F=sorted(m.name for m in f),R=sorted(m.name for m in r),shared=depths)


def main():
    filename="marked-streets--street-trunk.jsonl"
    records={r["map"]:r for z in (HERE/filename).read_bytes().splitlines() if (r:=json.loads(z)).get("first")}
    runs=read("street-trunk")
    counts=Counter()
    with (HERE/"shared-evidence.jsonl").open("x",encoding="utf8") as out:
        for split,name,tier,family,level in cases():
            if name not in records:
                continue
            row=records[name]
            first=row["first"]
            env=Environment(level)
            for char in runs[name]["actions"][:first["step"]-1]:
                env.step(MOVES[int(char)])
            data=analyze(env)
            a,b=first["parent"],first["child"]
            counts[row["verdict"],bool(data["shared"]),a in data["shared"],b in data["shared"]]+=1
            out.write(json.dumps(dict(map=name,split=split,verdict=row["verdict"],first=first,**data),
                                 ensure_ascii=False)+"\n")
    print(counts)


if __name__=="__main__":
    main()
