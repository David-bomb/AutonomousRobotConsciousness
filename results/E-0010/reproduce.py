"""Повтор физического запуска после изменений в общей части дерева контроллеров."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

from bench import AGENTS,HERE,cases,run_case
from compare import read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("agent",choices=AGENTS)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    previous=read(args.agent)
    folder=HERE/"confirm"
    folder.mkdir(exist_ok=True)
    path=folder/f"{args.agent}.jsonl"
    done=set()
    if args.resume and path.exists():
        for line in path.read_bytes().splitlines():
            row=json.loads(line)
            assert row["map"] not in done and row==previous[row["map"]]
            done.add(row["map"])
    def measured():
        jobs=((args.agent,c) for c in cases() if c[1] not in done)
        if args.workers==1:
            yield from map(run_case,jobs)
            return
        with ProcessPoolExecutor(args.workers) as pool:
            yield from pool.map(run_case,jobs)
    with path.open("a" if args.resume else "x",encoding="utf8") as out:
        for row in measured():
            name=row["map"]
            if row!=previous[name]:
                raise ValueError(f"Повтор расходится на {name}")
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
            out.flush()
    print(args.agent,len(previous),"точных повторов, включая каждую команду и счётчики режимов")


if __name__=="__main__":
    main()
