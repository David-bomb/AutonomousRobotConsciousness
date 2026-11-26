"""Физические поездки нескольких контроллеров на одной карте в одном процессе."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from zipfile import ZipFile

from bench import AGENTS, HERE, ROOT, SETS, cases, run_group
from maze.environment import Status


def check_maps():
    with ZipFile(HERE/"snapshots/shared-passages.zip") as archive:
        expected = json.loads(archive.read("manifest.json"))["maps"]
    for path in SETS.values():
        actual = sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected[path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Изменился набор {path}")


def recorded(agent, valid):
    path = HERE/f"{agent}.jsonl"
    saved = {}
    if path.exists():
        with path.open(encoding="utf8") as stream:
            for line in stream:
                row = json.loads(line)
                key = tuple(row[k] for k in ("split","map","tier","family"))
                if (key not in valid or key in saved or len(row["actions"]) != row["steps"]
                        or row["status"] not in Status.__members__ or row["status"] == "RUNNING"):
                    raise ValueError(f"Некорректная запись {path}, {key}")
                saved[key] = row
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents", nargs="+", choices=AGENTS)
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--verify", nargs="+", metavar="MAP", help="сверить готовые записи на указанных картах")
    maps.add_argument("--maps", nargs="+", metavar="MAP", help="дописать самостоятельные поездки на выбранных картах")
    args = parser.parse_args()
    if len(set(args.agents)) != len(args.agents):
        parser.error("Укажите каждого робота один раз")
    check_maps()
    selected = list(cases())
    valid = {case[:4] for case in selected}
    if len(valid) != len(selected) or len(valid) != 336:
        raise ValueError("Ожидалось 336 различных карт")
    previous = {agent: recorded(agent, valid) for agent in args.agents}
    subset = args.verify or args.maps
    if subset:
        selected = [c for c in selected if c[1] in subset]
        if {c[1] for c in selected} != set(subset):
            parser.error("Неизвестная карта")
    completed = dict.fromkeys(args.agents, 0)
    with ExitStack() as stack:
        streams = {} if args.verify else {
            agent: stack.enter_context((HERE/f"{agent}.jsonl").open("a",encoding="utf8",newline="\n"))
            for agent in args.agents}
        for case in selected:
            key = case[:4]
            missing = args.agents if args.verify else [agent for agent in args.agents if key not in previous[agent]]
            if not missing:
                continue
            for agent,row in run_group(missing,case):
                if args.verify:
                    if row != previous[agent].get(key):
                        raise ValueError(f"Расхождение независимой поездки {agent} на {case[1]}")
                else:
                    streams[agent].write(json.dumps(row,ensure_ascii=False)+"\n")
                    streams[agent].flush()
                completed[agent] += 1
    print("Проверено" if args.verify else "Дописано", completed, flush=True)


if __name__ == "__main__":
    main()
