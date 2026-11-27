"""Состав трёх сотен, контроль среды и допуск отложенной выборки."""
import argparse
from dataclasses import dataclass
import gzip
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from maze.datasets import geometry, load_split
from maze.interface import Move
from maze.levels import Level

SPLITS = ("I", "II", "III")
SOURCES = ("results/ARC-dot/source/ARC-stress-100/maps.jsonl",
           "results/E-0007/fresh.jsonl", "results/E-0009/fresh.jsonl")
ENV_FILES = ("environment", "memory", "interface", "grid", "levels", "datasets")
ARCHIVE = HERE / "stress-spec.json"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(data):
    return sha256(data).hexdigest()


def object_hash(data):
    encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return digest(encoded)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def lf_hash(path):
    return digest(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def path_name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


@dataclass(frozen=True)
class Case:
    split: str
    name: str
    geometry: str
    scenario: str
    level: Level


def hundreds(paths, kind):
    if kind == "historical":
        check(len(paths) == 3, "Нужны три архивных набора")
    group = 0
    for path in paths:
        raw = (ROOT / path).read_bytes()
        lines = raw.splitlines()
        check(lines and len(lines) % 100 == 0, "Нужны полные сотни")
        if kind == "historical":
            check(len(lines) == 100, "В архивном файле должно быть 100 записей")
        for start in range(0, len(lines), 100):
            split = SPLITS[group] if kind == "historical" else f"{kind}-{group+1}"
            yield split, path, digest(raw.replace(b"\r\n", b"\n")), lines[start:start+100]
            group += 1


def inventory(paths, kind="historical"):
    seen_ids, seen_geometries = set(), set()
    cases, sets = [], {}
    for split, path, file_sha, lines in hundreds(paths, kind):
        shapes, stocks, scenarios = [], [], []
        for line in lines:
            r = json.loads(line)
            name, grid, fuel, health = (r[k] for k in ("name", "grid", "fuel", "health"))
            check(type(name) is str and name.strip() == name and len(name.splitlines()) == 1
                  and name not in seen_ids, "Пустой или повторяющийся ID")
            check(type(fuel) is int and type(health) is int, "Запасы должны быть целыми")
            level = Level.parse("\n".join([name, f"{fuel} {health}", *grid]))
            walls = geometry(level.grid)
            check(walls == r.get("geometry", walls), "Исходный отпечаток стен не совпал")
            check(walls not in seen_geometries, "Повтор геометрии в оцениваемых наборах")
            seen_ids.add(name)
            seen_geometries.add(walls)
            scenario = object_hash([grid, fuel, health])
            cases.append(Case(split, name, walls, scenario, level))
            shapes.append([name, walls])
            stocks.append([name, fuel, health])
            scenarios.append([name, scenario])
        check(len(shapes) == 100, f"Нужно 100 карт в {split}")
        sets[split] = dict(path=path_name(ROOT / path), sha256_lf=file_sha,
                           geometries=object_hash(shapes), stocks=object_hash(stocks),
                           scenarios=object_hash(scenarios), count=len(shapes))
    sources = {f"maze/{name}.py": lf_hash(ROOT / f"maze/{name}.py") for name in ENV_FILES}
    spec = dict(version=1, threshold=None if kind == "validation" else 80, sets=sets, environment_sha256_lf=sources,
                actions=[m.name for m in Move], limit="8*height*width")
    return cases, spec


def archived():
    cases, spec = inventory(SOURCES)
    locked = read(ROOT / "results/E-0009/frozen.json")["sha256_lf"]
    earlier = read(ROOT / "results/E-0013/manifest.json")["sets_sha256_lf"]
    later = read(ROOT / "results/E-0015/registration.json")
    witnesses = read(ROOT / "results/E-0013/witnesses.json")
    check({c.name for c in cases} == set(witnesses), "Неполный реестр E-0013")
    for c in cases:
        check((c.split, c.geometry) == (witnesses[c.name]["split"], witnesses[c.name]["geometry"]),
              "Геометрия или сотня расходится с E-0013")
    for split, s in spec["sets"].items():
        check(s["sha256_lf"] == locked[s["path"]] == earlier[split] == later["stress_maps"][split],
              f"Расхождение архивных SHA-256 в {split}")
    for path, value in spec["environment_sha256_lf"].items():
        check(value == later["base_sources"][path], f"Изменился {path}")
    return cases, spec


def checked_suite(spec_file):
    spec = read(spec_file)
    kind = spec["kind"]
    paths = SOURCES if kind == "historical" else spec["inputs"]
    cases, rebuilt = inventory(paths, kind)
    check(spec["contract"] == rebuilt, "Изменились карты, запасы, параметры или код среды")
    if spec["kind"] == "historical":
        check(spec == read(ARCHIVE) and tuple(paths) == SOURCES, "Подмена исторических сотен")
        earlier = read(ROOT / "results/E-0013/manifest.json")["sets_sha256_lf"]
        later = read(ROOT / "results/E-0015/registration.json")
        for split in SPLITS:
            check(rebuilt["sets"][split]["sha256_lf"] == earlier[split] == later["stress_maps"][split],
                  "Расхождение с замороженными манифестами")
    elif kind == "validation":
        # Допуск геометрий выполняет transfer() до фиксации спецификации.
        # В поездке dev читаем только её карты и закреплённое свидетельство.
        evidence = spec["exclusion"]
        check(evidence["common"] == 0 and evidence["historical_count"] > 0
              and evidence["additional"], "Нет регистрации проверки пересечений dev")
    elif kind == "transfer":
        files = [ROOT / p for p in spec["exclusion"]["additional"]]
        check(files and spec["exclusion"] == exclusion(cases, files), "Реестр пересечений изменился")
    else:
        raise ValueError("Неизвестное назначение выборки")
    versions = read(ROOT / "results/E-0015/registration.json")["base_sources"]
    check(all(versions[p] == h for p, h in rebuilt["environment_sha256_lf"].items()),
          "Код симулятора расходится с E-0015")
    return cases


def previous_geometries():
    """Отпечатки прежних наборов. База открывается только для чтения."""
    shapes = {c.geometry for c in archived()[0]}
    shapes.update(read(ROOT / "results/ARC-dot/source/ARC-stress-100/excluded.json"))
    shapes.update(read(ROOT / "results/ARC-dot/replay/e6-fresh/diagnosis.json")["geometries"].values())
    shapes.update(geometry(c.level.grid) for s in ("train", "val") for c in load_split(s))
    shapes.update(geometry(Level.parse(p.read_text(encoding="utf-8")).grid)
                  for p in (ROOT / "maze/levels").glob("*.txt"))
    old = ROOT / "results/E-0012"
    with sqlite3.connect((old / "corpus.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        shapes.update(r[0] for r in db.execute("SELECT hash FROM geoms"))
    for s in ("train", "control"):
        with gzip.open(old / f"diagnostics/{s}.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                actual = geometry(Level.parse(r["source"]).grid)
                check(actual == r["geometry"], "Отпечаток диагностической карты не совпал")
                shapes.add(actual)
    return shapes


def exclusion(cases, exposure_files):
    historical = previous_geometries()
    seen = set(historical)
    evidence = {}
    for path in exposure_files:
        if Path(path).suffix == ".jsonl":
            records = [json.loads(line) for line in Path(path).read_bytes().splitlines()]
            values = [geometry(r["grid"]) for r in records]
            check(all(r.get("geometry", h) == h for r, h in zip(records, values)),
                  "Ошибка отпечатка в реестре использованных карт")
        else:
            values = Path(path).read_text(encoding="ascii").splitlines()
        check(values and all(len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in values),
              "Нужен непустой реестр канонических SHA-256, по одному на строку")
        seen.update(values)
        evidence[path_name(path)] = dict(count=len(values), sha256_lf=lf_hash(path))
    check(seen.isdisjoint(c.geometry for c in cases), "Отложенные геометрии пересекаются с известными")
    return dict(historical_count=len(historical), historical_sha256=object_hash(sorted(historical)),
                additional=evidence, common=0)


def transfer(paths, exposure_files, kind="transfer"):
    cases, spec = inventory(paths, kind)
    return dict(kind=kind, inputs=[path_name(p) for p in paths], contract=spec,
                exclusion=exclusion(cases, exposure_files))


def check_e16():
    folder = ROOT / "results/E-0016/data"
    registry = read(folder / "manifest.json")
    used = previous_geometries()
    check(len(used) == registry["excluded_count"] and
          digest("\n".join(sorted(used)).encode()) == registry["excluded_sha256"],
          "Расходится исходный реестр исключений E-0016")
    for name in ("train", "dev", "confirmation"):
        path = folder / (name + ".jsonl")
        expected = registry["splits"][name]
        check(file_hash(path) == expected["sha256"], "Файл E-0016 отличается от регистрации")
        cases, _ = inventory([path], name)
        keys = {c.geometry for c in cases}
        check(len(cases) == len(keys) == expected["count"] and keys.isdisjoint(used),
              "Неполная выборка или пересечение геометрий E-0016")
        used.update(keys)
    for filename in ("dev-spec.json", "confirmation-spec.json"):
        checked_suite(HERE / filename)
    return registry["splits"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    archive = subs.add_parser("archive")
    archive.add_argument("--write", type=Path)
    subs.add_parser("e16")
    for mode in ("transfer", "validation"):
        hold = subs.add_parser(mode)
        hold.add_argument("--maps", nargs="+", required=True, type=Path)
        hold.add_argument("--exposure", nargs="+", required=True, type=Path)
        hold.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "archive":
        _, contract = archived()
        result = dict(kind="historical", contract=contract)
        if args.write:
            write_new(args.write, result)
        else:
            check(read(ARCHIVE) == result, "Спецификация не совпала с архивом")
    elif args.command == "e16":
        print(json.dumps(check_e16(), indent=2))
        sys.exit()
    else:
        result = transfer(args.maps, args.exposure, args.command)
        write_new(args.out, result)
    print("SHA-256 спецификации (канонический JSON)", object_hash(result))
