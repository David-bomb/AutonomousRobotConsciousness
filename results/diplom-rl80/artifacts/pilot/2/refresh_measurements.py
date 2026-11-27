"""Create the lab's dev scorecard from full physical episodes and frozen checkpoint hashes."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_bytes())


def link(path):
    return quote(os.path.relpath(path, HERE).replace(os.sep, "/"), safe="/.-_")


def generate(project):
    exp = project / "results/E-0016"
    dev = exp / "data/dev.jsonl"
    raw = dev.read_bytes()
    cases = {r["name"]:r["tier"] for r in map(json.loads, raw.splitlines())}
    tiers = ["H1", "H2", "H3", "H4"]
    if Counter(cases.values()) != Counter(dict.fromkeys(tiers, 25)):
        raise ValueError("Unexpected dev composition")
    fingerprint = sha256(raw).hexdigest()
    lines = ["# Полные валидационные поездки E-0016", "",
             "Каждая строка — одни закреплённые веса на одних и тех же 100 новых картах dev. "
             "Во всех эпизодах модель использует обычное наблюдение, стартует с исходными запасами и "
             "движется до штатного исхода. H1–H4 содержат по 25 карт. Выбор весов по этому набору "
             "делает его выборкой разработки. Оценка на I, II и III ведётся отдельно.", "",
             "| Запуск | Веса | Финиши / 100 | H1 | H2 | H3 | H4 | Посетил оба сервиса | Покартовый журнал |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    archives = sorted((exp/"runs").glob("*/*.physical.json"), key=lambda p:p.stat().st_mtime_ns)
    for source in archives:
        report = load(source)
        tag = source.name.removesuffix(".physical.json")
        weights = source.with_name(tag + ".pt")
        if sha256(weights.read_bytes()).hexdigest() != report["weights_sha256"] or report["dev_sha256"] != fingerprint:
            raise ValueError(f"Checkpoint or dev mismatch: {source}")
        episodes = report["episodes"]
        if len(episodes) != len(cases) or {e["map"] for e in episodes} != set(cases):
            raise ValueError(f"Missing or repeated episodes: {source}")
        wins = Counter()
        both = 0
        for e in episodes:
            if cases[e["map"]]!=e["tier"] or e["outcome"] not in {"SUCCESS","EMPTY","BROKEN","LIMIT"}:
                raise ValueError(f"Unexpected episode: {e}")
            wins[e["tier"]] += e["outcome"]=="SUCCESS"
            both += bool(e["refuels"] and e["repairs"])
        summary = report["summary"]
        if dict(wins)!=summary["by_tier"] or sum(wins.values())!=summary["wins"] or both!=summary["both"]:
            raise ValueError(f"Summary disagrees with episodes: {source}")
        columns = [source.parent.name, f"[{tag}]({link(weights)})", str(sum(wins.values())),
                   *(str(wins[t]) for t in tiers), str(both), f"[JSON]({link(source)})"]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "Корпус dev, SHA-256 · " + fingerprint, "",
              "Таблица пересчитывается командой из корня лаборатории.", "",
              "\u0060\u0060\u0060powershell", "python projects/diplom-rl80/refresh_measurements.py", "\u0060\u0060\u0060", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, default=HERE.parents[3]/"University"/"Diplom")
    args = ap.parse_args()
    text = generate(args.project.resolve())
    dest = HERE/"measurements.md"
    if not dest.exists() or dest.read_text(encoding="utf-8")!=text:
        tmp = dest.with_suffix(".md.tmp")
        tmp.write_text(text,encoding="utf-8")
        tmp.replace(dest)
    print(dest)
