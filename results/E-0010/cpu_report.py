"""Отчёт по завершённым замерам исполнителей с подтверждённой квотой 2%."""
import json
from pathlib import Path


def report(path):
    cutoff = path.stat().st_size
    result = dict(samples=0, maximum=0, maximum_with_worker=0, over_80_with_worker=0,
                  over_80_without_worker=0, worst_overlap=None, last=None)
    quota = None
    with path.open("rb") as stream:
        while stream.tell() < cutoff:
            line = stream.readline()
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row.get("event") == "start":
                quota = row["cpu_quota"]
            if quota != 2 or "worker_cpu" not in row:
                continue
            total = row["total"]
            result["samples"] += 1
            result["last"] = row["time"]
            result["maximum"] = max(result["maximum"], total)
            if row["worker_cpu"] > 0:
                if total > result["maximum_with_worker"]:
                    result["maximum_with_worker"] = total
                    result["worst_overlap"] = row
                result["over_80_with_worker"] += total > 80
            else:
                result["over_80_without_worker"] += total > 80
    return result


if __name__ == "__main__":
    directory = Path(__file__).with_name("cpu")
    sources = {p.name: r for p in sorted(directory.glob("*.jsonl")) if (r := report(p))["samples"]}
    path = directory.parent / "cpu-summary.json"
    path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf8")
    print("Замеры ЦП", {p: (r["samples"], r["maximum_with_worker"], r["over_80_with_worker"])
                        for p, r in sources.items()}, flush=True)
