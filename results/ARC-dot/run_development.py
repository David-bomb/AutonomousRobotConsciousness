"""Записать результаты на наборах разработки в формате проверки compare_replay.py."""
import argparse
import csv
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stand", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("split", choices=("e5", "train", "val", "test", "holdout", "reserve-fresh"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.stand.resolve()))
    from scripts.audit_automaton import measure

    with args.output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if args.split == "e5":
            from maze.frontier_reserve import ReserveFrontier
            from scripts.evaluate_frontier import groups
            writer.writerow(("split","case","status","steps"))
            for group, cases in groups({"train","val","archive","confirm","holdout"}):
                for case in cases:
                    result = measure(case, ReserveFrontier)
                    writer.writerow((group,case.name,result["outcome"],result["steps"]))
            return
        from scripts.iterate_frontier import AGENTS, selected_cases
        writer.writerow(("case","agent","status","steps"))
        for case in selected_cases(args.split):
            for agent in "ABCDEF":
                result = measure(case, AGENTS[agent])
                writer.writerow((case.name,agent,result["outcome"],result["steps"]))


if __name__ == "__main__":
    main()
