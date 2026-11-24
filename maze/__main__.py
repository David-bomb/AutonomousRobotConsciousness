import argparse

from .automaton import Automaton
from .environment import Environment, Status
from .levels import catalogue
from .runner import run


def main() -> int:
    levels = catalogue()
    parser = argparse.ArgumentParser(description="Проверка навигационного автомата")
    parser.add_argument("--level", choices=levels, help="уровень для запуска")
    parser.add_argument("--visual", action="store_true", help="открыть окно выбора уровня")
    args = parser.parse_args()
    if args.visual:
        from .viewer import show
        show(levels, args.level, Automaton)
        return 0
    success = True
    for name, level in levels.items():
        if args.level and args.level != name:
            continue
        env = Environment(level)
        status = run(env, Automaton())
        print(f"{name} | {status.value} | ходов {env.steps} | заправок {env.refuels} | ремонтов {env.repairs}")
        success = success and status == Status.SUCCESS
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
