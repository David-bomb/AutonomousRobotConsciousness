import argparse
from functools import partial

from .automaton import Automaton, ResourceAutomaton
from .datasets import load_split
from .environment import Environment, Status
from .levels import catalogue
from .runner import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка навигационного автомата")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test", help="набор карт")
    parser.add_argument("--level", help="идентификатор уровня в выбранном наборе")
    parser.add_argument("--controller", choices=("pilot", "resources"), default="pilot",
                        help="исходный автомат D-0008 или ресурсный E-0002")
    parser.add_argument("--visual", action="store_true", help="открыть окно выбора уровня")
    parser.add_argument("--service-offset", type=int, default=0, metavar="N",
                        help="прибавить N единиц к порогу заправки и ремонта")
    args = parser.parse_args()
    if args.service_offset < 0:
        parser.error("--service-offset должен быть неотрицательным")
    levels = catalogue() if args.split == "test" else {c.name: c.level for c in load_split(args.split)}
    if args.level and args.level not in levels:
        parser.error(f"Уровень {args.level} отсутствует в {args.split}")
    controller_type = ResourceAutomaton if args.controller == "resources" else Automaton
    controller = partial(controller_type, service_offset=args.service_offset)
    if args.visual:
        from .viewer import show
        show(levels, args.level, controller)
        return 0
    success = True
    for name, level in levels.items():
        if args.level and args.level != name:
            continue
        env = Environment(level)
        status = run(env, controller())
        print(f"{name} | {status.value} | ходов {env.steps} | заправок {env.refuels} | ремонтов {env.repairs}")
        success = success and status == Status.SUCCESS
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
