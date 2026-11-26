import argparse
from functools import partial

from .automaton import Automaton, ResourceAutomaton
from .datasets import load_split
from .environment import Environment, Status
from .frontier_reserve import ReserveFrontier
from .levels import catalogue
from .runner import run
from .sight_automaton import SightDetour, SightRoute


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка навигационного автомата")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test", help="набор карт")
    parser.add_argument("--level", help="идентификатор уровня в выбранном наборе")
    frontier = {"frontier-reserve": ReserveFrontier, "sight-route": SightRoute, "sight-detour": SightDetour}
    parser.add_argument("--controller", choices=("pilot", "resources", *frontier, "range-roads", "streets",
                                                "many-streets", "street-trunk", "street-trunk-reserve",
                                                "mission-order-reserve", "simple-exit-paths", "mission-simple-paths",
                                                "station-simple-paths", "specific-paths", "second-window",
                                                "return-windows", "unseen-return", "ranked-standby"), default="pilot",
                        help="выбор автомата навигации")
    parser.add_argument("--visual", action="store_true", help="открыть окно выбора уровня")
    parser.add_argument("--service-offset", type=int, default=0, metavar="N",
                        help="прибавить N единиц к порогу заправки и ремонта")
    args = parser.parse_args()
    if args.controller == "range-roads":
        from .road_hypotheses import RangeRoads
        frontier["range-roads"] = RangeRoads
    elif args.controller in ("streets", "many-streets"):
        from .pattern_roads import ManyStreets, PatternRoads
        frontier.update({"streets": PatternRoads, "many-streets": ManyStreets})
    elif args.controller in ("street-trunk", "street-trunk-reserve"):
        from .street_trunk import StreetTrunk, StreetTrunkReserve
        frontier.update({"street-trunk": StreetTrunk, "street-trunk-reserve": StreetTrunkReserve})
    elif args.controller == "mission-order-reserve":
        from .mission_order import MissionOrderReserve
        frontier[args.controller] = MissionOrderReserve
    elif args.controller in ("simple-exit-paths", "mission-simple-paths"):
        from .simple_exit_paths import MissionSimplePaths, SimpleExitPaths
        frontier.update({"simple-exit-paths": SimpleExitPaths, "mission-simple-paths": MissionSimplePaths})
    elif args.controller == "station-simple-paths":
        from .nearest_station import StationSimplePaths
        frontier[args.controller] = StationSimplePaths
    elif args.controller == "specific-paths":
        from .specific_paths import SpecificPaths
        frontier[args.controller] = SpecificPaths
    elif args.controller in ("second-window", "return-windows"):
        from .second_window import ReturnWindows, SecondWindow
        frontier.update({"second-window": SecondWindow, "return-windows": ReturnWindows})
    elif args.controller == "unseen-return":
        from .unseen_approach import UnseenReturn
        frontier[args.controller] = UnseenReturn
    elif args.controller == "ranked-standby":
        from .standby_turn import RankedStandby
        frontier[args.controller] = RankedStandby
    if args.service_offset < 0:
        parser.error("--service-offset должен быть неотрицательным")
    levels = catalogue() if args.split == "test" else {c.name: c.level for c in load_split(args.split)}
    if args.level and args.level not in levels:
        parser.error(f"Уровень {args.level} отсутствует в {args.split}")
    if args.controller in frontier:
        if args.service_offset:
            parser.error(f"Для {args.controller} закреплены исходные пороги")
        controller = frontier[args.controller]
    else:
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
