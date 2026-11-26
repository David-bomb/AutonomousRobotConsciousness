"""Повтор конкретного эпизода рядом с решением, которое нужно разобрать."""
import argparse

from run import FILES, rows, Level, Environment, ReserveFrontier, SightDetour, SightRoute, Status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", choices=FILES)
    parser.add_argument("map")
    agents = {"reserve": ReserveFrontier, "sight-route": SightRoute, "sight-detour": SightDetour}
    parser.add_argument("--agent", choices=agents, default="sight-route")
    parser.add_argument("--step", type=int, default=1, help="первый отображаемый ход, с 1")
    parser.add_argument("--count", type=int, default=5, help="число строк в текстовом режиме")
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()
    if args.step <= 0 or args.count <= 0:
        parser.error("Номер хода и число строк должны быть положительными")
    item = next((r for r in rows(FILES[args.series]) if r["name"] == args.map), None)
    if item is None:
        parser.error("Карты с таким именем в этой серии нет")
    lines = item["lines"] if args.series == "val" else [
        item["name"], f'{item["fuel"]} {item["health"]}', *item["grid"]]
    level = Level.parse("\n".join(lines))
    factory = agents[args.agent]
    env, bot = Environment(level), factory()
    while env.status == Status.RUNNING and env.steps + 1 < args.step:
        env.step(bot.choose(env.view))
    if env.status != Status.RUNNING:
        parser.error(f"Эпизод закончился на ходу {env.steps}")
    if args.visual:
        import tkinter as tk
        from maze.viewer import Viewer
        root = tk.Tk()
        viewer = Viewer(root, {args.map: level}, args.map, factory)
        viewer.env, viewer.bot = env, bot
        root.title(f"{args.map} · {args.agent}")
        viewer.draw()
        root.mainloop()
        return
    for _ in range(args.count):
        v = env.view
        move = bot.choose(v)
        print(f"{env.steps+1:4}  {v.position}  F={v.fuel}/{v.fuel_distance}  "
              f"R={v.health}/{v.repair_distance}  {move.name:5}  {bot.activity}  {'/'.join(v.window)}")
        env.step(move)
        if env.status != Status.RUNNING:
            print(f"{env.status.value}, ход {env.steps}")
            break


if __name__ == "__main__":
    main()
