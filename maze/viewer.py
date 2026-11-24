"""Просмотр одного эпизода. Таймер и спрайты существуют только в этом модуле."""
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk

from .environment import Environment, Status
from .interface import Controller
from .levels import Level
from .runner import advance

BG = "#101c28"
PANEL = "#1b2c3b"
INK = "#edf4f4"
MUTED = "#acc2cb"
ACCENT = "#80e3cc"
ASSETS = Path(__file__).with_name("assets")
TICK_MS = 500


class Viewer:
    def __init__(self, root: tk.Tk, levels: dict[str, Level], initial: str | None,
                 controller_factory: Callable[[], Controller]):
        self.root = root
        self.controller_factory = controller_factory
        self.levels = list(levels.values())
        self.timer = None
        self.sprites = {k: tk.PhotoImage(master=root, file=ASSETS / (name + ".png"))
                        for k, name in (("@", "rover"), ("F", "fuel"), ("R", "repair"), ("B", "finish"))}
        self.map_sprites = {1: self.sprites}
        root.title("Навигация робота")
        root.configure(bg=BG)
        root.minsize(770, 570)
        root.geometry("990x640")
        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=22, pady=(18, 12))
        tk.Label(top, text="Навигация робота", font=("Segoe UI", 20, "bold"),
                 bg=BG, fg=INK).pack(side="left")
        self.selector = ttk.Combobox(top, state="readonly", width=30, font=("Segoe UI", 11),
                                     values=[f"{i:02}   {l.title}" for i, l in enumerate(self.levels, 1)])
        self.selector.pack(side="right")
        self.selector.current(list(levels).index(initial) if initial else 0)
        self.selector.bind("<<ComboboxSelected>>", lambda _: self.reset())

        self.canvas = tk.Canvas(root, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=22)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.info = tk.StringVar()
        tk.Label(root, textvariable=self.info, font=("Segoe UI", 11), fg=INK, bg=BG,
                 justify="left", anchor="w").pack(fill="x", padx=22, pady=(8, 0))
        legend = tk.Frame(root, bg=BG)
        legend.pack(fill="x", padx=22, pady=4)
        for symbol, name in (("@", "Робот"), ("F", "Заправка"), ("R", "Мастерская"), ("B", "Финиш")):
            tk.Label(legend, image=self.sprites[symbol], text=name, compound="left",
                     bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(side="left", padx=(0, 10))
        tk.Label(legend, text="Бирюзовая рамка — обзор 3×3", bg=BG, fg=ACCENT).pack(side="right")
        controls = tk.Frame(root, bg=BG)
        controls.pack(fill="x", padx=22, pady=(0, 16))
        self.play = ttk.Button(controls, command=self.toggle)
        self.play.pack(side="left")
        ttk.Button(controls, text="Сначала", command=self.reset).pack(side="left", padx=8)
        tk.Label(controls, text="0,5 с за ход", bg=BG, fg=MUTED).pack(side="left")
        self.full = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Показать полную карту", variable=self.full,
                        command=self.draw).pack(side="right")
        self.reset()

    def reset(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
        self.timer = None
        self.env = Environment(self.levels[self.selector.current()])
        self.bot = self.controller_factory()
        self.play.configure(text="Пуск", state="normal")
        self.draw()

    def toggle(self):
        if self.timer is None:
            self.play.configure(text="Пауза")
            self.timer = self.root.after(TICK_MS, self.tick)
        else:
            self.root.after_cancel(self.timer)
            self.timer = None
            self.play.configure(text="Продолжить")

    def tick(self):
        self.timer = None
        advance(self.env, self.bot)
        self.draw()
        if self.env.status == Status.RUNNING:
            self.timer = self.root.after(TICK_MS, self.tick)
        else:
            self.play.configure(text="Завершено", state="disabled")

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        env = self.env
        view = env.view
        board = env.level.grid if self.full.get() else view.memory
        w, h = len(board[0]), len(board)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        size = max(1, min(48, (width - 20) // w, (height - 20) // h))
        sprite_width = self.sprites["@"].width()
        factor = (sprite_width + size - 1) // size
        if factor not in self.map_sprites:
            self.map_sprites[factor] = {k: img.subsample(factor) for k, img in self.sprites.items()}
        sprites = self.map_sprites[factor]
        ox, oy = (width - size*w)//2, (height - size*h)//2
        for j, row in enumerate(board):
            for i, symbol in enumerate(row):
                x, y = ox+i*size, oy+j*size
                fill = {"#": "#41596b", "?": "#152430"}.get(symbol, "#263e4d")
                canvas.create_rectangle(x+1, y+1, x+size-1, y+size-1, fill=fill, outline="")
                if symbol == "#":
                    canvas.create_line(x+2, y+2, x+size-2, y+2, fill="#5b7382")
                elif symbol == "?":
                    canvas.create_text(x+size//2, y+size//2, text="·", fill="#49606e")
                if symbol in sprites:
                    canvas.create_image(x+size//2, y+size//2, image=sprites[symbol])
        i, j = view.position
        canvas.create_rectangle(ox+max(0, i-1)*size, oy+max(0, j-1)*size,
                                ox+min(w, i+2)*size, oy+min(h, j+2)*size, outline=ACCENT, width=2)
        canvas.create_image(ox+i*size+size//2, oy+j*size+size//2, image=sprites["@"])
        fuel = "—" if view.fuel_distance is None else str(view.fuel_distance)
        repair = "—" if view.repair_distance is None else str(view.repair_distance)
        activity = getattr(self.bot, "activity", "") if env.steps else "Старт"
        self.info.set(f"{env.status.value}    ·    Ход {env.steps}    {activity}    |    "
                      f"Топливо {view.fuel}/100     Исправность {view.health}/100\n"
                      f"До финиша   Dx {view.dx:+}   Dy {view.dy:+}     |     "
                      f"Длина пути до заправки  {fuel}     до мастерской  {repair}")


def show(levels: dict[str, Level], initial: str | None, controller_factory: Callable[[], Controller]):
    root = tk.Tk()
    Viewer(root, levels, initial, controller_factory)
    root.mainloop()
