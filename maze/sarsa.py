"""Табличная SARSA(0) с общими оценками для одинаковых признаков физических ходов."""
from dataclasses import dataclass
from functools import lru_cache
from random import Random

from .environment import Status
from .grid import destination
from .interface import Grid, Move, Observation, Point

GAMMA = 0.97
ALPHA = 0.025
MOVES = tuple(Move)


@lru_cache(maxsize=8192)
def _layout(memory: Grid, goal: Point, position: Point):
    # Один бит на клетку и столбец-разделитель, чтобы сдвиг не переносил ход на другую строку.
    stride = len(memory[0]) + 1
    flat = "#".join(memory)
    def mask(symbols):
        return int("".join("1" if c in symbols else "0" for c in reversed(flat)), 2)

    unknown = mask("?")
    optimistic = mask(".BFR") | unknown
    origin = 1 << (position[1]*stride + position[0])
    legal = []
    for move in MOVES:
        point = destination(memory, position, move)
        if point is not None:
            legal.append((move, 1 << (point[1]*stride + point[0])))

    def distances(targets, roads):
        if targets & origin:
            return 0, frozenset()
        front = seen = targets
        length = 0
        while front:
            previous = front
            east, west = (front << 1) & roads, (front >> 1) & roads
            north, south = (front >> stride) & roads, (front << stride) & roads
            # Каждый диагональный переход требует двух свободных боковых клеток.
            diagonals = (((north << 1) & (east >> stride)) | ((north >> 1) & (west >> stride)) |
                         ((south << 1) & (east << stride)) | ((south >> 1) & (west << stride)))
            front = (east | west | north | south | diagonals) & roads & ~seen
            seen |= front
            length += 1
            if front & origin:
                return length, frozenset(move for move, bit in legal if previous & bit)
        return None, frozenset()

    goal_bit = 1 << (goal[1]*stride + goal[0])
    fields = [(distances(t, optimistic), distances(t, optimistic ^ unknown))
              for t in (goal_bit, mask("F"), mask("R"))]
    return tuple(move for move, _ in legal), *fields


@dataclass(frozen=True)
class Choice:
    options: tuple[tuple[Move, int], ...]
    potential: float
    service_signals: tuple[str, ...]


def encode(view: Observation) -> Choice:
    x, y = view.position
    legal, (optimistic, known), fuel, repair = _layout(view.memory, (x + view.dx, y + view.dy),
                                                       view.position)
    estimate, goal = known if known[0] is not None else optimistic
    if estimate is None:
        estimate = max(abs(view.dx), abs(view.dy))
    # Интервал запаса на поиск зависит от наличия уже известного пути до B.
    budget = estimate if known[0] is not None else 2*estimate + 9
    risks = []
    if known[0] is None or min(view.fuel, view.health) < estimate:
        for name, stock, dist, (possible, confirmed) in (
                ("fuel_distance", view.fuel, view.fuel_distance, fuel),
                ("repair_distance", view.health, view.repair_distance, repair)):
            if dist and dist <= stock and (stock <= budget or stock-dist <= 14):
                directions = confirmed[1] if confirmed[0] == dist else possible[1]
                risks.append((stock, dist, name, directions))
    risks.sort()
    first = risks[0][3] if risks else ()
    second = risks[1][3] if len(risks) == 2 else ()
    context = 8*len(risks)
    options = tuple((move, context + 4*(move in goal) + 2*(move in first) + (move in second))
                    for move in (legal or MOVES))
    return Choice(options, -0.65*estimate + 0.06*(view.fuel + view.health), tuple(r[2] for r in risks))


class Sarsa:
    def __init__(self, values=None):
        self.values = list(values) if values is not None else [0.0] * 24
        if len(self.values) != 24:
            raise ValueError("Ожидалось 24 табличных значения")

    def select(self, choice: Choice, epsilon=0.0, rng: Random | None = None) -> tuple[Move, int]:
        if epsilon and rng.random() < epsilon:
            return rng.choice(choice.options)
        return max(choice.options, key=lambda option: self.values[option[1]])

    def choose(self, view: Observation) -> Move:
        return self.select(encode(view))[0]

    def update(self, key: int, reward: float, next_key: int | None):
        target = reward if next_key is None else reward + GAMMA*self.values[next_key]
        self.values[key] += ALPHA*(target - self.values[key])


def train_episode(env, policy: Sarsa, epsilon: float, rng: Random):
    before = env.reset()
    state = encode(before)
    move, key = policy.select(state, epsilon, rng)
    while True:
        view = env.step(move)
        terminal = env.status != Status.RUNNING
        reward = -1.0
        # Сигнал о продвижении использует две последовательные фактические позиции.
        for weight, name in zip((3.0, 1.0), state.service_signals):
            reward += weight*(getattr(before, name) - getattr(view, name))
        if terminal:
            reward += 40 if env.status == Status.SUCCESS else -40
            policy.update(key, reward - state.potential, None)
            return
        following = encode(view)
        next_move, next_key = policy.select(following, epsilon, rng)
        policy.update(key, reward + GAMMA*following.potential - state.potential, next_key)
        before, state, move, key = view, following, next_move, next_key
