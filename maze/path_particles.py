"""Проверить, какая доля продолжений измеренной длины образует действительную короткую дорогу."""
from bisect import bisect_right
from itertools import accumulate
from random import Random

from .bridge_automaton import suffixes, transitions
from .cell_hypotheses import occupancy
from .interface import Move
from .paired_witness import open_step
from .pilot_automaton import Mode
from .resource_automaton import Service
from .route_witness import agrees, geometry
from .street_trunk import StreetTrunkReserve


def thin(paths, limit, rng):
    """Ограничить число путей, сохранив суммарный вес и пропорции при повторном отборе."""
    if len(paths) <= limit:
        return paths
    states = list(paths)
    cumulative = list(accumulate(paths.values()))
    step = cumulative[-1] / limit
    offset = rng.random() * step
    kept = {}
    for i in range(limit):
        j = min(bisect_right(cumulative, offset + i * step), len(states) - 1)
        state = states[j]
        kept[state] = kept.get(state, 0.) + step
    return kept


def implied_walls(q, previous, visited, opened, w, h):
    """Какие углы обязаны быть стенами, чтобы диагональ не сократила выбранный путь."""
    x,y = q%w,q//w
    walls = 0
    for dx in (-1,1):
        if not 0<=x+dx<w:
            continue
        for dy in (-1,1):
            other = q+dx+dy*w
            if not 0<=y+dy<h or other==previous or not (visited & (1<<other)):
                continue
            a,b = 1<<(q+dx),1<<(q+dy*w)
            if opened&a and opened&b:
                return None
            if opened&a:
                walls |= b
            if opened&b:
                walls |= a
    return walls


def path_fractions(view, readings, old, prior, sites, options, kind, *, corner_walls=False, limit=32, seed=42, exact=False):
    memory = view.memory
    width = len(memory[0])
    index = lambda p: p[1] * width + p[0]
    origin = index(view.position)
    goal = view.position[0] + view.dx, view.position[1] + view.dy
    known = sum(1 << i for i, c in enumerate("".join(memory)) if c not in "#?") | (1 << index(goal))
    marks = [sum(1 << index(p) for p in group) for group in sites]
    shifts, adjacent = geometry(width, len(memory))
    service = Service((Mode.FUEL, Mode.REPAIR)[kind], (view.fuel, view.health)[kind], readings[kind], sites[kind])
    moves = tuple(Move)
    chance = occupancy(memory)[1] if corner_walls else None
    unknown = sum(1<<i for i,c in enumerate("".join(memory)) if c=="?") if corner_walls else 0
    for restricted in (True, False):
        neighbors, turns = transitions(memory, restricted)
        layers = {t: values.copy() for t, values in suffixes(view, service, old[kind], restricted)}
        starts = [(move, i, int(neighbors[i, origin])) for i, move in enumerate(moves) if move in options]
        starts = [(move, i, q) for move, i, q in starts if layers[1][i, q] > 0.]
        if starts:
            break
    constraints = [(origin, readings)]
    if prior is not None:
        constraints.append((index(prior), old))
    def consistent(mask):
        for p, gauges in constraints:
            active = [i for i, d in enumerate(gauges) if d is not None]
            if not active:
                continue
            required = {active.index(kind)} if p == origin and kind in active else set()
            if not agrees(mask, shifts, p, [marks[i] for i in active], [gauges[i] for i in active], required):
                return False
        return True

    rng = Random(seed)
    chances = {}
    for move, first, q in starts:
        seen = (1 << origin) | (1 << q)
        paths = {(q, first, seen, known | open_step(origin, q, move), 0): 1.}
        for t in range(1, service.distance):
            children = {}
            for (p, last, visited, opened, blocked), weight in paths.items():
                denominator = layers[t][last, p]
                for j, direction in enumerate(moves):
                    n = int(neighbors[j, p])
                    value = turns[last, j, p] * layers[t+1][j, n]
                    if not value or (visited >> n) & 1 or adjacent[n] & (visited ^ (1 << p)):
                        continue
                    proposed = opened | open_step(p,n,direction)
                    next_blocked = blocked
                    mass = weight * (value / denominator)
                    if corner_walls:
                        if proposed & blocked:
                            continue
                        needed = implied_walls(n,p,visited,proposed,width,len(memory))
                        if needed is None:
                            continue
                        new_walls = needed & unknown & ~blocked
                        next_blocked |= new_walls
                        while new_walls:
                            bit = new_walls & -new_walls
                            mass *= 1-chance[bit.bit_length()-1]
                            new_walls ^= bit
                    state = n,j,visited | (1<<n),proposed,next_blocked
                    children[state] = children.get(state, 0.) + mass
                    if exact and len(children)>limit:
                        return None
            paths = thin(children, limit, rng)
            if not paths:
                break
        chances[move] = sum(weight for (*_, mask, _blocked), weight in paths.items() if consistent(mask))
    return chances


def particle_choice(view, readings, old, prior, sites, options, kind, default, raw, *, corner_walls=False):
    factors = {}
    if 0 <= (view.fuel, view.health)[kind] - readings[kind] < 2:
        factors = path_fractions(view, readings, old, prior, sites, options, kind,corner_walls=corner_walls)
    if not any(factors.values()):
        return max(options, key=lambda m: (raw[m], m == default))
    mean = sum(factors.values()) / len(factors)
    return max(options, key=lambda m: (raw[m] * factors.get(m, mean), raw[m], m == default))


class PathParticles(StreetTrunkReserve):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        raw = self.direction_votes(view, readings, old, prior, sites)[kind]
        return particle_choice(view, readings, old, prior, sites, options, kind, default, raw)
