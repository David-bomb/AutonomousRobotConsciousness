"""Сравнивать неизвестные участки предполагаемых дорог с уже обследованными коридорами."""
from functools import lru_cache
from math import exp, log

from .exchanged_roads import ExchangedRoads, exchanged_samples
from .room_lengths import room_samples
from .route_witness import geometry


def motifs(opened, centers, shifts):
    """Тупик, прямая, поворот, развилка, перекрёсток или свободный квадрат."""
    adjacent = [s(opened) for s in shifts]
    room = 0
    for i in (0,2):
        for j in (1,3):
            room |= shifts[i](adjacent[j]) & adjacent[i] & adjacent[j]
    # Клетки с хотя бы одним разрешённым диагональным ходом образуют отдельный класс.
    counts = [0]*7
    counts[6] = (centers & opened & room).bit_count()
    for mask in range(16):
        region = centers & opened & ~room
        for i,bits in enumerate(adjacent):
            region &= bits if mask & (1<<i) else ~bits
        degree = mask.bit_count()
        label = degree
        if degree==2:
            label = 2 if mask in (5,10) else 3
        elif degree>2:
            label = degree+1
        counts[label] += region.bit_count()
    return counts


@lru_cache(maxsize=64)
def observed_motifs(memory):
    w,h = len(memory[0]),len(memory)
    shifts,_ = geometry(w,h)
    terrain = "".join(memory)
    roads = sum(1<<i for i,c in enumerate(terrain) if c not in "#?")
    hidden = sum(1<<i for i,c in enumerate(terrain) if c=="?")
    n,e,s,v = shifts
    extended = hidden|e(hidden)|v(hidden)
    uncertain = (extended|n(extended)|s(extended)) & ((1<<(w*h))-1)
    return motifs(roads,~uncertain,shifts),uncertain,shifts


class TerrainTrials(ExchangedRoads):
    def trial_weights(self,view,trials):
        observed,unseen,shifts = observed_motifs(view.memory)
        n = sum(observed)
        if not n:
            return {}
        logs = {}
        # Соседние квадраты 3×3 перекрываются. Девять таких центров дают
        # в среднем один независимый фрагмент, при старте доступен один пример.
        scale = max(1/n,1/9)
        for trial in trials:
            for mask in trial:
                if mask not in logs:
                    sample = motifs(mask,unseen,shifts)
                    size = sum(sample)+len(sample)
                    logs[mask] = scale*sum(a*log((b+1)/size) for a,b in zip(observed,sample))
        maximum = max(logs.values(),default=0.)
        return {mask:exp(value-maximum) for mask,value in logs.items()}


@lru_cache(maxsize=64)
def mixed_trials(memory,goal,origin,gauges,prior,old):
    args = memory,goal,origin,gauges,prior,old
    streets = exchanged_samples(*args)
    rooms = room_samples(*args)
    return streets + tuple((mask,) for mask,_ in rooms)


class TerrainRooms(TerrainTrials):
    sample_trials = staticmethod(mixed_trials)
