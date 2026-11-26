"""Проверять совместимость двух предполагаемых дорог с измеренными длинами."""
from functools import lru_cache
from random import Random

from .bridge_automaton import suffixes, transitions
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions
from .road_hypotheses import RangeRoads,field_votes


@lru_cache(maxsize=32)
def geometry(width,height):
    left = sum(1<<(y*width) for y in range(height))
    right = left<<(width-1)
    shifts = (lambda x:x>>width,lambda x:(x & ~right)<<1,
              lambda x:x<<width,lambda x:(x & ~left)>>1)
    adjacent = tuple(sum(step(1<<p) for step in shifts) for p in range(width*height))
    return shifts,adjacent


def agrees(opened,shifts,start,markers,readings,required):
    north,east,south,west = shifts
    corners = [(a,b,a(opened)&b(opened)) for a in (north,south) for b in (east,west)]
    reached = 1<<start
    for d in range(max(readings)+1):
        for i,(sites,actual) in enumerate(zip(markers,readings)):
            if d<actual and reached & sites or d==actual and i in required and not reached & sites:
                return False
        adjacent = north(reached)|south(reached)|east(reached)|west(reached)
        for a,b,mask in corners:
            adjacent |= a(b(reached)) & mask
        new = reached | (adjacent & opened)
        if new==reached:
            return all(i not in required or reached & sites for i,sites in enumerate(markers))
        reached = new
    return True


def paths(view,service,old,origin,adjacent,rng,with_points=False):
    d = service.distance
    options = possible_directions(view,service,old)
    moves = tuple(Move)
    if not options:
        return []
    for restricted in (True,False):
        neighbors,turns = transitions(view.memory,restricted)
        layers = {t:v.copy() for t,v in suffixes(view,service,old,restricted)}
        starts = [(i,int(neighbors[i,origin]),layers[1][i,neighbors[i,origin]])
                  for i,m in enumerate(moves) if m in options]
        starts = [(i,p,mass) for i,p,mass in starts if mass>0]
        if starts:
            break
    out = []
    for first,q,start_weight in starts:
        for _ in range(max(4,32//len(starts))):
            p,last = q,first
            visited = (1<<origin)|(1<<q)
            opened = visited
            weight = start_weight
            points = [origin,q] if with_points else None
            # Первый ход проходит только по реально наблюдавшемуся квадрату.
            for t in range(1,d):
                choices = []
                total = possible = 0.
                for j,m in enumerate(moves):
                    n = int(neighbors[j,p])
                    value = turns[last,j,p]*layers[t+1][j,n]
                    if value<=0:
                        continue
                    total += value
                    if (visited>>n)&1 or adjacent[n] & (visited ^ (1<<p)):
                        continue
                    possible += value
                    choices.append((possible,j,n))
                if not choices:
                    break
                weight *= possible/total
                draw = rng.random()*possible
                _,j,n = next(z for z in choices if z[0]>=draw)
                dx,dy = moves[j].value
                if dx and dy:
                    opened |= (1<<(p+dx)) | (1<<(n-dx))
                p,last = n,j
                visited |= 1<<p
                opened |= 1<<p
                if with_points:
                    points.append(p)
            else:
                if with_points:
                    out.append((moves[first],opened,weight,tuple(points)))
                else:
                    out.append((moves[first],opened,weight))
    return out


class WitnessRoads(RangeRoads):
    rank_return = True

    def direction_votes(self,view,readings,old,prior,sites):
        memory,position = view.memory,view.position
        base = field_votes(memory,position,readings,old,prior,sites)
        if any(v is None or not 0<v<=stock for v,stock in zip(readings,(view.fuel,view.health))):
            return base
        width,height = len(memory[0]),len(memory)
        index = lambda p:p[1]*width+p[0]
        origin = index(position)
        known = sum(1<<i for i,c in enumerate("".join(memory)) if c not in "#?")
        known |= 1<<index((position[0]+view.dx,position[1]+view.dy))
        markers = [sum(1<<index(p) for p in group) for group in sites]
        shifts,adjacent = geometry(width,height)
        rng = Random(42)
        accepted = []
        for i,mode in enumerate((Mode.FUEL,Mode.REPAIR)):
            service = Service(mode,(view.fuel,view.health)[i],readings[i],sites[i])
            candidates = paths(view,service,old[i],origin,adjacent,rng)
            pool = {}
            for move,opened,weight in candidates:
                key = move,opened
                pool[key] = pool.get(key,0.)+weight
            accepted.append([(m,mask,w) for (m,mask),w in pool.items()
                             if agrees(known|mask,shifts,origin,markers,readings,{i})])
        votes = [dict.fromkeys(v,0.) for v in base]
        for ma,a,wa in accepted[0]:
            for mb,b,wb in accepted[1]:
                opened = known|a|b
                if not agrees(opened,shifts,origin,markers,readings,{0,1}):
                    continue
                if prior is not None and None not in old and not agrees(
                        opened,shifts,index(prior),markers,old,set()):
                    continue
                votes[0][ma] += wa*wb
                votes[1][mb] += wa*wb
        return votes if any(votes[0].values()) else base
