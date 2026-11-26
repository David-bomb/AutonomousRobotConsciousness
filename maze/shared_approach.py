"""Выбор общей начальной дороги и сверка независимых оценок направления."""
from .bridge_automaton import bridge_evidence
from .cell_hypotheses import cell_votes
from .grid import destination,locations
from .interface import Move
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions,service_order
from .road_hypotheses import RangeRoads,field_votes


class SharedApproach(RangeRoads):
    def _fallback_move(self,view,goal,stops):
        old = tuple(self._fallback.last_lengths.get(m) if view.moved else None for m in (Mode.FUEL,Mode.REPAIR))
        demand = service_order(view,self._fallback.mode,self.RESERVE)
        default = super()._fallback_move(view,goal,stops)
        f,r = view.fuel_distance,view.repair_distance
        if demand is None or not f or not r or "несовместимого" in self.activity:
            return default
        # При достаточном запасе один из порядков допускает возврат через эту клетку.
        if view.health>=2*f+r or view.fuel>=2*r+f:
            return default
        services = [Service(m,s,d,tuple(locations(view.memory,k))) for m,s,d,k in (
            (Mode.FUEL,view.fuel,f,"F"),(Mode.REPAIR,view.health,r,"R"))]
        options = [possible_directions(view,s,p) for s,p in zip(services,old)]
        common = options[0].keys() & options[1].keys()
        if not common or default in common:
            return default
        prior=None
        if view.moved:
            dx,dy=view.last_move.value
            prior=view.position[0]-dx,view.position[1]-dy
        votes=field_votes(view.memory,view.position,(f,r),old,prior,tuple(s.sites for s in services))
        def rank(m):
            return sum(scores[m]/max(max(scores.values()),1e-12) for scores in votes)
        chosen=max((m for m in Move if m in common),key=rank)
        self.activity="Общий начальный путь к двум сервисам"
        self._fallback.returning=False
        return chosen


class Consensus(RangeRoads):
    rank_return=True

    def direction_votes(self,view,readings,old,prior,sites):
        args=view.memory,view.position,readings,old,prior,sites
        networks=field_votes(*args)
        cells=cell_votes(*args)
        out=[]
        for i,(mode,stock) in enumerate(((Mode.FUEL,view.fuel),(Mode.REPAIR,view.health))):
            result=dict.fromkeys(networks[i],0.)
            d=readings[i]
            if d is not None and 0<d<=stock:
                service=Service(mode,stock,d,sites[i])
                choices=possible_directions(view,service,old[i])
                bridge=bridge_evidence(view,service,old[i],True) or bridge_evidence(view,service,old[i])
                for opinions in (networks[i],cells[i],bridge):
                    total=sum(opinions.get(m,0.) for m in choices)
                    if total:
                        for m in choices:
                            result[m]+=opinions.get(m,0.)/total
            out.append(result)
        return out
