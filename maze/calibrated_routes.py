"""Проверка конкурирующих оценок направления последним реальным замером."""
from .bridge_automaton import bridge_evidence
from .cell_hypotheses import cell_votes
from .interface import Move
from .pattern_roads import PatternRoads,pattern_votes
from .pilot_automaton import Mode
from .resource_automaton import Service,possible_directions


class StreetConsensus(PatternRoads):
    rank_return=True
    calibrate=False

    def direction_votes(self,view,readings,old,prior,sites):
        evidence=(None,None) if self.calibrate else old
        anchor=None if self.calibrate else prior
        args=view.memory,view.position,readings,evidence,anchor,sites
        sources=[pattern_votes(*args),cell_votes(*args),[]]
        choices=[]
        for i,(mode,stock) in enumerate(((Mode.FUEL,view.fuel),(Mode.REPAIR,view.health))):
            d=readings[i]
            if d is None or not 0<d<=stock:
                choices.append({})
                sources[2].append({})
                continue
            service=Service(mode,stock,d,sites[i])
            choices.append(possible_directions(view,service,old[i]))
            sources[2].append(bridge_evidence(view,service,evidence[i],True) or
                              bridge_evidence(view,service,evidence[i]))
        out=[dict.fromkeys(table,0.) for table in sources[0]]
        for source in sources:
            weight=1.
            if self.calibrate and view.moved:
                dx,dy=view.last_move.value
                reverse=Move((-dx,-dy))
                for i in range(2):
                    total=sum(source[i].values())
                    if not total or readings[i] is None or old[i] is None:
                        continue
                    probability=source[i].get(reverse,0.)/total
                    observed=probability if old[i]<readings[i] else 1-probability
                    # Показание на обратной стороне действительно получено.
                    # Один факт имеет конечный вес даже при нулевой доле выборок.
                    weight*=(0.5+observed)/2
            for i,allowed in enumerate(choices):
                total=sum(source[i].get(m,0.) for m in allowed)
                if total:
                    for m in allowed:
                        out[i][m]+=weight*source[i].get(m,0.)/total
        return out


class CalibratedRoutes(StreetConsensus):
    calibrate=True
