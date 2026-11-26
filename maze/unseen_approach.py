"""При последней попытке предпочесть выход, где ещё предстоит наблюдать дорогу."""
from .grid import destination, locations
from .pilot_automaton import Mode
from .resource_automaton import known_paths, possible_directions
from .second_window import ReturnWindows
from .sight_automaton import window_unknown


class UnseenReturn(ReturnWindows):
    after_return_only = True
    approach_label = "Последний подход через новое окно"

    def alternatives(self, view, options):
        return {m:d for m,d in options.items()
                if window_unknown(view.memory, destination(view.memory, view.position, m))}

    def refine_approach(self, view, service, prior_lengths, returning, default):
        parent = super().refine_approach(view, service, prior_lengths, returning, default)
        if (service is None or self.after_return_only and not returning
                or "несовместимого" in self.activity):
            return parent
        kind = int(service.mode == Mode.REPAIR)
        other = (view.health, view.fuel)[kind]
        if not 0 <= min(service.stock, other-1) - service.distance < 2:
            return parent
        old = tuple(prior_lengths.get(k) if view.moved else None for k in (Mode.FUEL,Mode.REPAIR))
        if old[kind] is not None and old[kind] < service.distance:
            return parent
        options = possible_directions(view, service, old[kind])
        if parent not in options or len(options) < 2:
            return parent
        unseen = self.alternatives(view, options)
        if parent in unseen or not unseen:
            return parent
        roads = known_paths(view.memory, view.position)
        if any(roads.get(p,(None,))[0] == service.distance for p in service.sites):
            return parent
        prior = None
        if view.moved:
            dx,dy = view.last_move.value
            prior = view.position[0]-dx, view.position[1]-dy
        sites = tuple(tuple(locations(view.memory, k)) for k in "FR")
        best = default if default in unseen else next(iter(unseen))
        if len(unseen) > 1:
            best = self.rank_directions(view, (view.fuel_distance,view.repair_distance), old, prior,
                                        sites, unseen, kind, best)
        self.activity = self.approach_label + " — " + service.mode.value
        return best


class UnseenApproach(UnseenReturn):
    after_return_only = False
