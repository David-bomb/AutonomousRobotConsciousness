"""Выбирать способ проверки подхода по числу станций, совместимых с показанием."""
from .frontier_reserve import _lengths, _optimistic
from .path_particles import particle_choice
from .pilot_automaton import Mode
from .service_exit_paths import StationUsefulPaths


class SpecificPaths(StationUsefulPaths):
    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        measured = readings[kind]
        if measured is not None and 0 <= (view.fuel, view.health)[kind] - measured < 2:
            lower = _lengths(_optimistic(view.memory), view.position)
            plausible = [p for p in sites[kind] if lower.get(p, float("inf")) <= measured]
            if len(plausible) == 1:
                raw = self.direction_votes(view, readings, old, prior, sites)[kind]
                return particle_choice(view, readings, old, prior, sites, options, kind, default, raw)
        return super().rank_directions(view, readings, old, prior, sites, options, kind, default)


class ReturnPaths(SpecificPaths):
    """После возврата с неудачной пробы заново оценить оставшиеся направления."""
    rank_return = True


class LastReturnPaths(ReturnPaths):
    """Повторно сравнивать направления, когда ресурсов уже не хватит на ещё одну пробу с возвратом."""
    def refine_approach(self, view, service, prior_lengths, returning, default):
        if returning and service:
            other = view.health if service.mode == Mode.FUEL else view.fuel
            spare = min(service.stock, other-1) - service.distance
            if spare >= 2:
                return default
        return super().refine_approach(view, service, prior_lengths, returning, default)
