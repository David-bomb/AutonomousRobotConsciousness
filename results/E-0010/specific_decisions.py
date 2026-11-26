"""Восстановить правило, команду и доступные варианты перед каждым отказом specific-paths."""
from collections import Counter
from argparse import ArgumentParser
import json

from bench import HERE, MOVES, cases, clear_episode_caches
from compare import read
from maze.environment import Environment
from maze.grid import destination
from maze.resource_automaton import known_paths, possible_directions
from maze.specific_paths import SpecificPaths


class Recorder(SpecificPaths):
    def refine_approach(self, view, service, prior_lengths, returning, default):
        if self.capture and self.refinements:
            note = dict(default=default.name, returning=returning, activity=self.activity)
            if service:
                old = prior_lengths.get(service.mode) if view.moved else None
                roads = known_paths(view.memory, view.position)
                note.update(kind=service.mode.value, distance=service.distance, previous=old,
                            options=[m.name for m in possible_directions(view, service, old)],
                            known=any(roads.get(p, (None,))[0] == service.distance for p in service.sites))
            self.calls.append(note)
        return super().refine_approach(view, service, prior_lengths, returning, default)

    def rank_directions(self, view, readings, old, prior, sites, options, kind, default):
        move = super().rank_directions(view, readings, old, prior, sites, options, kind, default)
        if self.capture:
            self.calls.append(dict(kind="FR"[kind], options=[m.name for m in options],
                                   default=default.name, chosen=move.name))
        return move

    def pick(self, view, goal, arrivals):
        choices = list(arrivals)
        result = super().pick(view, goal, choices)
        if self.capture and result is not None:
            self.calls.append(dict(window=result.point, chosen=result.first.name,
                                   alternatives=len({a.point for a in choices})))
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--missed-rank", action="store_true")
    missed = parser.parse_args().missed_rank
    runs = read("specific-paths")
    source = HERE / "specific-last-waypoints.jsonl"
    losses = {r["map"]: r for r in map(json.loads, source.read_bytes().splitlines())}
    if missed:
        reference = map(json.loads, (HERE / "specific-decisions.jsonl").read_bytes().splitlines())
        losses = {r["map"]: losses[r["map"]] for r in reference if not r["calls"]}
    result = []
    for split, name, _, _, level in cases():
        if name not in losses:
            continue
        clear_episode_caches()
        env, agent = Environment(level), Recorder()
        agent.calls = []
        agent.refinements = missed
        record = losses[name]
        step = record["loss"]["step"]
        for i, c in enumerate(runs[name]["actions"][:step], 1):
            agent.capture = i == step
            choice = agent.choose(env.view)
            assert choice == MOVES[int(c)], (name, i)
            if i < step:
                env.step(choice)
        viable = set(record["loss"]["viable"])
        approaches = {}
        for kind, field in env._lengths.items():
            if env.position in field:
                approaches[kind] = [m.name for m in MOVES if (q := destination(level.grid, env.position, m))
                                    and field[q] == field[env.position] - 1]
        first_stops = sorted({s["kind"] for group in record["waypoints"].values() for s in group})
        result.append(dict(map=name, split=split, step=step, activity=agent.activity, calls=agent.calls,
                           first_stops=first_stops, viable=sorted(viable), downhill=approaches))
    assert len(result) == len(losses)
    path = HERE / ("specific-missed-rank.jsonl" if missed else "specific-decisions.jsonl")
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result), encoding="utf8")
    print(Counter(r["activity"] for r in result))


if __name__ == "__main__":
    main()
