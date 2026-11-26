"""Сравнить решения после одной и той же физической разведки на успешных и неуспешных поездках."""
from collections import Counter
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import PhysicalRoutes, read, transition
from maze.environment import Environment
from maze.resource_automaton import known_paths, possible_directions
from maze.specific_paths import SpecificPaths


class WatchReturns(SpecificPaths):
    def refine_approach(self, view, service, prior_lengths, returning, default):
        if returning and service is not None:
            previous = prior_lengths.get(service.mode) if view.moved else None
            if previous is not None and previous >= service.distance:
                roads = known_paths(view.memory, view.position)
                options = possible_directions(view, service, previous)
                if default in options and len(options) > 1 and not any(
                        roads.get(p, (None,))[0] == service.distance for p in service.sites):
                    self.probed = True
        return super().refine_approach(view, service, prior_lengths, returning, default)


def main():
    runs = read("specific-paths")
    records = []
    for split, name, _, _, level in cases():
        clear_episode_caches()
        run = runs[name]
        env, parent = Environment(level), WatchReturns()
        routes = None
        for c in run["actions"]:
            if env.refuels and env.repairs:
                break
            before = deepcopy(vars(parent)) if parent._fallback.returning else None
            parent.probed = False
            move = parent.choose(env.view)
            assert move == MOVES[int(c)]
            if before is not None and parent.probed:
                if routes is None:
                    routes = PhysicalRoutes(level)
                safe = [m.name for m in MOVES if transition(routes, (env.position, env.fuel, env.health),
                                                            m, env.limit-env.steps-1)]
                if not safe:
                    break
                choices = {"specific-paths": move.name}
                for n in ("return-paths", "last-return-paths"):
                    bot = controller(n)
                    bot.__dict__.update(deepcopy(before))
                    choices[n] = bot.choose(env.view).name
                records.append(dict(map=name, split=split, step=env.steps+1, status=run["status"],
                                    safe=safe, choices=choices))
            env.step(move)
    (HERE / "return-controls.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf8")
    for child in ("return-paths", "last-return-paths"):
        print(child, Counter((r["choices"]["specific-paths"] in r["safe"], r["choices"][child] in r["safe"])
                              for r in records), len(set(r["map"] for r in records)))


if __name__ == "__main__":
    main()
