"""На каком шаге поездки исчезал физически возможный возврат к возобновляемому сервису."""
from collections import Counter
import json

from bench import HERE, MOVES, cases
from compare import PhysicalRoutes, read
from maze.environment import Environment
from maze.frontier_automaton import _arrive, _paths
from maze.renewable_routes import renewable_routes
from maze.remote_station import can_rejoin


def main():
    pairs = [json.loads(b) for b in (HERE/"marked-streets--street-trunk.jsonl").read_bytes().splitlines()]
    losses = {r["map"]:r["child_loss"] for r in pairs if "child_loss" in r}
    runs = read("street-trunk")
    totals = Counter()
    with (HERE/"return-boundary.jsonl").open("x", encoding="utf8") as out:
        for split, name, tier, family, level in cases():
            event = losses.get(name)
            if not event or not (event["refuels"] and event["repairs"]):
                continue
            env = Environment(level)
            routes = PhysicalRoutes(level)
            stops = tuple((p,k) for p,k in routes.stops if k in "FR")
            needs = renewable_routes(level.grid, stops)
            def can_return():
                for site,(kind,need) in needs.items():
                    length = routes.lengths[site].get(env.position)
                    if length is None:
                        continue
                    stocks = _arrive(kind, length, env.fuel, env.health) if length else (env.fuel,env.health)
                    if stocks and stocks[1 if kind=="F" else 0] >= need:
                        return True
                return False
            was_safe = can_return()
            initially = was_safe
            departure = None
            for step,c in enumerate(runs[name]["actions"][:event["step"]],1):
                before = env.view
                env.step(MOVES[int(c)])
                now_safe = can_return()
                assert not (now_safe and not was_safe), (name,step)
                if was_safe and not now_safe:
                    known = can_rejoin(before,stops,_paths(before.memory,before.position))
                    departure = dict(step=step,pos=before.position,move=MOVES[int(c)].name,fuel=before.fuel,
                                     health=before.health,known_return=known)
                was_safe = now_safe
            lag = event["step"]-departure["step"] if departure else None
            totals["states"] += 1
            totals["began_with_base_reachable"] += initially
            totals["departure_at_loss"] += lag==0
            totals["earlier_departure"] += lag is not None and lag>0
            totals["still_recoverable"] += was_safe
            row = dict(map=name,split=split,critical=event,departure=departure,lag=lag,
                       initially_recoverable=initially,after_loss_recoverable=was_safe)
            out.write(json.dumps(row)+"\n")
            out.flush()
    print(totals)


if __name__ == "__main__":
    main()
