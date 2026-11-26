"""После одной внешней поправки найти следующий необратимый ход и сравнить наблюдения."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import PhysicalRoutes, read, transition
from maze.environment import Environment
from maze.resource_automaton import opposite


def measure(job):
    name,level,run,previous = job
    clear_episode_caches()
    env,robot = Environment(level),controller("return-windows")
    routes = PhysicalRoutes(level)
    first = previous["first_loss"]
    for c in run["actions"][:first-1]:
        action = robot.choose(env.view)
        assert action == MOVES[int(c)]
        env.step(action)
    rejected = robot.choose(env.view)
    assert previous["proposed"] == rejected.name
    base = env.view
    rows = []
    for trial in previous["trials"]:
        move = next(m for m in MOVES if m.name == trial["move"])
        episode,machine = deepcopy(env),deepcopy(robot)
        episode.step(move)
        while True:
            v = episode.view
            choice = machine.choose(v)
            state = v.position,v.fuel,v.health
            left = episode.limit-episode.steps-1
            if not transition(routes,state,choice,left):
                safe = [m.name for m in MOVES if transition(routes,state,m,left)]
                assert safe
                rows.append(dict(corrected=move.name, next_step=episode.steps+1,
                                 pos=v.position, stocks=[v.fuel,v.health],
                                 gauges=[v.fuel_distance,v.repair_distance],
                                 mode=machine.activity, proposed=choice.name, safe=safe,
                                 reverse_immediate=episode.steps==first and choice==opposite(move),
                                 new_cells=sum(a.count("?")-b.count("?") for a,b in zip(base.memory,v.memory)),
                                 visits=[episode.refuels-env.refuels,episode.repairs-env.repairs]))
                break
            episode.step(choice)
            assert episode.status.name=="RUNNING", (name,trial)
    return dict(map=name,split=previous["split"],first=first,pos=base.position,proposed=rejected.name,
                first_mode=robot.activity, corrections=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--early",action="store_true",help="проверить ранние отказы, оставшиеся после всех разовых поправок")
    early = parser.parse_args().early
    runs = read("return-windows")
    saved = {r["map"]:r for r in map(json.loads,(HERE/"return-windows-recovery.jsonl").read_bytes().splitlines())
             if all(r["serviced"]) != early and all(t["status"]!="SUCCESS" for t in r["trials"])}
    assert len(saved)==(29 if early else 21)
    path = HERE/("return-windows-next-early.jsonl" if early else "return-windows-next-loss.jsonl")
    done = {r["map"] for r in map(json.loads,path.read_bytes().splitlines())} if path.exists() else set()
    with ProcessPoolExecutor(2) as pool,path.open("a",encoding="utf8") as stream:
        jobs = [(name,level,runs[name],saved[name]) for _,name,_,_,level in cases() if name in saved and name not in done]
        for row in pool.map(measure,jobs):
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
    print(len(saved),"карт")


if __name__=="__main__":
    main()
