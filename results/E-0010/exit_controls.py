"""Выбор выезда на одинаковых наблюдениях поздних отказов и успешных поездок."""
from concurrent.futures import ProcessPoolExecutor
import json

from bench import HERE, MOVES, cases, clear_episode_caches, controller
from compare import PhysicalRoutes, read, transition
from maze.environment import Environment
from maze.first_window import window_arrivals


MODE = "Первое новое наблюдение на пути к финишу"
RULES = ("ranked-standby","window-patterns","passage-patterns","street-exits","gauge-exits",
         "serviced-windows","available-exits","flexible-exits")


def measure(job):
    case,run,loss = job
    split,name,_,_,level = case
    clear_episode_caches()
    env,robot = Environment(level),controller("ranked-standby")
    answers = {}
    for c in run["actions"]:
        view = env.view
        action = robot.choose(view)
        assert action==MOVES[int(c)]
        step = env.steps+1
        if ((loss is None or step==loss) and env.refuels and env.repairs and robot.activity==MODE):
            options = list(window_arrivals(view))
            if loss is not None or len({a.first for a in options})>1:
                routes = PhysicalRoutes(level)
                left = env.limit-env.steps
                for label in RULES:
                    candidate = robot if label=="ranked-standby" else controller(label)
                    arrival = candidate.pick(view,level.goal,options)
                    if label=="ranked-standby":
                        assert arrival.first==action,(name,step)
                    viable = transition(routes,(view.position,view.fuel,view.health),arrival.first,left-1)
                    terminal = routes.allows(arrival.point,arrival.fuel,arrival.health,left-arrival.length)
                    answers[label] = dict(move=arrival.first.name,point=arrival.point,length=arrival.length,
                                          fuel=arrival.fuel,health=arrival.health,viable=viable,arrival_viable=terminal)
                assert answers["ranked-standby"]["viable"]==(loss is None)
                break
        env.step(action)
    assert loss is None or answers
    return dict(map=name,split=split,late_failure=loss is not None,step=env.steps+1 if answers else None,choices=answers)


def main():
    runs = read("ranked-standby")
    failures = {r["map"]:r["first"] for r in map(json.loads,(HERE/"return-windows-next-loss.jsonl").read_bytes().splitlines())}
    old = read("return-windows")
    assert all(old[name]["actions"]==runs[name]["actions"] for name in failures)
    path = HERE/"exit-controls.jsonl"
    completed = {r["map"] for r in map(json.loads,path.read_bytes().splitlines())} if path.exists() else set()
    jobs = [(c,runs[c[1]],failures.get(c[1])) for c in cases() if c[1] not in completed
            and (c[1] in failures or runs[c[1]]["status"]=="SUCCESS" and MODE in runs[c[1]]["modes"])]
    with path.open("a",encoding="utf8") as stream, ProcessPoolExecutor(2) as pool:
        for row in pool.map(measure,jobs):
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
            stream.flush()
    print(len(jobs),"проверенных поездок",flush=True)


if __name__=="__main__":
    main()
