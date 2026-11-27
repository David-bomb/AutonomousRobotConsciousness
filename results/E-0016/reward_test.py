"""Исполнение архивных команд только для аудита награды."""
from itertools import cycle
from math import isclose

from maze.environment import Environment, Status
from maze.interface import Move
from neural_rl.rollout import reward, unknown
from .data import OUT, ROOT, STRESS, level, read, rows, save

PARAMS = dict(gamma=.97, step=-.12, terminal=8., goal=.6, scale=32, coverage=.8, deficit=.8, reserve=10)
MOVES = tuple(Move)


def potential(v):
    p = PARAMS
    coverage = 1 - unknown(v.memory)/(len(v.memory)*len(v.memory[0]))
    deficits = sum(min(1., max(0., (length + p["reserve"] - stock)/p["reserve"]))
                   for stock, length in ((v.fuel, v.fuel_distance), (v.health, v.repair_distance))
                   if length is not None)
    return p["coverage"]*coverage - p["goal"]*max(abs(v.dx),abs(v.dy))/p["scale"] - p["deficit"]*deficits


def shaped(before, after, status, gamma):
    terminal = status != Status.RUNNING
    end = (PARAMS["terminal"] if status == Status.SUCCESS else -PARAMS["terminal"]) if terminal else 0.
    return PARAMS["step"] + end + (0. if terminal else gamma*potential(after)) - potential(before)


def drive(env, actions):
    gamma = PARAMS["gamma"]
    initial = potential(env.view)
    n = 0
    old = new = 0.
    for action in actions:
        before = env.view
        env.step(MOVES[int(action)])
        assert env.moved
        old += gamma**n * reward(before, env.view, env.status)
        new += gamma**n * shaped(before, env.view, env.status, gamma)
        n += 1
        if env.status != Status.RUNNING:
            break
    assert env.status != Status.RUNNING
    terminal = PARAMS["terminal"] if env.status == Status.SUCCESS else -PARAMS["terminal"]
    exact = PARAMS["step"]*(1-gamma**n)/(1-gamma) + terminal*gamma**(n-1) - initial
    assert isclose(new, exact, abs_tol=1e-10)
    return dict(steps=n, outcome=env.status.name, old=old, new=new, residual=abs(new-exact))


def main():
    cases = {r["name"]:level(r) for f in STRESS for r in rows(f)}
    witness = read(ROOT/"results/E-0013/witnesses.json")
    wins = {k:drive(Environment(v), witness[k]["actions"]) for k,v in cases.items()}
    assert all(r["outcome"]=="SUCCESS" and r["steps"]==witness[k]["steps"] for k,r in wins.items())
    pairs = []
    for family in ("double","dueling","rainbow","ppo"):
        for r in read(ROOT/f"results/E-0014/{family}-traces.json"):
            k = r["map"]
            loss = drive(Environment(cases[k]), r["actions"])
            assert (loss["steps"],loss["outcome"]) == (r["steps"],r["outcome"])
            assert isclose(loss["old"],r["discounted_return"],abs_tol=1e-8)
            assert isclose(wins[k]["old"],r["witness_return"],abs_tol=1e-8)
            pairs.append(dict(family=family,map=k,winning=wins[k],losing=loss))
    k = "H3-04"
    env = Environment(cases[k])
    actions = [int(a) for a in witness[k]["actions"]]
    sites = []
    for i, a in enumerate(actions,1):
        env.step(MOVES[a])
        tile = env.level.grid[env.position[1]][env.position[0]]
        if tile in "FR":
            sites.append((i,tile))
        if sites and tile in "FR" and tile != sites[0][1]:
            break
    route = actions[sites[0][0]:i]
    reverse = [MOVES.index(Move(tuple(-v for v in MOVES[a].value))) for a in reversed(route)]
    loop = drive(env, cycle(reverse+route))
    env.reset()
    for a in actions[:i]:
        env.step(MOVES[a])
    finish = drive(env,actions[i:])
    assert loop["outcome"]=="LIMIT" and finish["outcome"]=="SUCCESS" and loop["old"]>finish["old"]
    assert loop["new"]<finish["new"] and all(p["winning"]["new"]>p["losing"]["new"] for p in pairs)
    summary = dict(pairs=len(pairs),old_win=sum(p["winning"]["old"]>p["losing"]["old"] for p in pairs),
                   new_win=len(pairs),shuttle=dict(map=k,from_step=i,period=len(route)*2,finish=finish,loop=loop))
    save(OUT/"reward-test.json",dict(params=PARAMS,summary=summary,pairs=pairs))
    print(summary,flush=True)


if __name__ == "__main__":
    main()
