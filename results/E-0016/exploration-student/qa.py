"""Независимые проверки геометрии входа, покартовых исходов и стадий поездки."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from collections import Counter,deque
import statistics
import numpy as np
import torch
from maze.environment import Environment,Status
from maze.grid import cell,destination,locations,wave
from maze.levels import Level
from .adapter import Controller
from .bank import DEV,HERE,PARENT,ROOT,digest,level,read,rows,save,train_order
from .features import DLL,MOVES,encode


def distances(grid,origin,excluded):
    found={origin:1}
    queue=deque([origin])
    while queue:
        p=queue.popleft()
        for move in MOVES:
            q=destination(grid,p,move)
            if q is not None and q!=excluded and q not in found:
                found[q]=found[p]+1
                queue.append(q)
    return found


def check_view(v):
    grid=v.memory
    x,y=v.position
    goal=(x+v.dx,y+v.dy)
    h,w=len(grid),len(grid[0])
    raw=np.empty((8,20),np.float32)
    tokens=np.empty((8,8,12),np.float32)
    DLL.describe("".join(grid).encode("ascii"),w,h,y*w+x,goal[1]*w+goal[0],v.fuel,v.health,raw,tokens)
    sites=[[goal],locations(grid,"F"),locations(grid,"R")]
    homes=[{p:d for p,d,_ in wave(grid,group)} for group in sites[1:]]
    checked=0
    for a,move in enumerate(MOVES):
        first=destination(grid,v.position,move)
        if first is None:
            assert not raw[a].any() and not tokens[a].any()
            continue
        found=distances(grid,first,v.position)
        assert len(found)==raw[a,12]
        for k,ss in enumerate(sites):
            assert raw[a,4*k]==min([512]+[found.get(s,512) for s in ss])
        hidden={p:sum(cell(grid,p[0]+m.value[0],p[1]+m.value[1])=="?" for m in MOVES) for p in found}
        fronts=[p for p in found if hidden[p]]
        assert len(fronts)==raw[a,13] and min([512]+[found[p] for p in fronts])==raw[a,19]
        for t in tokens[a]:
            if t[0]==0:
                continue
            p=x+int(t[8]),y+int(t[9])
            assert p in fronts and (found[p],hidden[p])==tuple(t[:2])
            assert tuple(home.get(p,512) for home in homes)==tuple(t[4:6])
            checked+=1
    return checked


def phases(episodes):
    result={}
    for outcome in ("success","failure"):
        group=[e for e in episodes if (e["outcome"]=="SUCCESS")==(outcome=="success") and e["both_at"] is not None]
        result[outcome]=dict(n=len(group))
        if group:
            result[outcome].update({key:statistics.median(e[key] for e in group)
                                    for key in ("post_both_steps","post_both_discovered","post_both_service_visits")})
    return result


def compare(episodes,baseline):
    ref={r["map"]:r for r in baseline}
    assert set(ref)=={r["map"] for r in episodes}
    gain=[]; loss=[]; post_gain=[]; post_loss=[]
    joint=Counter()
    for e in episodes:
        b=ref[e["map"]]
        before=b["outcome"]=="SUCCESS"
        after=e["outcome"]=="SUCCESS"
        baseline_both=min(b["refuels"],b["repairs"])>0
        if before!=after:
            (gain if after else loss).append(e["map"])
            if baseline_both:
                (post_gain if after else post_loss).append(e["map"])
        if baseline_both and e["both_at"] is not None:
            joint.update(total=1,baseline_wins=int(before),student_wins=int(after))
    return dict(gained=gain,lost=loss,gained_on_baseline_both=post_gain,lost_on_baseline_both=post_loss,
                both_reached_by_each=dict(joint))


def main():
    torch.set_num_threads(1)
    reg=read(HERE/"registration.json")
    assert all(digest(ROOT/p)==s for p,s in reg["sources"].items())
    selected=read(HERE/"selected.json")
    winner=selected["tag"]
    model=Controller(HERE/selected["weights"],selected["sha256"])
    assert read(HERE/"repeat.json")["tag"]==winner
    bank=train_order()
    snapshots=candidates=0
    for r in bank[:16]:
        env=Environment(level(r))
        demos=read(HERE/"teacher-traces"/(r["name"]+".json"))
        sampled_after=False
        for a in demos["actions"]:
            if env.steps in (0,9) or (not sampled_after and min(env.refuels,env.repairs)>0):
                candidates+=check_view(env.view)
                snapshots+=1
                sampled_after |= min(env.refuels,env.repairs)>0
            env.step(MOVES[int(a)])
            if sampled_after and env.steps>10: break
    example="QA\n30 30\n###########\n#A..F....B#\n#.........#\n#..R......#\n#.........#\n###########"
    one=Environment(Level.parse(example)); two=Environment(Level.parse(example.replace("#..R......#","#..R....#.#")))
    assert one.level.grid!=two.level.grid and one.view==two.view
    assert all(np.array_equal(a,b) for a,b in zip(encode(one.view),encode(two.view),strict=True))
    assert model.choose(one.view)==model.choose(two.view)

    evaluation={row["name"]:row for row in rows(DEV)}
    results={}
    excursions={}
    for tag in ("random","imitation12","imitation32","corrected16"):
        data=read(HERE/"dev"/(tag+".json"))
        assert data["weights_sha256"]==digest(HERE/"weights"/(tag+".pt"))
        assert data["dev_sha256"]==digest(DEV)
        assert [e["map"] for e in data["episodes"]]==reg["dev_names"]
        for e in data["episodes"]:
            env=Environment(level(evaluation[e["map"]]))
            station=None
            discovered=False
            returns=same_station=0
            for a in e["actions"]:
                before=env.view
                had_both=min(env.refuels,env.repairs)>0
                env.step(MOVES[int(a)])
                assert env.moved
                if tag==winner and min(env.refuels,env.repairs)>0:
                    kind=cell(env.view.memory,*env.position)
                    if kind in "FR":
                        if had_both and discovered and env.status==Status.RUNNING:
                            returns+=1
                            same_station+=station==env.position
                        station=env.position
                        discovered=False
                    elif had_both and env.view.memory!=before.memory:
                        discovered=True
            assert (env.status.name,env.steps,env.refuels,env.repairs)==(
                e["outcome"],e["steps"],e["refuels"],e["repairs"])
            if tag==winner and e["both_at"] is not None:
                excursions[e["map"]]=dict(outcome=e["outcome"],returns_after_discovery=returns,
                                           returns_to_same_station=same_station)
        results[tag]=dict(summary=data["summary"],post_both_medians=phases(data["episodes"]),
                          weights_sha256=data["weights_sha256"],episode_file_sha256=digest(HERE/"dev"/(tag+".json")))
    held=set(evaluation)
    teacher=set(read(HERE/"teacher.json")["files"])
    correction=set(read(HERE/"correction.json")["files"])
    assert not (teacher&correction or teacher&held or correction&held)
    archives={}
    for kind in ("teacher","correction"):
        m=read(HERE/(kind+"-samples.json"))
        counts=Counter()
        for name,s in m["files"].items():
            path=HERE/f"{kind}-samples"/(name+".npz")
            assert digest(path)==s
            with np.load(path) as d:
                counts.update(map(int,d["phase"]))
        archives[kind]=dict(examples_by_service_count=dict(counts),records=m["records"],maps=m["maps"])
    previous=PARENT/"runs/recovery800/rl.physical.json"
    baseline=read(previous)
    assert baseline["dev_sha256"]==reg["dev_sha256"] and baseline["weights_sha256"]=="020574084dcfaa3efffca38d0c057419e0e90307548f06bfb62acad5be47946f"
    chosen=read(HERE/"dev"/(winner+".json"))["episodes"]
    save(HERE/"audit.json",dict(selected=selected,native_snapshots=snapshots,native_candidates=candidates,
         synthetic_hidden_wall_invariance=True,physically_replayed=400,results=results,training=archives,
         post_both_excursions=excursions,
         baseline=dict(file=str(previous.relative_to(ROOT)),sha256=digest(previous),summary=baseline["summary"],
                       paired=compare(chosen,baseline["episodes"]))))
    print("audit",winner,snapshots,candidates,"400 episodes replayed")


if __name__=="__main__":
    main()
