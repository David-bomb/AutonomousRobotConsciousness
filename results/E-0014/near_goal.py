"""Проверка выбора хода, когда дорога уже раскрыта в собственной памяти робота."""
from collections import Counter
from pathlib import Path

import torch

from maze.environment import Environment, Status
from maze.grid import destination, wave
from neural_rl.corpus_study.learn import setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import Network, batch
from neural_rl.perception import MOVES, encode
from neural_rl.stress_hundreds import ROOT, maps

OUT=Path(__file__).resolve().parent
STRESS=ROOT/"results/E-0013"


@torch.inference_mode()
def main():
    witnesses=read(STRESS/"witnesses.json")
    manifest=read(STRESS/"manifest.json")["models"]
    samples=[]
    for _,case in maps():
        env=Environment(case.level)
        actions=witnesses[case.name]["actions"]
        for index,code in enumerate(actions):
            left=len(actions)-index
            if left in (1,5):
                view=env.view
                goal=(view.position[0]+view.dx,view.position[1]+view.dy)
                memory=view.memory
                lengths=({p:d for p,d,_ in wave(memory,[goal])}
                         if memory[goal[1]][goal[0]] not in "#?" else {})
                known=lengths.get(view.position)
                viable=known is not None and known<=min(view.fuel,view.health)
                best=[i for i,m in enumerate(MOVES) if viable and
                      lengths.get(destination(memory,view.position,m)) == known-1]
                if left==1:
                    assert known==1 and best==[int(code)]
                samples.append((case.name,left,encode(view,False),known if viable else None,best))
            env.step(MOVES[int(code)])
        assert env.status==Status.SUCCESS
    results={}
    for name,model in manifest.items():
        if not model["frozen"]:
            continue
        setup(model["config"]["seed"])
        assert digest(ROOT/model["weights"])==model["sha256"]
        net=Network(model["config"]).cuda().eval()
        net.load_state_dict(torch.load(ROOT/model["weights"],map_location="cuda",weights_only=True))
        family=model["config"]["family"]
        finish={r["map"]:r["outcome"]=="SUCCESS" for r in read(OUT/(family+"-controls.json"))["last5"]}
        counts={1:Counter(),5:Counter()}
        rows=[]
        for i in range(0,len(samples),48):
            group=samples[i:i+48]
            choices=net(batch([v[2] for v in group]))[0].argmax(1).tolist()
            for (case,left,state,length,best),action in zip(group,choices):
                counts[left]["all"]+=1
                rows.append(dict(map=case,left=left,length=length,best=best,action=action))
                if length is not None:
                    counts[left].update(known_path=1,choose_shortest=int(action in best),
                                        uniform_expected=len(best)/sum(map(int,state.legal)))
                    if left==5:
                        counts[left]["finish_from_given_position"]+=int(finish[case])
        results[family]=dict(summary=counts,rows=rows)
        print(name,counts)
    save(OUT/"near-goal.json",results)


if __name__=="__main__":
    setup(0)
    main()
