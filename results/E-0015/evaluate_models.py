"""Проверка конечных весов на одинаковых развилках, валидации и трёх сотнях."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import gzip
from hashlib import sha256
import json

import numpy as np
import torch

from maze.environment import Environment, Status
from neural_rl.corpus_study.diagnostic_eval import DiagnosticPool, expected_moves
from neural_rl.corpus_study.diagnostics import probe_view
from neural_rl.corpus_study.learn import Chunk, Pool, setup
from neural_rl.experiment import digest, read, save
from neural_rl.networks import batch
from neural_rl.perception import MOVES, encode
from neural_rl.rollout import MapPool, evaluate
from neural_rl.stress_hundreds import ROOT, SOURCE, SPLITS, bench, maps
from .train import OUT, REG, network


class FromFork(DiagnosticPool):
    def __init__(self, records):
        super().__init__(records)
        self.prefix={r["name"]:r["prefix"] for r in records}

    def start(self, case):
        env=super().start(case)
        for code in self.prefix[case.name]:
            env.step(MOVES[int(code)])
        assert env.status==Status.RUNNING
        return env


def paired_decisions(net, records):
    result=[]
    assert len(records)==50
    for a,b in zip(records[::2],records[1::2]):
        assert a["pair"]==b["pair"] and a["group"]==b["group"]
        views=[probe_view(r) for r in (a,b)]
        va,vb=views
        assert replace(va,memory=vb.memory)==vb
        assert (va.memory!=vb.memory)==(a["group"]==4)
        states=[encode(v,False) for v in views]
        assert np.array_equal(states[0].numbers,states[1].numbers)
        assert np.array_equal(states[0].legal,states[1].legal)
        assert np.array_equal(states[0].image,states[1].image)==(a["group"]==11)
        q=net(batch(states))[0].tolist()
        choices=np.argmax(q,axis=1).tolist()
        targets=[expected_moves(r,v) for r,v in zip((a,b),views)]
        assert all(len(t)==1 for t in targets) and targets[0]!=targets[1]
        left,right=[t[0] for t in targets]
        if a["group"]==11:
            assert choices[0]==choices[1]
        result.append(dict(pair=a["pair"],maps=[a["name"],b["name"]],choices=choices,expected=[left,right],
                           q_values=[[v if ok else None for v,ok in zip(row,s.legal)] for row,s in zip(q,states)],
                           correct=int(choices[0]==left)+int(choices[1]==right),
                           changed=choices[0]!=choices[1],
                           margin_shift=(q[0][left]-q[0][right])-(q[1][left]-q[1][right])))
    assert len({r["pair"] for r in result})==25
    return result


def replay(net, levels, stress):
    selected=[r for r in stress if r["outcome"]=="SUCCESS"]
    for split in SPLITS:
        failures=[r for r in stress if r["split"]==split and r["outcome"]!="SUCCESS"]
        if failures:
            selected.append(min(failures,key=lambda r:(r["steps"],r["map"])))
    verified={}
    active=[(r,Environment(levels[r["map"]]),[]) for r in selected]
    while active:
        actions=net(batch([encode(env.view,False) for _,env,_ in active]))[0].argmax(1).tolist()
        pending=[]
        for (r,env,path),a in zip(active,actions):
            env.step(MOVES[a])
            path.append(str(a))
            if env.status==Status.RUNNING:
                pending.append((r,env,path))
            else:
                assert (env.status.name,env.steps,env.refuels,env.repairs)==tuple(
                    r[k] for k in ("outcome","steps","refuels","repairs")), r["map"]
                verified[r["map"]]="".join(path)
        active=pending
    return verified


@torch.inference_mode()
def main():
    reg=read(REG)
    assert reg["stress_maps"]=={s:sha256(bench().SETS[s].read_bytes().replace(b"\r\n",b"\n")).hexdigest() for s in SPLITS}
    path=SOURCE/"diagnostics/control.jsonl.gz"
    assert digest(path)==reg["control_sha256"]
    with gzip.open(path,"rt",encoding="utf8") as stream:
        records=[json.loads(line) for line in stream]
    forks=[r for r in records if r["group"]==4]
    identical=[r for r in records if r["group"]==11]
    cases=maps()
    split_for={c.name:s for s,c in cases}
    assert len(split_for)==300 and Counter(split_for.values())==dict.fromkeys(SPLITS,100)
    stress_pools=[MapPool([c for _,c in cases[i:i+50]]) for i in range(0,300,50)]
    levels={c.name:c.level for _,c in cases}
    val=Pool("validation_id",60)
    for config in reg["configs"]:
        for arm in reg["arms"]:
            run=OUT/"runs"/config["id"]/arm
            job=read(run/"training.json")
            assert job["sha256"]==digest(run/"weights.pt") and job["config"]==config
            setup(config["seed"])
            net=network(arm,config,reg["input_canvas"]).eval()
            net.load_state_dict(torch.load(run/"weights.pt",map_location="cuda",weights_only=True))
            stress=[r for pool in stress_pools for r in evaluate(net,pool,1)]
            assert len(stress)==300 and {r["map"] for r in stress}==set(split_for)
            for r in stress:
                r["split"]=split_for[r["map"]]
            val_rows=[r for i in range(0,240,32) for r in evaluate(net,Chunk(val,val.cases[i:i+32]),1)]
            assert len(val_rows)==len({r["map"] for r in val_rows})==240
            if arm=="cnn" and job["archive_max_abs_difference"]==0:
                archived=read(ROOT/"results/E-0013/models"/(config["id"]+".json"))["rows"]
                assert sorted(stress,key=lambda r:r["map"])==sorted(archived,key=lambda r:r["map"])
                assert val_rows==read(SOURCE/"runs"/config["id"]/"validation.json")["after"]
            decisions=paired_decisions(net,forks)
            uniform=None
            if arm=="query":
                flat=deepcopy(net)
                flat.query.weight.zero_()
                flat.query.bias.zero_()
                uniform=paired_decisions(flat,forks)
            fork_outcomes=evaluate(net,FromFork(forks),1)
            assert {r["map"] for r in fork_outcomes}=={r["name"] for r in forks}
            save(run/"evaluation.json",dict(sha256=job["sha256"],forks=decisions,
                 indistinguishable=paired_decisions(net,identical),uniform_read=uniform,
                 fork_episodes=fork_outcomes,validation=val_rows,stress=stress,
                 checked_paths=replay(net,levels,stress)))
            print(config["id"],arm,"evaluated",flush=True)
    val.db.close()


if __name__=="__main__":
    main()
