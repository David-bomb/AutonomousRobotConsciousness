"""Проверка итоговых артефактов и независимый повтор выбранных поездок без пропуска циклов."""
from collections import Counter
import argparse
import gzip
import json
import sqlite3
import time

from maze.environment import Status
from neural_rl.experiment import digest,read,save
from .corpus import DB,MOVES,TARGETS
from .diagnostics import FOLDER
from .final_eval import SIZES,shortlist
from .learn import DirectNetwork,FAMILIES,Pool,metrics,setup
from .report import generate,load_runs,write_text
from .session import OUT,ROOT,log,check_time


def validate_training(runs,manifest):
    by_id={r["config"]["id"]:r for r in runs}
    hashes=set()
    screen_names={f"validation_id-{i:05}-{j}" for i in range(60) for j in range(4)}
    prior=read(ROOT/"results/E-0011/manifest.json")["sources"]
    for name,signature in manifest["source_sha256"].items():
        assert digest(ROOT/name)==signature, name
        if name.startswith("maze/"):
            assert prior[name]==signature, name
    assert len(by_id)==len(manifest["runs"])==200
    for c in manifest["runs"]:
        r=by_id[c["id"]]
        assert r["config"]==c and r["parameter_delta"]>0
        assert r["train_steps"]==manifest["primary_train_steps"]==3*c["lanes"]*c["ticks"]
        assert len(r["training"])==3 and all(s["updates"]>0 for s in r["training"])
        visited=set()
        for s in r["training"]:
            assert s["steps"]==c["lanes"]*c["ticks"]
            for name in s["maps"]:
                split,i,j=name.rsplit("-",2)
                assert split=="train" and 0<=int(i)<c["geometry_count"] and 0<=int(j)<4
            visited.update(s["maps"])
        assert len(visited)==r["train_scenarios"]
        folder=OUT/"runs"/c["id"]
        assert digest(folder/"weights.pt")==r["weights_sha256"]
        hashes.add(r["weights_sha256"])
        screens=read(folder/"validation.json")
        for key,field in (("before","initial"),("after","screen")):
            rows=screens[key]
            assert len(rows)==240 and {x["map"] for x in rows}==screen_names
            assert metrics(rows)==r[field]
    assert len(hashes)==200


def raw_results(model,split):
    rows=[]
    folder=OUT/"evaluation"/split/model
    expected_hash=read(OUT/"runs"/model/"run.json")["weights_sha256"]
    for p in sorted(folder.glob("*.json.gz")):
        with gzip.open(p,"rt",encoding="utf-8") as stream:
            saved=json.load(stream)
        assert saved["weights_sha256"]==expected_hash
        rows.extend(saved["rows"])
    assert len(rows)==4*SIZES[split]
    expected={f"{split}-{i:05}-{j}" for i in range(SIZES[split]) for j in range(4)}
    assert {r["map"] for r in rows}==expected
    summary=read(folder/"summary.json")
    assert (summary["id"],summary["split"],summary["weights_sha256"])==(model,split,expected_hash)
    assert all(summary[k]==v for k,v in metrics(rows).items())
    return rows


def full_steps(net,chosen):
    import torch
    from neural_rl.networks import batch
    from neural_rl.perception import encode
    active=[]
    for split,r in chosen:
        i,j=map(int,r["map"].rsplit("-",2)[1:])
        pool=Pool(split,1,offset=i)
        active.append((pool.start(pool.cases[j]),r))
        pool.db.close()
    net.eval()
    verified=[]
    with torch.no_grad():
        while active:
            check_time()
            actions=net(batch([encode(env.view,False) for env,_ in active]))[0].argmax(1).tolist()
            following=[]
            for (env,record),a in zip(active,actions):
                env.step(MOVES[a])
                if env.status==Status.RUNNING:
                    following.append((env,record))
                else:
                    assert (env.status.name,env.steps,env.refuels,env.repairs)==tuple(
                        record[k] for k in ("outcome","steps","refuels","repairs")),record["map"]
                    verified.append(record["map"])
            active=following
    return verified


def main():
    import torch
    runs=load_runs()
    manifest=read(OUT/"registration.json")
    validate_training(runs,manifest)
    short=shortlist(runs)
    assert short==read(OUT/"shortlist.json")
    frozen=read(OUT/"frozen.json")
    audit_events=[json.loads(line) for line in (OUT/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
    freezes=[i for i,e in enumerate(audit_events) if e["event"]=="models_frozen"]
    assert freezes
    first_freeze=freezes[0]
    assert all(audit_events[i]["families"]==frozen for i in freezes)
    selected={m for pair in short.values() for m in pair}
    full_before={e["id"] for e in audit_events[:first_freeze] if e["event"]=="evaluation_completed"
                 and e["split"]=="validation_id" and e["scenarios"]==12000}
    assert full_before==selected
    assert not any(e["event"] in ("evaluation_progress","evaluation_completed") and e["split"]!="validation_id"
                   for e in audit_events[:first_freeze])
    profile=read(OUT/"corpus-profile.json")
    assert profile["geometries"]==76500 and profile["scenarios"]==306000
    diagnostic_audit=read(FOLDER/"audit.json")
    assert all(diagnostic_audit[k]["scenarios"]==1000 for k in ("train","control"))
    with sqlite3.connect(DB.as_uri()+"?mode=ro",uri=True) as db:
        assert dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))==TARGETS
        main_keys={r[0] for r in db.execute("SELECT hash FROM geoms")}
        assert db.execute("SELECT count(*) FROM geoms WHERE split='shift_development' AND "
                          "seed NOT BETWEEN 800000000000+1000*idx AND 800000000999+1000*idx").fetchone()==(0,)
    diag=read(FOLDER/"verified.json")
    banks={}
    for name in ("train","control"):
        f=FOLDER/f"{name}.jsonl.gz"
        assert digest(f)==diag["sha256"][name]
        with gzip.open(f,"rt",encoding="utf8") as stream: records=[json.loads(l) for l in stream]
        assert Counter(r["group"] for r in records)=={i:50 for i in range(1,21)}
        banks[name]={r["geometry"] for r in records}
        assert banks[name].isdisjoint(main_keys)
    assert banks["train"].isdisjoint(banks["control"])
    repeats={}
    diagnostic_steps=0
    full_val=closed=0
    for family in FAMILIES:
        for model in short[family]:
            full_val+=len(raw_results(model,"validation_id"))
        def validation_order(candidate):
            score=read(OUT/"evaluation"/"validation_id"/candidate/"summary.json")
            config=read(OUT/"runs"/candidate/"run.json")["config"]
            length=score["mean_success_steps"]
            return score["worst"],score["wins"],-(float("inf") if length is None else length),-config["variant"],-config["seed"]
        model=frozen[family]["id"]
        assert max(short[family],key=validation_order)==model
        assert digest(OUT/"runs"/model/"weights.pt")==frozen[family]["sha256"]
        pool_results=[]
        for split in ("shift_development","test_id","test_ood"):
            rows=raw_results(model,split)
            closed+=len(rows)
            # Берём случай каждого фактически наблюдаемого исхода. Для долгих циклов
            # самый короткий эпизод данного исхода по-прежнему требует обычных шагов.
            for status in sorted({r["outcome"] for r in rows}):
                candidate=min((r for r in rows if r["outcome"]==status),key=lambda r:(r["steps"],r["map"]))
                pool_results.append((split,candidate))
        config=read(OUT/"runs"/model/"run.json")["config"]
        setup(config["seed"])
        net=DirectNetwork(config).cuda()
        net.load_state_dict(torch.load(OUT/"runs"/model/"weights.pt",weights_only=True,map_location="cuda"))
        repeats[model]=full_steps(net,pool_results)
        diagnostic=read(FOLDER/f"score-{family}.json")
        assert diagnostic["id"]==model and sum(len(item["rows"]) for item in diagnostic["groups"].values())==1000
        folder=FOLDER/"training"/family
        result=read(folder/"run.json")
        scored=read(folder/"control.json")
        assert result["base"]==frozen[family]
        assert result["sha256"]==digest(folder/"weights.pt")==scored["weights_sha256"]
        assert set(map(int,result["coverage"]))==set(range(1,21)) and result["steps"]==73728
        assert len(scored["rows"])==len({r["map"] for r in scored["rows"]})==1000
        diagnostic_steps+=result["steps"]
    assert any(e["event"]=="witness_audit_completed" and e["all_splits"] and e["verified"]==306000 for e in audit_events)
    finished=[r for r in audit_events if r["event"]=="trial_completed" and not r["technical"]]
    assert len(finished)==200 and len({r["id"] for r in finished})==200
    report=dict(weights=200,initial_and_final_screen=200*240*2,full_validation_rows=full_val,
                selected_evaluation_rows=closed,physical_replays=repeats,corpus_certificates=306000,diagnostic_certificates=2000,
                additional_diagnostic_training_steps=diagnostic_steps)
    save(OUT/"verification.json",report)
    write_text(OUT/"verification.md","## Контроль результата\n\n"
               f"Сверены 200 конфигураций и 200 разных контрольных сумм весов, 96 000 покартовых записей до и после, "
               f"{full_val:,} строк полной val и {closed:,} строк по выбранным моделям. "
               f"На {sum(map(len,repeats.values()))} контрольных поездках из сохранённых весов совпали исход, физические ходы и оба счётчика обслуживания. "
               "Для этого повтора симулятор исполнил каждый физический ход. Хеши модулей оболочки совпали с E-0011. "
               "[Протокол сверки](verification.json).\n")
    log("final_verification",**{k:v for k,v in report.items() if k!="physical_replays"},
        repeat_episodes=sum(map(len,repeats.values())))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--when-ready",action="store_true")
    args=parser.parse_args()
    if args.when_ready:
        while True:
            check_time()
            audit=OUT/"audit-summary.json"
            if (audit.is_file() and read(audit)["all_splits"] and
                    (FOLDER/"training-results.md").is_file() and (OUT/"final-results.md").is_file()):
                break
            time.sleep(20)
    main()
    generate()
