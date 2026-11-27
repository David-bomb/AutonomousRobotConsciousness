"""Resume the locked transfer study in one process after Windows process startup failed."""
from hashlib import sha256
import shutil
from combination_trials import OUT,PLAN,freeze,initialize,score
from combination_policy import load
from early_trials import physical
from renewable_train import HERE,Environment,Status,data,digest,moves,save


def main():
    codes=freeze()
    groups=[data.rows(OUT/(str(s)+".jsonl")) for s in data.read(PLAN)["seeds"]]
    save(OUT/"serial-recovery.json",dict(prior_lock=digest(OUT/"lock.json"),runner=digest(HERE/"combination_serial.py"),
        reason="Windows ProcessPoolExecutor raised BrokenProcessPool at child startup. Execute the same locked score(row) in one process with the same policy initialization and ranking."))
    initialize()
    for i,pair in enumerate(zip(*groups,strict=True)):
        for row in pair:score(row)
        if (i+1)%10==0:print(2*(i+1),"paired maps",flush=True)
    table={}
    for code in codes:
        counts=[];h4=0
        for group in groups:
            wins=[]
            for r in group:
                e=data.read(OUT/"rows"/(r["name"]+".json"))
                assert e["lock"]==digest(OUT/"lock.json") and e["geometry"]==r["geometry"]
                if e["results"][code]["outcome"]=="SUCCESS":wins.append(r)
            counts.append(len(wins));h4+=sum(r["tier"]=="H4" for r in wins)
        table[code]=dict(counts=counts,h4=h4,ranking=[min(counts),sum(counts),h4,-codes.index(code)])
    winner=max(codes,key=lambda code:table[code]["ranking"])
    model=OUT/"model.pt"
    if not model.exists():
        tmp=model.with_suffix(".tmp");shutil.copyfile(OUT/winner/"model.pt",tmp);tmp.replace(model)
    assert digest(model)==digest(OUT/winner/"model.pt")
    save(OUT/"frozen.json",dict(winner=winner,record=table,model_sha256=digest(model),lock_sha256=digest(OUT/"lock.json")))
    checks=[]
    for group in groups:
        for tier in ("H1","H2","H3","H4"):
            r=next(row for row in group if row["tier"]==tier);env=Environment(data.level(r))
            net=load(model.read_bytes());cache={};commands=bytearray()
            while env.status==Status.RUNNING:
                action=net.choose(env.view,cache);commands.append(moves.index(action));env.step(action)
            rec=data.read(OUT/"rows"/(r["name"]+".json"))["results"][winner];n=rec["prefix"]
            assert (env.status==Status.SUCCESS)==(rec["outcome"]=="SUCCESS")
            assert sha256(commands[:n]).hexdigest()==rec["prefix_sha256"]
            if rec["outcome"]!="exact_cycle":
                assert env.status.name==rec["outcome"] and env.steps==n+rec["suffix"]
                token=sha256(env.status.name.encode()).digest()
                for a in reversed(commands[n:]):token=sha256(bytes([a])+token).digest()
                assert token.hex()==rec["suffix_chain_sha256"]
            checks.append(dict(map=r["name"],outcome=env.status.name,steps=env.steps))
    save(OUT/"replay.json",dict(checks=checks,model_sha256=digest(model)))
    for tag,ablate in (("dev",False),("ablated",True)):physical(OUT,tag,load(model.read_bytes(),ablate))


if __name__=="__main__":main()
