"""Learn how strongly to use inferred remaining routes once safe exploration is exhausted."""
from hashlib import sha256
import numpy as np
import torch
from renewable_train import EXP,HERE,Environment,Status,data,digest,moves,save
from renewable_policy import targets
from final_leg_train import dataset
from early_trials import physical
from latent_finish_policy import LatentFinish,load

OUT=HERE/"latent-finish"


def check(net):
    source=HERE/"renewable-search/exact-check.json"
    reference={}
    for name,move,alive,length in data.read(source)["cases"]:
        reference.setdefault(name,{})[move]=length if alive else None
    checked=0
    for r in data.rows(HERE/"return-search/bank.jsonl"):
        env=Environment(data.level(r));v=env.view;mem=env.level.grid;w=len(mem[0]);x,y=v.position
        expected=reference[r["name"]];best=min(z for z in expected.values() if z is not None)
        measured=np.empty((8,5));p=y*w+x
        net.worlds.latent_finish("".join(mem).encode("ascii"),w,len(mem),p,p,targets(mem,env.level.goal),
                np.array([v.fuel_distance,v.repair_distance],np.int32),np.array([-1,-1],np.int32),
                v.fuel,v.health,1,measured)
        for a,m in enumerate(moves):
            if m.name not in expected:continue
            length=expected[m.name]
            assert bool(measured[a,0])==(length is not None) and bool(measured[a,1])==(length==best)
            if length is not None:assert np.isclose(measured[a,3],np.exp(-(length-1)/55))
            checked+=1
    save(OUT/"full-map-check.json",dict(training_maps=len(reference),legal_actions=checked,
          exact_path_matched=True,reference_sha256=digest(source),native_sha256=digest(HERE/"latent_finish.dll")))


def learn(tag,rows,options,net):
    folder=OUT/tag;folder.mkdir(exist_ok=True);total=np.zeros(len(options),int);episodes=[]
    for i,r in enumerate(rows):
        path=folder/(r["name"]+".json")
        if path.exists():
            blob=data.read(path);assert blob["options"]==options.tolist();group=blob["episodes"]
        else:
            env=Environment(data.level(r));old=(None,None);prefix=[];known=set()
            while env.status==Status.RUNNING and not net.route(env.view,net.cfg):
                v=env.view;key=v,old
                if key in known:break
                known.add(key);a=int(net.early_scores(v,old).argmax());prefix.append(a)
                old=(v.fuel_distance,v.repair_distance);env.step(moves[a])
            reached=env.status==Status.RUNNING and net.route(env.view,net.cfg)
            from copy import copy
            group=[];cache={}
            for j in range(len(options)):
                branch=copy(env);branch._memory=copy(env._memory);past=old;seen=set();tail=bytearray()
                while reached and branch.status==Status.RUNNING:
                    v=branch.view;k=v,past
                    if k in seen:break
                    seen.add(k)
                    if k not in cache:
                        q,x=net.late_parts(v,past);cache[k]=(q[None,:]+options@x.T).argmax(1)
                    a=int(cache[k][j]);tail.append(a);past=(v.fuel_distance,v.repair_distance);branch.step(moves[a])
                group.append(dict(map=r["name"],tier=r["tier"],head=j,gate=reached,steps=branch.steps,
                    outcome=branch.status.name if branch.status!=Status.RUNNING else "exact_cycle",
                    actions_sha256=sha256(bytes(prefix)+tail).hexdigest()))
            save(path,dict(options=options.tolist(),episodes=group,protocol_sha256=digest(OUT/"protocol.json")))
        total+=np.array([e["outcome"]=="SUCCESS" for e in group]);episodes.extend(group)
        if (i+1)%10==0:print(tag,i+1,total.tolist(),flush=True)
    save(OUT/(tag+".json"),dict(options=options.tolist(),wins=total.tolist(),episodes=episodes))
    return total


def main():
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    parent=HERE/"service-window/fixed/model.pt"
    assert digest(parent)=="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c"
    pack=torch.load(parent,weights_only=True,map_location="cpu")
    pack.update(format="latent-finish-1",samples=48,coeff=torch.zeros(7,dtype=torch.float64))
    train,_=dataset()
    bank=EXP/"sequence/spatial/large.jsonl";selection=data.rows(bank)[2400:2500]
    assert not ({r["geometry"] for r in train}&{r["geometry"] for r in selection})
    options=np.array([[0]*7,[8,0,2,0,0,0,0],[8,0,6,0,0,0,0],[0,2,4,2,0,0,0],[0,0,12,1,0,0,0],
        [8,4,6,1,1,1,0],[8,2,8,0,1,2,1],[0,6,3,2,0,0,1],[0,2,5,0,0,0,2]],float)
    save(OUT/"protocol.json",dict(parent=digest(parent),fit=[r["name"] for r in train],select=[r["name"] for r in selection],
        select_bank=digest(bank),settings=options.tolist(),samples=pack["samples"],
        hashes={n:digest(HERE/n) for n in ("latent_finish.cpp","latent_finish.dll","origin_worlds.cpp","belief_worlds.cpp",
               "latent_finish_policy.py","latent_finish_train.py","proper_frontier.dll","early_trials.py")},
        rule="Guard known safe observation routes. Where none remains, sample unseen roads from ordinary cell memory. For every real first move measure shortest physically sustainable goal route on each hypothesis, weighting by agreement with current/previous gauges. Terminal successes on earlier late-failure cohort; top three nonzero heads, their mean, and zero on 100 other train starts. Freeze before complete physical dev and ablation."))
    model=LatentFinish(pack);check(model)
    a=learn("fit",train,options,model)
    top=options[np.argsort(-a[1:],kind="stable")[:3]+1];final=np.vstack((options[0],top,top.mean(0)))
    b=learn("select",selection,final,model)
    theta=final[b.argmax()];p=OUT/"model.pt"
    if not p.exists():
        tmp=p.with_suffix(".tmp");torch.save(dict(pack,coeff=torch.tensor(theta,dtype=torch.float64)),tmp);tmp.replace(p)
    save(OUT/"frozen.json",dict(coefficients=theta.tolist(),model_sha256=digest(p),fit=a.tolist(),select=b.tolist(),
                               protocol_sha256=digest(OUT/"protocol.json")))
    for tag,ablate in (("dev",False),("ablated",True)):
        physical(OUT,tag,load(p.read_bytes(),ablate))


if __name__=="__main__":main()
