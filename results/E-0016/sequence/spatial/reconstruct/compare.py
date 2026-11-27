from ....data import digest, read, save
from ..rooms.physical import summarize
from .bank import HERE, RUN, DEV


def pair(a,b):
    assert len(a)==len(b)
    assert all(x["map"]==y["map"] for x,y in zip(a,b))
    return dict(gain=sum(x["outcome"]!="SUCCESS" and y["outcome"]=="SUCCESS" for x,y in zip(a,b)),
                loss=sum(x["outcome"]=="SUCCESS" and y["outcome"]!="SUCCESS" for x,y in zip(a,b)),
                changed_record=sum(x!=y for x,y in zip(a,b)))


def train_comparison(record,labels,baseline):
    columns={k:[dict(ep["episodes"][i],map=ep["map"],tier=ep["tier"]) for ep in record["records"]]
             for i,k in enumerate(labels)}
    return {k:dict(summary=summarize(items),paired_to_missing=pair(columns[baseline],items)) for k,items in columns.items()}


def main():
    dagger=RUN.parent/"reconstruct-dagger200"
    original=read(RUN/"missing.physical.json")
    sig=digest(DEV)
    methods={}
    for name in ("missing","marginal","direct","marginal1","direct1"):
        path=RUN/(name+".physical.json")
        record=read(path)
        assert record["dev_sha256"]==sig and record["weights_sha256"]==digest(RUN/(name+".pt"))
        methods[name]=dict(summary=record["summary"],paired_to_missing=pair(original["episodes"],record["episodes"]),
                           weights=record["weights_sha256"],report=digest(path))
    shadow=read(dagger/"rl.physical.json")
    previous=read(RUN/"rl.physical.json")
    identical=shadow["weights_sha256"]==previous["weights_sha256"]
    if identical:assert shadow["episodes"]==previous["episodes"]
    first=read(HERE/"registration.json")["selection_order"]
    later=("previous","marginal","direct","probability")
    save(RUN/"comparison.json",dict(dev_sha256=sig,dev=methods,selected=read(RUN/"locked-before-dev.json")["chosen"],
          selection=train_comparison(read(RUN/"selection.json"),first,"missing"),
          dagger_selection=train_comparison(read(dagger/"selection.json"),later,"previous"),
          dagger=dict(weights=shadow["weights_sha256"],selected=read(dagger/"locked-before-dev.json")["name"],
                      exact_repeat=100 if identical else None,summary=shadow["summary"])))
    print({k:(v["summary"]["wins"],v["paired_to_missing"]) for k,v in methods.items()},flush=True)


if __name__=="__main__":main()
