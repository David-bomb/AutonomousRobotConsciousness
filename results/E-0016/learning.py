"""Парные результаты полного dev. Каждый элемент — та же независимая геометрия."""
from .data import OUT, read, save


def main():
    runs=OUT/"runs"
    pairs=[("pilot200/initial","pilot200/imitation"),("pilot200/imitation","pilot200/rl"),
           ("pilot200/rl","recovery800/imitation"),("recovery800/imitation","recovery800/rl"),
           ("recovery800/rl","critical800/imitation"),("critical800/imitation","critical800/rl"),
           ("rich800/imitation","rich800/rl1"),("rich800/rl1","rich800/rl2")]
    result=[]
    for before,after in pairs:
        a=runs/(before+".physical.json")
        b=runs/(after+".physical.json")
        if not a.exists() or not b.exists():
            continue
        first,second=read(a),read(b)
        assert first["dev_sha256"]==second["dev_sha256"]
        old={r["map"]:r["outcome"]=="SUCCESS" for r in first["episodes"]}
        new={r["map"]:r["outcome"]=="SUCCESS" for r in second["episodes"]}
        assert old.keys()==new.keys() and len(old)==100
        result.append(dict(before=before,after=after,wins_before=sum(old.values()),wins_after=sum(new.values()),
                           gained=[k for k in old if new[k] and not old[k]],lost=[k for k in old if old[k] and not new[k]]))
    save(OUT/"learning.json",result)
    print([(r["after"],len(r["gained"]),len(r["lost"])) for r in result])


if __name__=="__main__":
    main()
