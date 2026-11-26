"""Внешняя проверка согласия гипотез в первом расхождении streets и range-roads."""
from collections import Counter
import json

from bench import HERE,MOVES,cases
from compare import read
from maze.environment import Environment
from maze.pattern_roads import PatternRoads,pattern_votes


class Observer(PatternRoads):
    capture=False

    def rank_directions(self,view,readings,old,prior,sites,options,kind,default):
        result=super().rank_directions(view,readings,old,prior,sites,options,kind,default)
        if self.capture:
            groups=pattern_votes(view.memory,view.position,readings,old,prior,sites,split_votes=True)
            ranks=[max(options,key=lambda m:(group[kind][m],m==default)) for group in groups]
            self.evidence=dict(kind="FR"[kind],choices=[m.name for m in options],
                               halves=[m.name for m in ranks],agree=ranks[0]==ranks[1] and
                               all(group[kind][ranks[0]]>0 for group in groups))
        return result


def main():
    pair=[json.loads(s) for s in (HERE/"range-roads--streets.jsonl").read_bytes().splitlines()]
    changed={r["map"]:r for r in pair if r["verdict"] in ("приобретён","потерян")}
    histories=read("streets")
    counts=Counter()
    with (HERE/"streets-confidence.jsonl").open("x",encoding="utf8") as out:
        for _,name,*_,level in cases():
            if name not in changed:
                continue
            r=changed[name]
            env=Environment(level)
            bot=Observer()
            for t,c in enumerate(histories[name]["actions"][:r["first"]["step"]],1):
                bot.capture=t==r["first"]["step"]
                move=bot.choose(env.view)
                assert move==MOVES[int(c)]
                env.step(move)
            counts[r["verdict"],bool(bot.evidence["agree"])]+=1
            out.write(json.dumps(dict(map=name,verdict=r["verdict"],**bot.evidence),
                                 ensure_ascii=False,default=bool)+"\n")
            out.flush()
    print(counts)


if __name__=="__main__":
    main()
