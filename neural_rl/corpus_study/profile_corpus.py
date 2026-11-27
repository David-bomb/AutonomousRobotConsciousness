"""Сверка состава выборок, размеров и воспроизводимости генерации E-0012."""
import argparse
from collections import Counter
import json
import sqlite3
import time
import zlib

from .corpus import DB, SPLITS, TARGETS, design, make
from .geometry import FAMILIES, SIZES
from .report import table, write_text
from .session import OUT, check_time, log
from .transfer import make as make_shift


def profile(wait):
    with sqlite3.connect(DB.as_uri()+"?mode=ro", uri=True) as db:
        while True:
            check_time()
            counts = dict(db.execute("SELECT split,count(*) FROM geoms GROUP BY split"))
            if counts == TARGETS:
                break
            if not wait:
                raise RuntimeError("Ещё не все выборки готовы")
            time.sleep(20)
        cells = {limit: Counter() for limit in (300, 3000, 30000, 60000)}
        groups = {split: Counter() for split in SPLITS}
        needs = {split: Counter() for split in SPLITS}
        reproduced = {}
        geometries = 0
        for row in db.execute("SELECT split,idx,hash,family,size,seed,group_name,data FROM geoms ORDER BY split,idx"):
            split,index,_,family,size,seed,group,data = row
            if index % 100 == 0:
                check_time()
            expected, width, label = design(split,index)
            height = width + 4 if split == "shift_development" and group in ("parameters","combined") else width
            if (family,group,size) != (expected,label,max(height,width)):
                raise ValueError(f"Не совпадает тип или размер {split}/{index}")
            cases = json.loads(zlib.decompress(data))
            if [r["mode"] for r in cases] != [0,1,2,3]:
                raise ValueError(f"Неполные ресурсные варианты {split}/{index}")
            for case in cases:
                title, stocks, *grid = case["source"].splitlines()
                if len(grid)!=height or any(len(line)!=width for line in grid):
                    raise ValueError(f"Не совпали фактические размеры {title}")
                fuel,health = map(int, stocks.split())
                need = (fuel < case["direct"]) + 2*(health < case["direct"])
                needs[split][need] += 1
            if split == "train":
                for limit, matrix in cells.items():
                    if index < limit:
                        matrix[family,width] += 1
            bucket = (group,family,width,height)
            groups[split][group] += 1
            key = (split,*bucket)
            if key not in reproduced:
                if split == "shift_development":
                    rebuilt = make_shift((index, seed-8*10**11-index*1000))
                else:
                    attempt = seed-(1+SPLITS.index(split))*10**11-index*1000
                    rebuilt = make((split,index,attempt))
                if rebuilt != row:
                    raise ValueError(f"Не воспроизвёлся генератор {split}/{index}")
                reproduced[key] = index
            geometries += 1
        for limit,matrix in cells.items():
            if matrix != {(f,s):limit//30 for f in FAMILIES for s in SIZES}:
                raise ValueError(f"Нарушен баланс учебной выборки {limit}")
        if groups["shift_development"] != dict.fromkeys(("size","parameters","services","resources","combined"),300):
            raise ValueError("Нарушен состав умеренного переноса")
        if groups["test_ood"] != dict.fromkeys(("size","generator","topology","services","resources","combined"),1000):
            raise ValueError("Нарушен состав проверки новых условий")
        if any(sum(needs[s].values())!=4*TARGETS[s] for s in SPLITS):
            raise ValueError("Не совпало число сценариев")
    indices = [dict(split=k[0],group=k[1],family=k[2],width=k[3],height=k[4],index=i) for k,i in reproduced.items()]
    record = dict(geometries=geometries,scenarios=4*geometries,groups=groups,checked_seeds=indices,
                  train_counts={n:{f:[cells[n][f,s] for s in SIZES] for f in FAMILIES} for n in cells},
                  train_sizes=SIZES,necessary_by_direct_path=needs)
    write_text(OUT/"corpus-profile.json",json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    balance = [[n,6,5,n//30,n*4] for n in cells]
    stocks = [[s,*[needs[s][i] for i in range(4)]] for s in SPLITS]
    write_text(OUT/"corpus-profile.md","# Состав корпуса E-0012\n\n"
               f"Прочитаны все {geometries:,} рисунков стен и все {4*geometries:,} сценариев. "
               f"Размеры проверены по строкам каждого уровня. Из сохранённых зёрен заново получены "
               f"{len(indices)} записи, по одной на каждое сочетание выборки, типа переноса, генератора и размера. "
               "Совпали точные карты, маршруты, результаты финиша и сжатые записи.\n\n"+
               table(["Учебных геометрий","Типов","Размеров","На каждую пару типа и размера","Сценариев"],balance)+
               "\n\nУмеренный перенос содержит пять групп по 300 рисунков. Проверка новых условий содержит шесть групп по 1 000 рисунков.\n\n"
               "В таблице ниже начальные запасы сравниваются с длиной кратчайшего пути до финиша. "
               "Если запас меньше этой длины, соответствующее обслуживание необходимо. Объезд может потребовать дополнительного обслуживания.\n\n"+
               table(["Набор","Оба запаса покрывают длину","Мало топлива","Мало исправности","Малы оба запаса"],stocks)+
               "\n\n[Зёрна и распределение по группам](corpus-profile.json).\n")
    log("corpus_composition_verified",geometries=geometries,scenarios=4*geometries,reproduced=len(indices))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--when-ready",action="store_true")
    profile(parser.parse_args().when_ready)
