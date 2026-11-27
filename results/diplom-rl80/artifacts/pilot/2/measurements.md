# Полные валидационные поездки E-0016

Каждая строка — одни закреплённые веса на одних и тех же 100 новых картах dev. Во всех эпизодах модель использует обычное наблюдение, стартует с исходными запасами и движется до штатного исхода. H1–H4 содержат по 25 карт. Выбор весов по этому набору делает его выборкой разработки. Оценка на I, II и III ведётся отдельно.

| Запуск | Веса | Финиши / 100 | H1 | H2 | H3 | H4 | Посетил оба сервиса | Покартовый журнал |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pilot200 | [initial](../../../../University/Diplom/results/E-0016/runs/pilot200/initial.pt) | 0 | 0 | 0 | 0 | 0 | 0 | [JSON](../../../../University/Diplom/results/E-0016/runs/pilot200/initial.physical.json) |
| pilot200 | [imitation](../../../../University/Diplom/results/E-0016/runs/pilot200/imitation.pt) | 21 | 8 | 5 | 6 | 2 | 38 | [JSON](../../../../University/Diplom/results/E-0016/runs/pilot200/imitation.physical.json) |
| pilot200 | [rl](../../../../University/Diplom/results/E-0016/runs/pilot200/rl.pt) | 20 | 7 | 6 | 5 | 2 | 41 | [JSON](../../../../University/Diplom/results/E-0016/runs/pilot200/rl.physical.json) |
| recovery800 | [imitation](../../../../University/Diplom/results/E-0016/runs/recovery800/imitation.pt) | 29 | 16 | 7 | 4 | 2 | 43 | [JSON](../../../../University/Diplom/results/E-0016/runs/recovery800/imitation.physical.json) |
| recovery800 | [rl](../../../../University/Diplom/results/E-0016/runs/recovery800/rl.pt) | 28 | 17 | 5 | 5 | 1 | 43 | [JSON](../../../../University/Diplom/results/E-0016/runs/recovery800/rl.physical.json) |

Корпус dev, SHA-256 · 4ce3fcd9106332497bf6a6e6ad26040aae303fc297a77dc7606828c5edabe8a6

Таблица пересчитывается командой из корня лаборатории.

```powershell
python projects/diplom-rl80/refresh_measurements.py
```
