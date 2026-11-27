# Полные валидационные поездки E-0016

Каждая строка — одни закреплённые веса на одних и тех же 100 новых картах dev. Во всех эпизодах модель использует обычное наблюдение, стартует с исходными запасами и движется до штатного исхода. H1–H4 содержат по 25 карт. Выбор весов по этому набору делает его выборкой разработки. Оценка на I, II и III ведётся отдельно.

| Запуск | Веса | Финиши / 100 | H1 | H2 | H3 | H4 | Посетил оба сервиса | Покартовый журнал |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pilot200 | [initial](../E-0016/runs/pilot200/initial.pt) | 0 | 0 | 0 | 0 | 0 | 0 | [JSON](../E-0016/runs/pilot200/initial.physical.json) |
| pilot200 | [imitation](../E-0016/runs/pilot200/imitation.pt) | 21 | 8 | 5 | 6 | 2 | 38 | [JSON](../E-0016/runs/pilot200/imitation.physical.json) |
| pilot200 | [rl](../E-0016/runs/pilot200/rl.pt) | 20 | 7 | 6 | 5 | 2 | 41 | [JSON](../E-0016/runs/pilot200/rl.physical.json) |
| recovery800 | [imitation](../E-0016/runs/recovery800/imitation.pt) | 29 | 16 | 7 | 4 | 2 | 43 | [JSON](../E-0016/runs/recovery800/imitation.physical.json) |
| recovery800 | [rl](../E-0016/runs/recovery800/rl.pt) | 28 | 17 | 5 | 5 | 1 | 43 | [JSON](../E-0016/runs/recovery800/rl.physical.json) |
| critical800 | [imitation](../E-0016/runs/critical800/imitation.pt) | 21 | 12 | 6 | 2 | 1 | 46 | [JSON](../E-0016/runs/critical800/imitation.physical.json) |
| critical800 | [rl](../E-0016/runs/critical800/rl.pt) | 23 | 11 | 5 | 4 | 3 | 47 | [JSON](../E-0016/runs/critical800/rl.physical.json) |
| rich800 | [imitation](../E-0016/runs/rich800/imitation.pt) | 10 | 5 | 1 | 3 | 1 | 47 | [JSON](../E-0016/runs/rich800/imitation.physical.json) |
| rich800 | [rl1](../E-0016/runs/rich800/rl1.pt) | 19 | 10 | 3 | 5 | 1 | 51 | [JSON](../E-0016/runs/rich800/rl1.physical.json) |
| rich800 | [rl2](../E-0016/runs/rich800/rl2.pt) | 27 | 12 | 5 | 5 | 5 | 50 | [JSON](../E-0016/runs/rich800/rl2.physical.json) |
| mission600 | [s3](../E-0016/sequence/runs/mission600/s3.pt) | 29 | 16 | 8 | 4 | 1 | 52 | [JSON](../E-0016/sequence/runs/mission600/s3.physical.json) |
| mission600 | [s8](../E-0016/sequence/runs/mission600/s8.pt) | 34 | 15 | 9 | 8 | 2 | 57 | [JSON](../E-0016/sequence/runs/mission600/s8.physical.json) |
| exploration-student | [random](../E-0016/exploration-student/weights/random.pt) | 0 | 0 | 0 | 0 | 0 | 12 | [JSON](../E-0016/exploration-student/dev/random.json) |
| exploration-student | [imitation12](../E-0016/exploration-student/weights/imitation12.pt) | 31 | 17 | 9 | 4 | 1 | 48 | [JSON](../E-0016/exploration-student/dev/imitation12.json) |
| exploration-student | [imitation32](../E-0016/exploration-student/weights/imitation32.pt) | 27 | 15 | 6 | 4 | 2 | 43 | [JSON](../E-0016/exploration-student/dev/imitation32.json) |
| mission600 | [rl](../E-0016/sequence/runs/mission600/rl.pt) | 36 | 17 | 9 | 8 | 2 | 52 | [JSON](../E-0016/sequence/runs/mission600/rl.physical.json) |
| exploration-student | [corrected16](../E-0016/exploration-student/weights/corrected16.pt) | 21 | 14 | 5 | 2 | 0 | 29 | [JSON](../E-0016/exploration-student/dev/corrected16.json) |
| explore1600 | [s1200](../E-0016/sequence/runs/explore1600/s1200.pt) | 37 | 20 | 9 | 6 | 2 | 57 | [JSON](../E-0016/sequence/runs/explore1600/s1200.physical.json) |
| explore1600 | [s3200](../E-0016/sequence/runs/explore1600/s3200.pt) | 35 | 21 | 8 | 2 | 4 | 58 | [JSON](../E-0016/sequence/runs/explore1600/s3200.physical.json) |
| explore1600 | [rl](../E-0016/sequence/runs/explore1600/rl.pt) | 36 | 19 | 9 | 6 | 2 | 54 | [JSON](../E-0016/sequence/runs/explore1600/rl.physical.json) |
| exit600 | [s900](../E-0016/sequence/runs/exit600/s900.pt) | 44 | 19 | 13 | 8 | 4 | 56 | [JSON](../E-0016/sequence/runs/exit600/s900.physical.json) |
| exit600 | [s2700](../E-0016/sequence/runs/exit600/s2700.pt) | 43 | 19 | 10 | 10 | 4 | 54 | [JSON](../E-0016/sequence/runs/exit600/s2700.physical.json) |
| exit600 | [q75](../E-0016/sequence/runs/exit600/q75.pt) | 42 | 20 | 11 | 7 | 4 | 57 | [JSON](../E-0016/sequence/runs/exit600/q75.physical.json) |
| exit600 | [q250](../E-0016/sequence/runs/exit600/q250.pt) | 39 | 19 | 11 | 5 | 4 | 55 | [JSON](../E-0016/sequence/runs/exit600/q250.physical.json) |
| returns240 | [q120](../E-0016/sequence/runs/returns240/q120.pt) | 26 | 13 | 8 | 3 | 2 | 48 | [JSON](../E-0016/sequence/runs/returns240/q120.physical.json) |
| returns240 | [q450](../E-0016/sequence/runs/returns240/q450.pt) | 9 | 4 | 2 | 2 | 1 | 33 | [JSON](../E-0016/sequence/runs/returns240/q450.physical.json) |
| returns240 | [q450x5](../E-0016/sequence/runs/returns240/q450x5.pt) | 2 | 1 | 0 | 1 | 0 | 24 | [JSON](../E-0016/sequence/runs/returns240/q450x5.physical.json) |
| terminal-s900 | [rl](../E-0016/sequence/runs/terminal-s900/rl.pt) | 44 | 19 | 13 | 8 | 4 | 56 | [JSON](../E-0016/sequence/runs/terminal-s900/rl.physical.json) |
| joint400 | [rl](../E-0016/sequence/runs/joint400/rl.pt) | 59 | 24 | 18 | 12 | 5 | 64 | [JSON](../E-0016/sequence/runs/joint400/rl.physical.json) |
| field3200 | [rl](../E-0016/sequence/runs/field3200/rl.pt) | 60 | 23 | 18 | 12 | 7 | 66 | [JSON](../E-0016/sequence/runs/field3200/rl.physical.json) |
| calibrated600 | [rl](../E-0016/sequence/runs/calibrated600/rl.pt) | 51 | 20 | 15 | 12 | 4 | 55 | [JSON](../E-0016/sequence/runs/calibrated600/rl.physical.json) |
| field3000-clean | [rl](../E-0016/sequence/runs/field3000-clean/rl.pt) | 60 | 23 | 17 | 12 | 8 | 66 | [JSON](../E-0016/sequence/runs/field3000-clean/rl.physical.json) |
| local12000 | [rl](../E-0016/sequence/runs/local12000/rl.pt) | 58 | 23 | 18 | 13 | 4 | 63 | [JSON](../E-0016/sequence/runs/local12000/rl.physical.json) |
| gates3000 | [rl](../E-0016/sequence/runs/gates3000/rl.pt) | 57 | 23 | 15 | 12 | 7 | 63 | [JSON](../E-0016/sequence/runs/gates3000/rl.physical.json) |
| paired200-observed | [rl](../E-0016/sequence/runs/paired200-observed/rl.pt) | 40 | 20 | 11 | 6 | 3 | 53 | [JSON](../E-0016/sequence/runs/paired200-observed/rl.physical.json) |
| risk01 | [rl](../E-0016/sequence/runs/risk01/rl.pt) | 60 | 23 | 17 | 12 | 8 | 66 | [JSON](../E-0016/sequence/runs/risk01/rl.physical.json) |
| route1200 | [rl](../E-0016/sequence/runs/route1200/rl.pt) | 52 | 20 | 15 | 11 | 6 | 59 | [JSON](../E-0016/sequence/runs/route1200/rl.physical.json) |
| route1200 | [pre](../E-0016/sequence/runs/route1200/pre.pt) | 53 | 19 | 16 | 13 | 5 | 59 | [JSON](../E-0016/sequence/runs/route1200/pre.physical.json) |
| rooms3000 | [rl](../E-0016/sequence/runs/rooms3000/rl.pt) | 59 | 23 | 17 | 12 | 7 | 65 | [JSON](../E-0016/sequence/runs/rooms3000/rl.physical.json) |
| reconstruct400 | [missing](../E-0016/sequence/runs/reconstruct400/missing.pt) | 45 | 19 | 11 | 9 | 6 | 49 | [JSON](../E-0016/sequence/runs/reconstruct400/missing.physical.json) |
| reconstruct400 | [rl](../E-0016/sequence/runs/reconstruct400/rl.pt) | 45 | 19 | 11 | 9 | 6 | 49 | [JSON](../E-0016/sequence/runs/reconstruct400/rl.physical.json) |
| reconstruct400 | [marginal](../E-0016/sequence/runs/reconstruct400/marginal.pt) | 50 | 19 | 12 | 12 | 7 | 54 | [JSON](../E-0016/sequence/runs/reconstruct400/marginal.physical.json) |
| reconstruct400 | [direct](../E-0016/sequence/runs/reconstruct400/direct.pt) | 49 | 20 | 12 | 10 | 7 | 53 | [JSON](../E-0016/sequence/runs/reconstruct400/direct.physical.json) |
| reconstruct-dagger200 | [rl](../E-0016/sequence/runs/reconstruct-dagger200/rl.pt) | 45 | 19 | 11 | 9 | 6 | 49 | [JSON](../E-0016/sequence/runs/reconstruct-dagger200/rl.physical.json) |
| reconstruct400 | [marginal1](../E-0016/sequence/runs/reconstruct400/marginal1.pt) | 49 | 19 | 12 | 11 | 7 | 54 | [JSON](../E-0016/sequence/runs/reconstruct400/marginal1.physical.json) |
| reconstruct400 | [direct1](../E-0016/sequence/runs/reconstruct400/direct1.pt) | 45 | 18 | 9 | 11 | 7 | 49 | [JSON](../E-0016/sequence/runs/reconstruct400/direct1.physical.json) |

## Отдельно согласованный вариант с двумя предыдущими показаниями

В каждой паре одинаковые веса. Отключение означает, что оба прошлых показания отсутствуют на протяжении всего эпизода. С памятью они получены на предыдущем реальном шаге и пусты на старте.

| Режим | Веса | Финиши / 100 | H1 | H2 | H3 | H4 | Посетил оба сервиса | Покартовый журнал |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Оба показания отключены | [post-teacher](../E-0016/prev-two/weights/post-teacher.pt) | 41 | 20 | 10 | 6 | 5 | 47 | [JSON](../E-0016/prev-two/dev/post-teacher-ablated.json) |
| Два прошлых показания | [post-teacher](../E-0016/prev-two/weights/post-teacher.pt) | 66 | 24 | 21 | 14 | 7 | 82 | [JSON](../E-0016/prev-two/dev/post-teacher-honest.json) |
| Оба показания отключены | [post-teacher-rl](../E-0016/prev-two/weights/post-teacher-rl.pt) | 43 | 20 | 10 | 8 | 5 | 49 | [JSON](../E-0016/prev-two/dev/post-teacher-rl-ablated.json) |
| Два прошлых показания | [post-teacher-rl](../E-0016/prev-two/weights/post-teacher-rl.pt) | 65 | 25 | 21 | 14 | 5 | 83 | [JSON](../E-0016/prev-two/dev/post-teacher-rl-honest.json) |
| Оба показания отключены | [route-rl](../E-0016/prev-two/weights/route-rl.pt) | 43 | 20 | 10 | 8 | 5 | 49 | [JSON](../E-0016/prev-two/dev/route-rl-ablated.json) |
| Два прошлых показания | [route-rl](../E-0016/prev-two/weights/route-rl.pt) | 65 | 25 | 21 | 14 | 5 | 83 | [JSON](../E-0016/prev-two/dev/route-rl-honest.json) |
| Оба показания отключены | [super1500](../E-0016/prev-two/weights/super1500.pt) | 32 | 19 | 5 | 6 | 2 | 53 | [JSON](../E-0016/prev-two/dev/super1500-ablated.json) |
| Два прошлых показания | [super1500](../E-0016/prev-two/weights/super1500.pt) | 44 | 20 | 15 | 6 | 3 | 81 | [JSON](../E-0016/prev-two/dev/super1500-honest.json) |
| Оба показания отключены | [super600](../E-0016/prev-two/weights/super600.pt) | 36 | 17 | 9 | 6 | 4 | 56 | [JSON](../E-0016/prev-two/dev/super600-ablated.json) |
| Два прошлых показания | [super600](../E-0016/prev-two/weights/super600.pt) | 36 | 18 | 9 | 8 | 1 | 75 | [JSON](../E-0016/prev-two/dev/super600-honest.json) |
| Оба показания отключены | [wave1](../E-0016/prev-two/weights/wave1.pt) | 34 | 18 | 7 | 5 | 4 | 49 | [JSON](../E-0016/prev-two/dev/wave1-ablated.json) |
| Два прошлых показания | [wave1](../E-0016/prev-two/weights/wave1.pt) | 53 | 24 | 15 | 10 | 4 | 83 | [JSON](../E-0016/prev-two/dev/wave1-honest.json) |
| Оба показания отключены | [wave1-pg](../E-0016/prev-two/weights/wave1-pg.pt) | 34 | 18 | 7 | 5 | 4 | 49 | [JSON](../E-0016/prev-two/dev/wave1-pg-ablated.json) |
| Два прошлых показания | [wave1-pg](../E-0016/prev-two/weights/wave1-pg.pt) | 53 | 24 | 15 | 10 | 4 | 83 | [JSON](../E-0016/prev-two/dev/wave1-pg-honest.json) |

Корпус dev, SHA-256 · 4ce3fcd9106332497bf6a6e6ad26040aae303fc297a77dc7606828c5edabe8a6

Таблица пересчитывается командой из корня лаборатории.

```powershell
python projects/diplom-rl80/refresh_measurements.py
```
