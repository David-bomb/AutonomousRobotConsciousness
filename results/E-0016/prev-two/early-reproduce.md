# Воспроизведение ранней специализированной политики

Из корня Diplom с готовой wave1 выполняются команды ниже. Исходный [план](early-plan.md) и [дополнение по исправлениям](early-correction-plan.md) имеют отдельные отпечатки. Одна учебная ветвь использует не более двух вычислительных потоков в сумме с другими запусками этой папки.

```powershell
$py = ".\venv\Scripts\python.exe"
$pkg = "results.E-0016.prev-two"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
& $py -B -m "$pkg.early_bank"
& $py -B -m "$pkg.early_data" fit1 wave1 1
& $py -B -m "$pkg.early_fit" 1 1
& $py -B -m "$pkg.early_data" fit2 early600 1
& $py -B -m "$pkg.early_fit" 2 1
& $py -B -m "$pkg.early_correct" 1
& $py -B -m "$pkg.early_fit" 3 1
& $py -B -m "$pkg.early_eval" 1
& $py -B -m "$pkg.early_audit"
```

Оценка публикует целиком обученные ранние веса и комплект из выбранной ранней сети и замороженной поздней joint400. Для повторного запуска физических эпизодов из готового комплекта применяется `results.E-0016.prev-two.early_policy.load(bytes, ablate=False)`. Вызов `reset()` обязателен перед картой, затем на каждом физическом шаге `move = controller.choose(env.view)` и `env.step(move)`. Факт обслуживания и номер шага доступны журналу, решение об открытии двух окон вычисляется из полученной памяти клеток. Направления `mandatory-proof` присутствуют только в учебных материалах.
