# Воспроизведение на CPU

Запускать из корня Diplom в PowerShell. Нужны PyTorch и NumPy проектного venv, библиотеки геометрии из [регистрации](registration.json), а для исторического учителя — SciPy, установленный в системном Python. Обучение и выбор весов используют зарегистрированные train-группы. Рабочий контроллер загружается отдельно от обучающих сценариев.

```powershell
$py = ".\venv\Scripts\python.exe"
$pkg = "results.E-0016.prev-two"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
& $py -B -m "$pkg.physical" mission600 baseline
& $py -B -m "$pkg.samples" initial
& $py -B -m "$pkg.fit" initial
& $py -B -m "$pkg.compare" both super600 super1500
& $py -B -m "$pkg.terminal" auto1 wave1 1
& $py -B -m "$pkg.compare" both wave1-pg wave1
& $py -B -m "$pkg.takeover"
& $py -B -m "$pkg.route_search"
& $py -B -m "$pkg.compare" both route-rl
& C:\Python313\python.exe -B -m "$pkg.mandatory"
& $py -B -m "$pkg.continuation_data"
& $py -B -m "$pkg.continuation_fit"
& $py -B -m "$pkg.continuation_rl"
& $py -B -m "$pkg.compare" both post-teacher post-teacher-rl
& $py -B -m "$pkg.release"
```

При полном пересчёте используйте отдельную копию проекта с исходными банками и регистрацией, поместив в неё исходники этой ветви без производных каталогов `weights`, `samples`, `takeovers`, `mandatory-train`, `terminal` и `dev`. В рабочем комплекте оценщик читает сохранённые совпадающие по отпечаткам результаты. Вызов `final_check` снова исполняет все 100 физических эпизодов через публичный адаптер с прямым выбором на каждом ходу. Оценщик предоставляет роботу только текущий Observation. Обе предыдущие величины обновляет сам робот. Перед новой картой вызывается `reset()`.
