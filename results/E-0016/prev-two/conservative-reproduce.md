# Воспроизведение консервативных серий

Команды запускаются из корня Diplom с рабочими исходными wave1 и joint400 в отдельной копии проекта. Хеши бинарных библиотек именно оценённой сборки сохранены вместе с отпечатком весового комплекта. Компилятор MSVC 2019 может записать иное время линковки при повторной сборке. Исходники обеих библиотек сохранены рядом.

```powershell
$py = ".\venv\Scripts\python.exe"
$pkg = "results.E-0016.prev-two"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
& .\results\E-0016\prev-two\build_cautious.cmd
& $py -B -m "$pkg.cautious_bank"
& $py -B -m "$pkg.cautious_collect"
& $py -B -m "$pkg.cautious_fit" collect
& $py -B -m "$pkg.cautious_fit" train
& $py -B -m "$pkg.cautious_eval"
& $py -B -m "$pkg.cautious_audit" conservative
& .\results\E-0016\prev-two\build_path_counts.cmd
& $py -B -m "$pkg.path_bank"
& $py -B -m "$pkg.path_fit"
& $py -B -m "$pkg.cautious_eval" geodesic
& $py -B -m "$pkg.cautious_audit" geodesic
& $py -B -m "$pkg.external_train"
```

Для проверки готовых комплекта и физики достаточно загрузки весов через соответствующий модуль в [передаче директору](handoff.md). Агент не требует доступа к файлам траекторий или обучающим геометриям. Расчёт признаков использует модули и локальные DLL из проекта. [Регистрация первого опыта](conservative-plan.md) · [регистрация серии 1 000](geodesic-plan.md) · [план внешней учебной проверки](external-train-plan.md)
