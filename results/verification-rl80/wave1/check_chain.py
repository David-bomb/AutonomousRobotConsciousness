"""Проверить загрузку исходных модулей и тензоров в процесс на CPU без предъявления карты."""
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from suite import check, file_hash, read, write_new
from bridge import load
import torch


def main():
    frozen = HERE / "frozen"
    selection = read(HERE / "selection.json")
    weights = frozen / "weights.pt"
    check(file_hash(weights) == selection["weights_sha256"], "Изменились веса")
    bot = load(weights.read_bytes())
    paths = {}
    for stem in ("frozen_adapter", "adapter", "policy", "critical_policy", "rich_policy"):
        module = sys.modules["results.E-0016." + stem]
        source = Path(module.__file__).resolve()
        check(source == frozen / (stem + ".py") and file_hash(source) == selection["source_sha256"][source.name],
              "Загружена иная версия Python-модуля")
        paths[stem] = source.relative_to(HERE).as_posix()
    dll = Path(sys.modules["results.E-0016.policy"].LIB._name).resolve()
    check(dll == frozen / "geometry.dll" and file_hash(dll) == selection["source_sha256"][dll.name],
          "Подмена нативной библиотеки")
    expected = torch.load(weights, map_location="cpu", weights_only=True)
    actual = bot.model.state_dict()
    check(expected.keys() == actual.keys()
          and all(v.device.type == "cpu" and torch.equal(v, expected[k]) for k,v in actual.items()),
          "Не все тензоры весов точно воспроизведены на CPU")
    check(not bot.model.training and not any(p.requires_grad for p in bot.model.parameters())
          and torch.get_num_threads() == 1, "Ожидался однопоточный режим применения весов")
    result = dict(weights_sha256=selection["weights_sha256"], modules=paths,
                  native_library=dll.relative_to(HERE).as_posix(), model=type(bot.model).__name__,
                  identical_cpu_tensors=len(actual), parameters=sum(p.numel() for p in bot.model.parameters()),
                  train_mode=False, gradients=False, torch_threads=torch.get_num_threads())
    out = HERE / "chain.json"
    if out.exists():
        check(read(out) == result, "Аудит цепочки изменился")
    else:
        write_new(out, result)
    print(result["model"], result["identical_cpu_tensors"], "identical CPU tensors")


if __name__ == "__main__":
    main()
