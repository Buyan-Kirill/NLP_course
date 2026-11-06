"""Read-only handoff diagnostics; never install packages or download weights."""
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate_host(system=None, machine=None, python=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).casefold()
    python = python or sys.version_info[:2]
    if tuple(python) != (3, 12):
        raise RuntimeError('Для этого комплекта нужен Python 3.12; выбран '
                           + '.'.join(map(str, python)) + '. Выберите его же в ядре ноутбука.')
    supported = {'Windows': {'amd64', 'x86_64'}, 'Linux': {'x86_64', 'amd64'},
                 'Darwin': {'arm64', 'aarch64'}}
    if machine not in supported.get(system, set()):
        raise RuntimeError(f'Комплект не подготовлен для {system}/{machine}. '
                           'Предусмотрены Windows/Linux x86-64 и macOS Apple Silicon; MPS не используется.')
    return system


def inspect():
    validate_host()
    required = dict(torch='2.9.1', numpy='2.2.6', tokenizers='0.22.1', ipykernel='7.4.0',
                    huggingface_hub='1.33.0', PyYAML='6.0.3')
    if sys.platform != 'win32':
        required.update({'ufal.udpipe':'1.3.1.1', 'jellyfish':'1.0.3', 'pyconll':'3.2.0',
                         'jsonpickle':'3.0.4', 'coloredlogs':'15.0.1', 'humanfriendly':'10.0', 'six':'1.17.0'})
    packages, missing = {}, []
    for name in required:
        try: packages[name] = version(name)
        except PackageNotFoundError: missing.append(name)
    if missing:
        raise RuntimeError('Не установлены зависимости: '+', '.join(missing)+'. Установите requirements.txt в Python ядра.')
    mismatches = [f'{name}: {packages[name]} вместо {wanted}' for name,wanted in required.items()
                  if packages[name].split('+')[0] != wanted]
    if mismatches:
        raise RuntimeError('Версии отличаются от комплекта: '+'; '.join(mismatches)+'. Установите зависимости по инструкции для вашей ОС.')
    paths = ['configs/inference.json',
             'artifacts/releases/poetry17/continuation.pt',
             'artifacts/tokenizers_v3/selected/tokenizer.json',
             'data/rhyme17_candidate/vocabulary_all_poetry_wordforms.json',
             'artifacts/evaluators/rpst/version.json']
    absent = [p for p in paths if not (ROOT / p).is_file()]
    if absent:
        raise RuntimeError('Загрузите ресурсы: python -m reproduction.download inference. Нет файлов: '+', '.join(absent))
    import torch
    import tokenizers
    import numpy
    return dict(python=sys.executable, version=platform.python_version(), platform=platform.platform(),
                device='cuda' if torch.cuda.is_available() else 'cpu', packages=packages,
                note='Проверены импорты и наличие основных файлов; SHA и загрузка RPST проверяются при инференсе.')


def main():
    try:print(json.dumps(inspect(), ensure_ascii=False, indent=2))
    except (RuntimeError, ImportError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':main()
