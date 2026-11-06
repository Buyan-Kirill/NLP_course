"""Small notebook interface using the selected Python environment and device."""
import json
import importlib
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/inference.json'


def resolve_device(device):
    """Validate this interpreter before starting the expensive rhyme search."""
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Устройство должно быть auto, cpu или cuda.')
    if sys.platform == 'darwin' and device == 'cuda':
        raise RuntimeError('На macOS используйте auto или cpu; CUDA недоступна, MPS в этой версии не подключён.')
    modules = {}
    missing = []
    for name in ('torch', 'tokenizers', 'numpy'):
        try:
            modules[name] = importlib.import_module(name)
        except ModuleNotFoundError as error:
            missing.append(error.name or name)
        except (ImportError, OSError) as error:
            raise RuntimeError(f'Не удалось загрузить {name} в Python {sys.executable}: {error}') from error
    if missing:
        packages = ', '.join(sorted(set(missing)))
        raise RuntimeError(
            f'В выбранном Python отсутствуют зависимости: {packages}.\n'
            f'Python ядра: {sys.executable}\n'
            'Выберите ядро окружения с зависимостями. Либо установите их в этом ядре:\n'
            '%pip install -r requirements.txt\n'
            'После установки перезапустите ядро и выполните Run All. '
            'Настройка CUDA описана в docs/reproduction.md.')
    torch = modules['torch']
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'В Python {sys.executable} CUDA недоступна. '
            f'Установлен PyTorch {torch.__version__}, CUDA-сборка: {torch.version.cuda}.\n'
            'Установите зависимости в Python выбранного ядра:\n'
            '%pip install -r requirements.txt\n'
            'Затем перезапустите ядро. Настройка описана в docs/reproduction.md.')
    return device


def check_environment(device='auto'):
    """Show the notebook interpreter and check its local dependencies/device."""
    from reproduction.check_environment import inspect
    inspect()
    print(f'Python ядра: {sys.executable}')
    selected = resolve_device(device)
    torch = importlib.import_module('torch')
    print(f'PyTorch {torch.__version__}; CUDA {torch.version.cuda}')
    if selected == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        if device == 'cpu':
            print('Устройство: CPU (задано явно).')
        elif sys.platform == 'darwin':
            print('Устройство: CPU (macOS; ускорение MPS не подключено).')
        else:
            print('Устройство: CPU — CUDA недоступна в выбранном Python. '
                  'Для NVIDIA GPU проверьте установку зависимостей и драйвер; '
                  'docs/reproduction.md.')
    return selected


def generate_continuation(text, *, device='auto', config=CONFIG):
    """Run locally in the notebook interpreter and return its exact saved result."""
    if not isinstance(text, str) or not text.strip() or '\n' in text or '\r' in text:
        raise ValueError('Введи одну непустую стихотворную строку.')
    if re.search(r'<\|[^\r\n]*?\|>', text):
        raise ValueError('В строке не должно быть служебных токенов.')
    device = resolve_device(device)
    log_dir = ROOT / 'runs/try_poetry' / uuid.uuid4().hex
    log_dir.mkdir(parents=True)
    log_path = log_dir / 'launch.log'
    command = [sys.executable, '-B', '-X', 'utf8', '-m', 'reproduction.infer',
               '--config', str(config), '--text', text, '--device', device]
    process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding='utf-8', errors='replace', check=False)
    log_path.write_text(process.stdout, encoding='utf-8')
    failure = 'Не удалось получить продолжение. Подробности: ' + log_path.relative_to(ROOT).as_posix()
    paths = [line[len('Artifacts: '):] for line in process.stdout.splitlines() if line.startswith('Artifacts: ')]
    if process.returncode not in (0, 1) or len(paths) != 1:
        raise RuntimeError(failure)
    artifacts = (ROOT / paths[0]).resolve()
    if not artifacts.is_relative_to((ROOT / 'runs/inference17').resolve()):
        raise RuntimeError(failure)
    report_path = artifacts / 'report.json'
    if not report_path.is_file():
        raise RuntimeError(failure)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('schema') != 'rhyme17-visible-system-generation-1'
            or report.get('complete') is not True or len(report.get('samples', [])) != 1):
        raise RuntimeError(failure)
    sample = report['samples'][0]
    if sample['case']['first_line'] != text:
        raise RuntimeError(failure)
    selected = sample.get('selected')
    if selected is None and process.returncode == 1:
        return dict(text=None, status=sample['status'], artifacts=artifacts.relative_to(ROOT).as_posix())
    if (process.returncode != 0 or sample['status'] != 'selected'
            or not isinstance(selected, dict) or not isinstance(selected.get('text'), str)
            or not selected['text'].strip()):
        raise RuntimeError(failure)
    return dict(text=text + '\n' + selected['text'], status='selected',
                artifacts=artifacts.relative_to(ROOT).as_posix())


def continue_poem(text, *, device='auto', config=CONFIG):
    """Display only progress and a fresh poem; technical records stay on disk."""
    try:
        device = resolve_device(device)
        print(f'Подбираю рифму и продолжение ({device.upper()})…', flush=True)
        result = generate_continuation(text, device=device, config=config)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error))
        return
    if result['text'] is None:
        if result['status'] == 'no_pool_candidates':
            print('В словаре не нашлось подходящих рифм. Можно указать ударение в последнем слове, например: саду́.')
        else:
            print('Рифмы найдены, но модель не построила строку в пределах поиска.')
        print('Подробности: ' + result['artifacts'])
        return
    print('\n' + result['text'], end='' if result['text'].endswith('\n') else '\n')
