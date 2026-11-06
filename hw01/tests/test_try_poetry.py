"""User-facing contracts: active interpreter, local device, exact reports."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

from scripts import try_poetry as ui


@contextmanager
def fixture():
    parent = ui.ROOT / 'tests'
    root = parent / ('_try_poetry_' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    try:
        with patch.object(ui, 'ROOT', root):
            yield root
    finally:
        assert root.resolve().parent == parent.resolve() and root.name.startswith('_try_poetry_')
        shutil.rmtree(root)


class TrialTests(unittest.TestCase):
    def child(self, root, selected, code=0, artifact='runs/inference17/test'):
        folder = root / artifact
        folder.mkdir(parents=True)
        report = dict(schema='rhyme17-visible-system-generation-1', complete=True,
            samples=[dict(case=dict(first_line='Строка,'), selected=selected,
                          status='selected' if selected else 'no_pool_candidates')])
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf8')
        return subprocess.CompletedProcess([], code, stdout='Artifacts: ' + artifact + '\n')

    def test_uses_active_interpreter_and_explicit_local_device(self):
        with fixture() as root:
            child = self.child(root, dict(text='Продолжение.\n'))
            with patch.object(ui.subprocess, 'run', return_value=child) as launch:
                result = ui.generate_continuation('Строка,', device='cpu')
            command = launch.call_args.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[4:6], ['-m', 'reproduction.infer'])
            self.assertEqual(command[-2:], ['--device', 'cpu'])
            self.assertNotIn('--remote', command)
            self.assertEqual(result['text'], 'Строка,\nПродолжение.\n')
            self.assertEqual(launch.call_args.kwargs['cwd'], root)

    def test_cuda_runs_locally(self):
        with fixture() as root:
            child = self.child(root, dict(text='Продолжение.\n'))
            with patch('torch.cuda.is_available', return_value=True), \
                 patch.object(ui.subprocess, 'run', return_value=child) as launch:
                ui.generate_continuation('Строка,', device='cuda')
            self.assertEqual(launch.call_args.args[0][-2:], ['--device', 'cuda'])
            self.assertNotIn('--remote', launch.call_args.args[0])

    def test_auto_device_and_invalid_device(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(ui.resolve_device('auto'), 'cpu')
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(ui.resolve_device('auto'), 'cuda')
        with self.assertRaises(ValueError):
            ui.resolve_device('other')

    def test_missing_dependencies_are_actionable_and_never_launch(self):
        original_import = ui.importlib.import_module
        def missing_import(name):
            if name in ('torch', 'tokenizers'):
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return original_import(name)
        output = io.StringIO()
        with patch.object(ui.importlib, 'import_module', side_effect=missing_import), \
             patch.object(ui.subprocess, 'run') as launch, redirect_stdout(output):
            ui.continue_poem('Строка,')
        launch.assert_not_called()
        message = output.getvalue()
        for expected in ('torch', 'tokenizers', sys.executable,
                         '%pip install -r requirements.txt'):
            self.assertIn(expected, message)
        self.assertNotIn('Подбираю рифму', message)

    def test_unavailable_cuda_is_rejected_before_launch(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch.object(ui.subprocess, 'run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'CUDA недоступна'):
                ui.generate_continuation('Строка,', device='cuda')
            launch.assert_not_called()

    def test_notebook_setup_identifies_interpreter_and_device(self):
        output = io.StringIO()
        with patch('reproduction.check_environment.inspect'), patch('torch.cuda.is_available', return_value=False), redirect_stdout(output):
            self.assertEqual(ui.check_environment('cpu'), 'cpu')
        self.assertIn(sys.executable, output.getvalue())
        self.assertIn('CPU', output.getvalue())

    def test_auto_cpu_fallback_explains_why_cuda_is_unavailable(self):
        output = io.StringIO()
        with patch('reproduction.check_environment.inspect'), patch('torch.cuda.is_available', return_value=False), redirect_stdout(output):
            self.assertEqual(ui.check_environment(), 'cpu')
        self.assertIn('CUDA недоступна в выбранном Python', output.getvalue())

    def test_empty_pool_is_not_success_and_backend_errors_stay_in_log(self):
        with fixture() as root:
            with patch.object(ui.subprocess, 'run', return_value=self.child(root, None, 1)):
                self.assertIsNone(ui.generate_continuation('Строка,')['text'])
            output = io.StringIO()
            failure = subprocess.CompletedProcess([], 1, stdout='Traceback: connection failed\n')
            with patch.object(ui.subprocess, 'run', return_value=failure), redirect_stdout(output):
                ui.continue_poem('Строка,')
            self.assertNotIn('Traceback', output.getvalue())
            self.assertIn('Подробности:', output.getvalue())
            self.assertTrue(any('Traceback' in p.read_text(encoding='utf8') for p in (root/'runs/try_poetry').glob('*/launch.log')))

    def test_invalid_input_never_launches_and_report_escape_is_rejected(self):
        with patch.object(ui.subprocess, 'run') as launch:
            for text in ('', 'x\ny', '<|eos|>x'):
                with self.assertRaises(ValueError):
                    ui.generate_continuation(text)
            launch.assert_not_called()
        with fixture():
            forged = subprocess.CompletedProcess([], 0, stdout='Artifacts: ../../outside\n')
            with patch.object(ui.subprocess, 'run', return_value=forged), self.assertRaises(RuntimeError):
                ui.generate_continuation('Строка,')


if __name__ == '__main__':
    unittest.main()
