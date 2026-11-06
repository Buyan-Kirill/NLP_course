"""Local memory admission is based on free VRAM, not the author's GPU model."""
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch
import uuid

from reproduction import local_runtime as local


class LocalRuntimeTests(unittest.TestCase):
    def test_eight_gib_gpu_with_sufficient_free_memory_is_accepted(self):
        audit = Mock()
        torch = Mock()
        torch.cuda.current_device.return_value = 1
        torch.cuda.device_count.return_value = 2
        torch.cuda.mem_get_info.return_value = (6400 * 1024**2, 8192 * 1024**2)
        local.admit_local(torch, 3072, audit)
        torch.cuda.mem_get_info.assert_called_once_with(1)
        self.assertEqual(audit.record.call_args.args[0], 'admission_granted')

    def test_low_memory_and_failed_telemetry_are_rejected_and_recorded(self):
        for options in [dict(return_value=(2400 * 1024**2, 8192 * 1024**2)),
                        dict(side_effect=RuntimeError('telemetry unavailable'))]:
            audit = Mock()
            torch = Mock()
            with patch.object(torch.cuda, 'mem_get_info', **options), self.assertRaises(RuntimeError):
                local.admit_local(torch, 3072, audit)
            self.assertEqual(audit.record.call_args.args[0], 'admission_rejected')

    def test_frozen_admission_is_restored_after_failure(self):
        folder = local.runtime.ROOT / 'tests' / ('_local_runtime_' + uuid.uuid4().hex)
        folder.mkdir()
        previous = local.shared.admit
        def fail(*args, **kwargs):
            self.assertIs(local.shared.admit, local.admit_local)
            raise RuntimeError('test failure')
        try:
            with patch.object(local.runtime, 'run', side_effect=fail), self.assertRaises(RuntimeError):
                local.run(Path('config.json'), Path('input.json'), Path('pool.json'),
                          folder / 'report.json', 'cpu', pool_sha256='a' * 64)
            self.assertIs(local.shared.admit, previous)
            self.assertTrue((folder / 'report.local_runtime.json').is_file())
        finally:
            assert folder.resolve().parent == (local.runtime.ROOT / 'tests').resolve()
            shutil.rmtree(folder)


if __name__ == '__main__':
    unittest.main()
