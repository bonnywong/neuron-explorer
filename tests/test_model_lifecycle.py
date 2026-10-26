"""Fast lifecycle tests: no torch, model weights, GPU or network required."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
from model_manager import ModelManager, MODEL, verified, write_json


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.weights = self.folder / 'model.safetensors'
        self.weights.write_bytes(b'weight fixture')
        self.manifest = {'model':MODEL,'revision':'a'*40,'snapshot':str(self.folder),'verified':True,
                         'files':[{'path':str(self.weights),'size':self.weights.stat().st_size,'mtime':self.weights.stat().st_mtime_ns}]}
        write_json(self.folder / 'manifest.json', self.manifest)
        self.runtime = Mock(dev=None,precision=None)
        self.manager = ModelManager(self.runtime,state_dir=self.folder)
        self.manager.diagnostics = Mock(return_value={'cuda':True,'mps':False,'device':'cuda'})

    def test_incomplete_changed_missing_and_empty_files_are_not_installed(self):
        self.assertTrue(verified(self.manifest))
        self.assertFalse(verified({**self.manifest,'files':[]}))
        self.weights.write_bytes(b'corrupt')
        self.assertFalse(verified(self.manifest))
        with self.assertRaisesRegex(ValueError,'Download and verify'): self.manager.start()
        self.weights.unlink()
        self.assertFalse(verified(self.manifest))

    def test_start_ready_duplicate_stop_restart(self):
        self.manager.start(); self.manager.job.join(2)
        self.assertEqual(self.manager.state,'ready')
        self.manager.start(); self.runtime.load.assert_called_once()
        self.runtime.warmup.assert_called_once()
        self.manager.cancel_download()  # unrelated cancellation must not unload a ready runtime
        self.assertEqual(self.manager.state,'ready')
        self.manager.stop(); self.manager.job.join(2)
        self.assertEqual(self.manager.state,'installed'); self.runtime.unload.assert_called_once()
        self.assertTrue(self.manager.cancel.is_set())
        self.manager.start(); self.manager.job.join(2)
        self.assertEqual(self.runtime.load.call_count,2)
        self.assertFalse(self.manager.cancel.is_set())

    def test_stop_while_loading_does_not_publish_ready(self):
        entered, finish = threading.Event(), threading.Event()
        self.runtime.load.side_effect=lambda *args: (entered.set(),finish.wait(2))
        self.manager.start(); entered.wait(2); self.manager.stop()
        self.assertEqual(self.manager.state,'stopping')
        self.manager.stop(); finish.set(); self.manager.job.join(3)
        self.assertEqual(self.manager.state,'installed')
        self.runtime.unload.assert_called_once()

    def test_load_oom_failure_can_recover(self):
        self.runtime.load.side_effect=RuntimeError('CUDA out of memory')
        self.manager.start(); self.manager.job.join(2)
        self.assertEqual(self.manager.state,'error')
        self.assertIn('Free device memory',self.manager.error)
        self.runtime.unload.assert_called_once()
        self.runtime.load.side_effect=None
        self.manager.start(); self.manager.job.join(2)
        self.assertEqual(self.manager.state,'ready')

    def test_cpu_requires_explicit_choice_and_cuda_cannot_fallback(self):
        self.manager.diagnostics.return_value={'device':'cpu','cuda':False,'mps':False}
        for device in ['auto','cuda','mps','invalid']:
            with self.assertRaises(ValueError): self.manager.start(device)
        self.runtime.load.assert_not_called()
        self.manager.start('cpu'); self.manager.job.join(2)
        self.runtime.load.assert_called_once_with(str(self.folder),'cpu')

    def test_restriction_applies_to_download_and_start(self):
        self.manager.restricted=True
        with self.assertRaises(ValueError): self.manager.download()
        with self.assertRaises(ValueError): self.manager.start()
        self.assertFalse(self.manager.status()['capabilities']['live'])

    def test_duplicate_download_cancel_and_resume(self):
        process=Mock(returncode=0); process.poll.return_value=None
        with patch('model_manager.subprocess.Popen',return_value=process) as popen:
            self.manager.download(); self.manager.download()
            popen.assert_called_once()
            self.manager.cancel_download(); process.terminate.assert_called_once()
            self.assertTrue(self.weights.exists())
            self.manager.download(); self.assertEqual(popen.call_count,2)
            process.poll.return_value=0
            self.assertEqual(self.manager.status()['state'],'installed')

    def test_interrupted_worker_reports_actionable_error(self):
        process=Mock(returncode=1); process.poll.return_value=1
        self.manager.process=process; self.manager.state='downloading'
        write_json(self.folder / 'progress.json',{'error':'Connection interrupted. Retry to resume.'})
        self.assertEqual(self.manager.status()['state'],'error')
        self.assertIn('Retry',self.manager.error)

    def test_verified_cache_survives_application_restart(self):
        new_manager=ModelManager(self.runtime,state_dir=self.folder)
        self.assertEqual(new_manager.state,'installed')
        self.runtime.load.assert_not_called()

    def test_stop_waits_for_entire_request_and_rejects_new_requests(self):
        self.manager.start(); self.manager.job.join(2)
        self.assertTrue(self.manager.begin_inference())
        cleared=Mock(); self.manager.on_stopped=cleared
        self.manager.stop()
        self.assertFalse(self.manager.begin_inference())
        self.runtime.unload.assert_not_called(); cleared.assert_not_called()
        self.manager.end_inference(); self.manager.job.join(2)
        self.runtime.unload.assert_called_once(); cleared.assert_called_once()
        self.assertEqual(self.manager.active_requests,0)


if __name__ == '__main__': unittest.main()
