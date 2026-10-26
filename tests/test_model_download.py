"""Small fixture files exercise download verification without network/model inference."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
import model_download as worker
from model_manager import read_json, write_json, verified, MODEL


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.state=self.root/'state'; self.state.mkdir()
        self.snapshot=self.root/'snapshot'; self.snapshot.mkdir()
        self.file=self.snapshot/'model.safetensors'; self.file.write_bytes(b'fixture')
        self.info=SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename='model.safetensors',size=7,
            lfs=SimpleNamespace(sha256=hashlib.sha256(b'fixture').hexdigest()))])
        self.hub=Mock(); self.hub.model_info.return_value=self.info
        self.stack=[]
        for target,value in [('HF_HOME',self.root),('HfApi',Mock(return_value=self.hub)),('hf_hub_download',Mock(return_value=str(self.file)))]:
            patcher=patch.object(worker,target,value); patcher.start(); self.addCleanup(patcher.stop)

    def test_low_disk_fails_before_file_transfer(self):
        with patch.object(worker.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError,'free disk'): worker.download(self.state)
        worker.hf_hub_download.assert_not_called()
        self.assertFalse((self.state/'manifest.json').exists())

    def test_bad_checksum_does_not_install(self):
        self.info.siblings[0].lfs.sha256='bad'
        with self.assertRaisesRegex(RuntimeError,'Checksum'): worker.download(self.state)
        self.assertFalse((self.state/'manifest.json').exists())

    def test_incomplete_file_does_not_install(self):
        self.info.siblings[0].size=8
        with self.assertRaisesRegex(RuntimeError,'Incomplete'): worker.download(self.state)
        self.assertFalse((self.state/'manifest.json').exists())

    def test_resume_retains_resolved_revision_and_verifies_before_install(self):
        write_json(self.state/'download-plan.json',{'revision':'a'*40})
        context=Mock(); context.__enter__=Mock(return_value=Mock(keys=Mock(return_value=['fixture']))); context.__exit__=Mock(return_value=False)
        with patch('safetensors.safe_open',return_value=context): worker.download(self.state)
        self.hub.model_info.assert_called_once_with(MODEL,revision='a'*40,files_metadata=True)
        manifest=read_json(self.state/'manifest.json')
        self.assertTrue(verified(manifest)); self.assertEqual(manifest['revision'],'a'*40)
        self.assertEqual(read_json(self.state/'progress.json')['downloaded'],7)

    def test_interruption_keeps_revision_without_installing(self):
        worker.hf_hub_download.side_effect=ConnectionError('interrupted')
        with self.assertRaises(ConnectionError): worker.download(self.state)
        self.assertEqual(read_json(self.state/'download-plan.json')['revision'],'a'*40)
        self.assertFalse((self.state/'manifest.json').exists())


if __name__ == '__main__': unittest.main()
