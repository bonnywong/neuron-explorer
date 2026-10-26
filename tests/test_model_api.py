"""Integration checks for a cache-only server: no model load or network requests."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault('EXPLORER_NO_MODEL','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'explorer'))
import server
from fastapi.testclient import TestClient
from model_manager import LIVE_PATHS


class ModelAPITests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(server.app,base_url='http://localhost:8791')
        self.addCleanup(self.client.close)

    def test_all_live_browser_and_assistant_routes_share_capability_error(self):
        with patch.object(server.models,'state','not_downloaded'):
            for path in LIVE_PATHS:
                with self.subTest(path=path):
                    result=self.client.post(path,json={})
                    self.assertEqual(result.status_code,409)
                    self.assertEqual(result.json(),server.unavailable())

    def test_offline_index_and_cached_lens_remain_available(self):
        with patch.object(server.rt,'load',side_effect=AssertionError('must not load')):
            self.assertEqual(self.client.get('/api/index').status_code,200)
            self.assertEqual(self.client.get('/api/lens?layer=22&n=2742').status_code,200)
            self.assertEqual(self.client.get('/vendor/three/build/three.module.js').status_code,200)

    def test_restricted_lifecycle_is_actionable(self):
        with patch.object(server.models,'restricted',True):
            result=self.client.post('/api/model/start',json={})
            self.assertEqual(result.status_code,409)
            self.assertIn('EXPLORER_NO_MODEL',result.json()['error'])

    def test_external_origins_cannot_operate_lifecycle(self):
        result=self.client.post('/api/model/stop',json={},headers={'Origin':'https://example.com'})
        self.assertEqual(result.status_code,403)

    def test_live_request_releases_its_lifecycle_lease(self):
        with patch.object(server.models,'state','ready'), patch.object(server.rt,'read',return_value={'units':[]}):
            result=self.client.post('/api/reads',json={'layer':22,'n':2742,'text':'Hello'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(server.models.active_requests,0)

    def test_stream_failure_releases_lease_and_surfaces_error(self):
        with patch.object(server.models,'state','ready'):
            result=self.client.post('/api/steer',json={'layer':22,'n':2742,'prompt':'Hello','dose':.8})
            self.assertIn('error',result.text)
            self.assertEqual(server.models.active_requests,0)


if __name__ == '__main__': unittest.main()
