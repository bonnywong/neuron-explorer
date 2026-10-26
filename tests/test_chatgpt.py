"""Subscription routing checks. Run with: python -m unittest discover -s tests"""
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

os.environ.setdefault('EXPLORER_NO_MODEL', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1] / '.venv' / 'hf'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
import server


class ChatGPTLoginTests(unittest.IsolatedAsyncioTestCase):
    async def check_account(self, account, accepted):
        proc = Mock(returncode=None)
        c = {'session': 'test', 'proc': None}
        rpc = AsyncMock(side_effect=[{}, {'account': account}, {'thread': {'id': 'test-thread'}}])
        with tempfile.TemporaryDirectory() as tmp:
            prompt = Path(tmp) / 'prompt.md'
            prompt.write_text('Neuron scientist', encoding='utf-8')
            with patch.object(server, '_codex_home', return_value=tmp), \
                 patch.object(server.asyncio, 'create_subprocess_exec', AsyncMock(return_value=proc)), \
                 patch.object(server, '_codex_pump', AsyncMock()), \
                 patch.object(server, '_rpc', rpc), \
                 patch.object(server, '_write', AsyncMock()) as write, \
                 patch.object(server, '_stop_turn', AsyncMock()) as stop, \
                 patch.object(server, '_meta', return_value={}), \
                 patch.object(server, '_meta_set'), \
                 patch.object(server, 'agent_prompt_file', return_value=str(prompt)):
                if accepted:
                    self.assertIs(await server._ensure_codex(c), proc)
                    self.assertEqual(c['thread'], 'test-thread')
                    self.assertEqual(rpc.call_args_list[-1].args[1], 'thread/start')
                    stop.assert_not_called()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Sign in with ChatGPT'):
                        await server._ensure_codex(c)
                    self.assertEqual(rpc.await_count, 2)  # no thread or model request
                    stop.assert_awaited_once_with(proc)
                    self.assertIsNone(c['proc'])
                write.assert_awaited_once_with(proc, {'jsonrpc': '2.0', 'method': 'initialized', 'params': {}})
                await c['pump']

    async def test_subscription_login(self):
        await self.check_account({'type': 'chatgpt'}, True)

    async def test_api_key_rejected(self):
        await self.check_account({'type': 'apiKey'}, False)

    async def test_missing_login_rejected(self):
        await self.check_account(None, False)


class ExplorerLoginIsolationTests(unittest.TestCase):
    def test_cli_login_is_not_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_home = Path(tmp) / 'explorer'
            cli_home = Path(tmp) / 'cli'
            cli_home.mkdir()
            (cli_home / 'auth.json').write_text('fake-cli-login', encoding='utf-8')
            with patch.object(server, 'CODEX_HOME', str(app_home)), \
                 patch.object(server.os.path, 'expanduser', side_effect=lambda path: str(cli_home / Path(path).name)):
                server._codex_home()
                self.assertFalse((app_home / 'auth.json').exists())
                self.assertEqual((cli_home / 'auth.json').read_text(), 'fake-cli-login')
                (app_home / 'auth.json').write_text('fake-app-login', encoding='utf-8')
                server._codex_home()
                self.assertEqual((app_home / 'auth.json').read_text(), 'fake-app-login')


if __name__ == '__main__':
    unittest.main()
