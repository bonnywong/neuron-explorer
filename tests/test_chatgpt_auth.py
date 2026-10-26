"""Auth lifecycle checks without credentials, a browser, or model inference."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
from chatgpt_auth import ChatGPTAuth


class AuthLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def make_auth(self):
        auth = ChatGPTAuth(lambda: 'unused', 'unused', AsyncMock())
        auth._ensure = AsyncMock()
        return auth

    async def test_status_exposes_only_account_summary(self):
        auth = self.make_auth()
        auth._rpc = AsyncMock(return_value={'account': {'type': 'chatgpt', 'email': 'test@example.com', 'planType': 'plus', 'accessToken': 'secret'}})
        result = await auth.status()
        self.assertTrue(result['connected'])
        self.assertEqual(result['email'], 'test@example.com')
        self.assertNotIn('secret', str(result))

    async def test_login_reuses_pending_flow_and_completion_clears_it(self):
        auth = self.make_auth()
        auth._rpc = AsyncMock(side_effect=[{'account': None}, {'loginId': 'test', 'authUrl': 'https://auth.openai.com/authorize'}, {'account': None}])
        first = await auth.start_login()
        self.assertEqual(first, await auth.start_login())
        self.assertEqual(auth._rpc.await_count, 3)
        auth._notification('account/login/completed', {'loginId': 'other', 'success': True})
        self.assertIsNotNone(auth.login)
        auth._notification('account/login/completed', {'loginId': 'test', 'success': True})
        self.assertIsNone(auth.login)
        self.assertIsNone(auth.error)

    async def test_failed_login_shows_error(self):
        auth = self.make_auth()
        auth.login = {'loginId': 'test'}
        auth._notification('account/login/completed', {'loginId': 'test', 'success': False, 'error': 'Login expired'})
        self.assertEqual(auth.error, 'Login expired')
        self.assertIsNone(auth.login)

    async def test_cancel_calls_official_cancel_method(self):
        auth = self.make_auth()
        auth.login = {'loginId': 'test'}
        auth._rpc = AsyncMock()
        await auth.cancel_login()
        auth._rpc.assert_awaited_once_with('account/login/cancel', {'loginId': 'test'})
        self.assertIsNone(auth.login)

    async def test_logout_stops_chats_before_clearing_credentials(self):
        auth = self.make_auth()
        calls = []
        auth.stop_chats = AsyncMock(side_effect=lambda: calls.append('stop'))
        auth._rpc = AsyncMock(side_effect=lambda *args: calls.append(args[0]))
        await auth.logout()
        self.assertEqual(calls, ['stop', 'account/logout'])

    async def test_reject_unexpected_login_url(self):
        auth = self.make_auth()
        auth._rpc = AsyncMock(side_effect=[{'account': None}, {'loginId': 'test', 'authUrl': 'https://example.com/phishing'}])
        with self.assertRaisesRegex(RuntimeError, 'unexpected authorization URL'):
            await auth.start_login()
        self.assertIsNone(auth.login)

    async def test_api_key_is_not_connected(self):
        auth = self.make_auth()
        auth._rpc = AsyncMock(return_value={'account': {'type': 'apiKey'}})
        self.assertFalse((await auth.status())['connected'])


if __name__ == '__main__':
    unittest.main()
