"""Local ChatGPT browser sign-in through Codex app-server. Never expose credentials."""
import asyncio
import json
import shutil
from urllib.parse import urlparse


class ChatGPTAuth:
    def __init__(self, home, cwd, stop_chats):
        self.home, self.cwd, self.stop_chats = home, cwd, stop_chats
        self.lock = asyncio.Lock()
        self.proc = None
        self.pending = {}
        self.serial = 0
        self.login = None
        self.error = None

    async def _write(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def _rpc(self, method, params=None):
        self.serial += 1
        key = self.serial
        future = asyncio.get_running_loop().create_future()
        self.pending[key] = future
        try:
            await self._write({"jsonrpc": "2.0", "id": key, "method": method, "params": params or {}})
            reply = await asyncio.wait_for(future, 25)
            if "error" in reply:
                raise RuntimeError(reply["error"].get("message", "ChatGPT connection failed"))
            return reply.get("result") or {}
        finally:
            self.pending.pop(key, None)

    def _notification(self, method, params):
        if method == "account/login/completed" and self.login and params.get("loginId") == self.login["loginId"]:
            self.login = None
            self.error = None if params.get("success") else (params.get("error") or "Sign-in was cancelled. Try again.")

    async def _pump(self, proc):
        try:
            async for line in proc.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                future = self.pending.get(message.get("id"))
                if future and not future.done():
                    future.set_result(message)
                elif "method" in message:
                    self._notification(message["method"], message.get("params") or {})
        finally:
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("ChatGPT connection closed. Try again."))
            if self.proc is proc:
                self.proc = None
                if self.login:
                    self.error = "Sign-in connection closed. Try again."
                    self.login = None

    async def _ensure(self):
        if self.proc and self.proc.returncode is None:
            return
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError("Install the Codex CLI, then restart the explorer to sign in.")
        import os
        self.proc = await asyncio.create_subprocess_exec(
            executable, "app-server", cwd=self.cwd,
            env={**os.environ, "CODEX_HOME": self.home()},
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=1 << 24)
        self.pump = asyncio.create_task(self._pump(self.proc))
        try:
            await self._rpc("initialize", {"clientInfo": {"name": "neuron-explorer-auth", "version": "0.1"}})
            await self._write({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        except Exception:
            await self.close()
            raise

    async def _status(self):
        result = await self._rpc("account/read", {"refreshToken": False})
        account = result.get("account") or {}
        connected = account.get("type") == "chatgpt"
        return {"connected": connected, "status": "connected" if connected else ("signing_in" if self.login else "signed_out"),
                "email": account.get("email") if connected else None,
                "plan": account.get("planType") if connected else None,
                "authUrl": self.login["authUrl"] if self.login else None,
                "error": self.error, "cliInstalled": True}

    async def status(self):
        async with self.lock:
            try:
                await self._ensure()
                return await self._status()
            except Exception as exc:
                return {"connected": False, "status": "unavailable", "email": None, "plan": None,
                        "error": str(exc), "cliInstalled": bool(shutil.which("codex"))}

    async def start_login(self):
        async with self.lock:
            await self._ensure()
            if (await self._status())["connected"]:
                return {"connected": True}
            if not self.login:
                self.error = None
                result = await self._rpc("account/login/start", {"type": "chatgpt"})
                url = urlparse(result.get("authUrl", ""))
                if url.scheme != "https" or url.hostname not in {"auth.openai.com", "auth0.openai.com", "chatgpt.com", "auth.chatgpt.com"}:
                    raise RuntimeError("The sign-in service returned an unexpected authorization URL.")
                self.login = {"loginId": result["loginId"], "authUrl": result["authUrl"]}
            return {"connected": False, "authUrl": self.login["authUrl"]}

    async def cancel_login(self):
        async with self.lock:
            if self.login:
                await self._ensure()
                await self._rpc("account/login/cancel", {"loginId": self.login["loginId"]})
                self.login = None
            self.error = None
            return {"ok": True}

    async def logout(self):
        async with self.lock:
            await self._ensure()
            if self.login:
                await self._rpc("account/login/cancel", {"loginId": self.login["loginId"]})
                self.login = None
            # Stop every chat so no child retains an authenticated session in memory.
            await self.stop_chats()
            await self._rpc("account/logout")
            self.error = None
            return {"ok": True}

    async def close(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        if hasattr(self, "pump"):
            await self.pump
        self.proc = None
