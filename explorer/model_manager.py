"""Explicit local model lifecycle. Importing this module never contacts the Hub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

MODEL = "Qwen/Qwen3.5-4B"
ROOT = Path(__file__).resolve().parents[1]
HF_HOME = Path(os.environ.get("HF_HOME", ROOT / ".model-cache")).resolve()
STATE = HF_HOME / "neuron-explorer"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value), encoding="utf-8")
    temp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def verified(manifest):
    if not manifest or not manifest.get("verified") or not manifest.get("files"):
        return False
    try:
        return manifest.get("model") == MODEL and all(Path(f["path"]).is_file() and Path(f["path"]).stat().st_size == f["size"]
                   and Path(f["path"]).stat().st_mtime_ns == f["mtime"] for f in manifest["files"])
    except (OSError, KeyError, TypeError):
        return False


class ModelManager:
    def __init__(self, runtime, restricted=False, state_dir=STATE):
        self.rt, self.restricted = runtime, restricted
        self.directory = Path(state_dir)
        self.guard = threading.RLock()
        self.idle = threading.Condition(self.guard)
        self.active_requests = 0
        self.on_stopped = lambda: None
        self.process = None
        self.job = None
        self.stage = "idle"
        self.error = None
        self.cancel = threading.Event()
        self.manifest = read_json(self.directory / "manifest.json")
        self.installed = verified(self.manifest)
        self.state = "installed" if self.installed else "not_downloaded"

    def diagnostics(self):
        import torch
        import psutil
        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()
        memory = psutil.virtual_memory()
        result = {"cuda": cuda, "mps": mps, "torch": torch.__version__,
                  "ram_available": memory.available, "device": "cuda" if cuda else "mps" if mps else "cpu"}
        if cuda:
            free, total = torch.cuda.mem_get_info()
            result.update(gpu=torch.cuda.get_device_name(), memory_free=free, memory_total=total,
                          allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        else:
            result.update(gpu=None, memory_free=memory.available, memory_total=memory.total)
        return result

    def status(self):
        with self.guard:
            if self.process and self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                self.manifest = read_json(self.directory / "manifest.json")
                self.installed = verified(self.manifest)
                progress = read_json(self.directory / "progress.json", {})
                self.state = "installed" if code == 0 and self.installed else "error"
                self.stage = "Download verified" if self.state == "installed" else "Download failed"
                self.error = None if self.state == "installed" else progress.get("error", "Download interrupted. Retry to resume.")
            folder = HF_HOME
            while not folder.exists():
                folder = folder.parent
            progress = read_json(self.directory / "progress.json", {})
            diagnostics = self.diagnostics()
            return {"model": MODEL, "state": self.state, "stage": self.stage,
                    "restricted": self.restricted, "installed": self.installed,
                    "ready": self.state == "ready", "error": self.error,
                    "cache_path": str(HF_HOME), "disk_free": shutil.disk_usage(folder).free,
                    "revision": (self.manifest or {}).get("revision"),
                    "download": progress, "diagnostics": diagnostics, "estimated_download_bytes": 9342816694,
                    "device": getattr(self.rt, "dev", None), "precision": getattr(self.rt, "precision", None),
                    "hook_count": getattr(self.rt, "hook_count", 0),
                    "capabilities": {"live": self.state == "ready", "cached": True},
                    "restriction_reason": "Live execution is disabled by EXPLORER_NO_MODEL. Restart using the normal launcher." if self.restricted else None}

    def download(self):
        with self.guard:
            self.status()
            if self.restricted:
                raise ValueError("Downloads are disabled in explicit cache-only mode. Restart using the normal launcher.")
            if self.state in {"downloading", "loading", "stopping", "ready"}:
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            self.error = None
            write_json(self.directory / "progress.json", {"stage": "Fetching download information", "downloaded": 0, "total": None})
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("model_download.py")), str(self.directory)],
                                            env={**os.environ, "HF_HOME": str(HF_HOME)},
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.state = "downloading"

    def cancel_download(self):
        with self.guard:
            if self.state != "downloading":
                return
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.process = None
            self.state = "installed" if self.installed else "not_downloaded"
            self.error = "Download cancelled. Retry/Resume reuses cached files."

    def start(self, device="auto"):
        with self.guard:
            self.status()
            if self.restricted:
                raise ValueError("Live execution is disabled by EXPLORER_NO_MODEL. Use the normal launcher.")
            if self.state in {"ready", "loading", "stopping", "downloading"}:
                return
            self.installed = verified(self.manifest)
            if not self.installed:
                raise ValueError("Download and verify Qwen before starting it.")
            info = self.diagnostics()
            if device == "auto":
                device = info["device"]
                if device == "cpu":
                    raise ValueError("No accelerator is available. Run setup to enable GPU support, or explicitly select CPU (slow).")
            if device not in {"cpu", "cuda", "mps"} or (device != "cpu" and not info.get(device)):
                raise ValueError("The selected accelerator is unavailable. Run setup or select CPU explicitly.")
            self.state, self.stage, self.error = "loading", "Loading local weights", None
            self.cancel.clear()
            self.rt.cancel = self.cancel
            self.job = threading.Thread(target=self._load, args=(device,), daemon=True)
            self.job.start()

    def _load(self, device):
        failure = None
        try:
            self.rt.load(self.manifest["snapshot"], device)
            self.stage = "Checking inference and neuron hooks"
            self.rt.warmup()
            with self.guard:
                if not self.cancel.is_set():
                    self.state, self.stage = "ready", "Ready"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}. Free device memory or check setup, then retry."
        # Leave the exception scope first: tracebacks can otherwise retain failed GPU allocations.
        if failure:
            self.rt.unload()
            with self.guard:
                if not self.cancel.is_set():
                    self.state, self.error = "error", failure

    def stop(self):
        with self.guard:
            if self.state == "downloading":
                self.cancel_download()
                return
            if self.state == "stopping":
                return
            previous = self.job
            self.cancel.set()
            self.state, self.stage = "stopping", "Waiting for computation to finish"
            self.job = threading.Thread(target=self._unload, args=(previous,), daemon=True)
            self.job.start()

    def _unload(self, previous):
        if previous and previous.is_alive():
            previous.join()
        with self.idle:
            self.idle.wait_for(lambda: self.active_requests == 0)
        self.rt.unload()
        self.on_stopped()
        with self.guard:
            self.state, self.stage, self.error = "installed" if self.installed else "not_downloaded", "Stopped", None

    def close(self):
        self.stop()
        if self.job and self.job.is_alive():
            self.job.join()

    def begin_inference(self):
        with self.guard:
            if self.state != 'ready' or self.cancel.is_set():
                return False
            self.active_requests += 1
            return True

    def end_inference(self):
        with self.idle:
            self.active_requests -= 1
            self.idle.notify_all()


LIVE_PATHS = {"/api/reads", "/api/steer", "/api/fires", "/api/infer", "/api/infer/view", "/api/infer/heads",
              "/api/infer/paths", "/api/neuron/heads", "/api/head/row", "/api/head/pattern", "/api/head/profile", "/api/head",
              "/api/agent/steer", "/api/agent/activations", "/api/agent/attention", "/api/agent/attn_drive",
              "/api/agent/logprobs", "/api/agent/compare", "/api/agent/fires"}
