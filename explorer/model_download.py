"""Cancellable download worker. Only a completely verified snapshot is installed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from model_manager import MODEL, HF_HOME, read_json, write_json

def verify_file(path, metadata):
    if path.stat().st_size != metadata.size:
        raise RuntimeError(f"Incomplete model file: {metadata.rfilename}")
    if metadata.lfs and metadata.lfs.sha256:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != metadata.lfs.sha256:
            raise RuntimeError(f"Checksum failed for {metadata.rfilename}. Retry the download to repair it.")
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        with safe_open(str(path), framework='np') as tensors:
            if not list(tensors.keys()):
                raise RuntimeError('Empty model weights')


def download(directory):
    directory = Path(directory)
    progress_path = directory / "progress.json"
    old = read_json(directory / "manifest.json", {}) or read_json(directory / "download-plan.json", {})
    info = HfApi().model_info(MODEL, revision=old.get("revision"), files_metadata=True)
    write_json(directory / "download-plan.json", {"revision": info.sha})
    files = [f for f in info.siblings if "/" not in f.rfilename and
             (f.rfilename.endswith((".json", ".safetensors", ".txt", ".model", ".jinja")))]
    total = sum(f.size or 0 for f in files)
    cache = HF_HOME / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    progress = {"stage": "Downloading model files", "downloaded": 0, "total": total, "revision": info.sha}
    write_json(progress_path, progress)
    completed = 0
    stop = threading.Event()
    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['disable'] = True
            super().__init__(*args, **kwargs)
            progress['downloaded'] = completed + self.n
        def update(self, amount=1):
            self.n += amount
            progress['downloaded'] = min(total, completed + self.n)
        def update_transfer(self, amount):
            progress['transferred'] = progress.get('transferred', 0) + amount
        def set_transfer_postfix_str(self, *args, **kwargs):
            pass
    def report():
        while not stop.wait(.5):
            write_json(progress_path, progress.copy())
    # Cached downloads require no extra disk; conservative bound for missing files.
    missing = sum(f.size or 0 for f in files if not (cache / "models--Qwen--Qwen3.5-4B" / "snapshots" / info.sha / f.rfilename).exists())
    if shutil.disk_usage(cache).free < missing + 512 * 1024**2:
        raise RuntimeError("Not enough free disk space for the model plus 512 MB working space.")
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    records = []
    try:
        for f in files:
            for repair in (False, True):
                p = Path(hf_hub_download(MODEL, f.rfilename, revision=info.sha, cache_dir=str(cache), tqdm_class=Progress, force_download=repair))
                progress["stage"] = f"Verifying {f.rfilename}"
                try:
                    verify_file(p, f)
                    break
                except Exception:
                    if repair: raise
                    progress["stage"] = f"Repairing {f.rfilename}"
            records.append({"path": str(p), "size": p.stat().st_size, "mtime": p.stat().st_mtime_ns})
            completed += f.size or 0
            progress['downloaded'] = completed
            progress["stage"] = "Downloading model files"
        if not any(f.rfilename.endswith(".safetensors") for f in files):
            raise RuntimeError("No model weights found")
        write_json(directory / "manifest.json", {"model": MODEL, "revision": info.sha, "verified": True,
                   "snapshot": str(p.parent), "files": records})
    finally:
        stop.set()
        thread.join()
    write_json(progress_path, {**progress, "stage": "Verified", "downloaded": total})


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    try:
        download(folder)
    except Exception as exc:
        old = read_json(folder / "progress.json", {})
        write_json(folder / "progress.json", {**old, "stage": "Failed", "error": str(exc)})
        raise
