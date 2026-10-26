"""One-time dependency setup, then offline launches without uv replacing GPU PyTorch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
WINDOWS = os.name == 'nt'
PYTHON = ROOT / '.venv' / ('Scripts/python.exe' if WINDOWS else 'bin/python')
STATE = ROOT / '.setup-state.json'
TORCH = '2.14.1'
CUDA = 'cu130'


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True, **kwargs)


def fingerprint():
    return hashlib.sha256((ROOT / 'uv.lock').read_bytes() + (ROOT / 'pyproject.toml').read_bytes()).hexdigest()


def verify(device):
    # Exercise the actual backend, not just nvidia-smi or a compiled-in backend flag.
    code = """
import json, torch
device = DEVICE
if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
if device == 'cpu' and DEVICE != 'cpu':
    raise RuntimeError('No usable accelerator. Install a compatible driver, or rerun the launcher with --cpu.')
x = torch.randn(32,32,device=device)
y = x @ x
assert torch.isfinite(y).all().item()
print(json.dumps({'device':device, 'torch':torch.__version__, 'verification':'matrix multiplication passed'}))
""".replace('DEVICE', repr(device))
    result = run([PYTHON, '-c', code], capture_output=True, text=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def setup(cpu=False, force=False, verify_only=False):
    if sys.version_info < (3, 11):
        raise RuntimeError('Install Python 3.11 or newer (3.12 recommended), then rerun the launcher.')
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    if previous.get('lock') == fingerprint() and PYTHON.exists() and not force and not verify_only and (not cpu or previous.get('device') == 'cpu'):
        print(f"Using verified {previous['device']} environment; installed packages are unchanged.")
        return
    if not verify_only:
        uv = shutil.which('uv')
        if not uv:
            bootstrap = ROOT / '.bootstrap'
            venv.EnvBuilder(with_pip=True).create(bootstrap)
            bootstrap_python = bootstrap / ('Scripts/python.exe' if WINDOWS else 'bin/python')
            run([bootstrap_python, '-m', 'pip', 'install', 'uv==0.5.3'])
            uv = bootstrap / ('Scripts/uv.exe' if WINDOWS else 'bin/uv')
        run([uv, 'sync', '--frozen', '--python', sys.executable, '--cache-dir', ROOT / '.venv/uv-cache'])
        if not cpu and shutil.which('nvidia-smi'):
            # This wheel is an intentional platform override to the cross-platform uv lock.
            run([uv, 'pip', 'install', '--python', PYTHON, '--cache-dir', ROOT / '.venv/uv-cache',
                 f'torch=={TORCH}+{CUDA}', '--index-url', f'https://download.pytorch.org/whl/{CUDA}'])
    result = verify('cpu' if cpu else 'auto')
    STATE.write_text(json.dumps({**result,'lock':fingerprint()},indent=2))
    print(f"Verified {result['device']} with PyTorch {result['torch']}. Model weights are downloaded later in the browser.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu',action='store_true',help='Explicit slow CPU mode')
    parser.add_argument('--repair',action='store_true',help='Reinstall locked dependencies and chosen backend')
    parser.add_argument('--verify-only',action='store_true',help='Validate an already installed project environment')
    args = parser.parse_args()
    try:
        setup(args.cpu,args.repair,args.verify_only)
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'\nSetup failed: {error}',file=sys.stderr)
        if getattr(error,'stderr',None): print(error.stderr,file=sys.stderr)
        print('For CUDA: install a current NVIDIA driver supporting CUDA 13, restart, then run with --repair. '
              'For Apple: use a supported Apple Silicon/macOS PyTorch environment. '
              'Or explicitly run with --cpu. The browser never installs packages or drivers.',file=sys.stderr)
        sys.exit(1)
