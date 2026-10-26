# Local model acceptance

Automated checks deliberately do not download model weights. Run:

```text
.venv/Scripts/python -m unittest discover -s tests -v
node --test tests/experiments.test.mjs
```

On macOS/Linux use `.venv/bin/python`. CI runs the standard-library lifecycle tests, small mocked download
fixtures, authentication unit tests and shared JavaScript experiment tests; full API tests use the project environment.

## Browser and hardware procedure

1. Run the normal launcher. Confirm its actual backend computation passes. Relaunch and confirm packages are
   not synchronized again. With network disconnected and no model weights, the interface, map, lenses and help
   must open. A cached snapshot must start without internet.
2. On a fresh browser profile, use the welcome panel. Inspect the three independent model/data/assistant cards.
   Leave ChatGPT disconnected. Open Help with the keyboard; search for multiplicative, dismiss with Escape,
   and verify focus returns. Repeat at a narrow viewport and with reduced motion enabled.
3. Download Qwen. Confirm measured bytes, total size, destination and pinned revision. Cancel midway: Start must
   stay unavailable and files remain reusable. Resume from another tab; repeated clicks must not spawn another
   download. Restart the app and confirm a verified download is still installed without loading it.
4. Start Qwen. Check the named loading/warm-up stages, Ready, actual device/precision and available memory.
   Use `nvidia-smi` alongside `/api/model/status` to verify GPU computation and allocation, not detection alone.
5. Start the tour, visit L22 N2742, and inspect reference data. Explicitly run “Who are you?” with additive 0.8.
   Observe all three columns streaming sequentially, each capped at 96 tokens. Check the experiment record,
   including whether normalization came from a local reference passage. Similar or unexpected answers are valid.
6. Advance to activations and run editable text. Inspect token values with hover and keyboard focus. Try Back,
   Next, Skip and Exit; verify original neuron, settings and chat return. Reload during an experiment, resume
   via Help, and ensure no inference repeats automatically. Restart must return to Prepare.
7. Start a new run, cancel it, and check the partial label. Start another run and Stop Qwen from setup. The tour
   must pause, streaming must end cooperatively, and allocation must return close to baseline after computation exits
   (small backend caches and the CUDA context may remain).
   Start again: hook count must equal its earlier value, with no stale captures. Run activations again.
8. Verify CPU selection is explicit. Simulate unavailable CUDA, low disk, missing/corrupt shards, interrupted
   transfer and a model-load/OOM error with the mocked tests; never corrupt a user's real cached model to test.

## Windows RTX 4070 Ti run

Initial acceptance on 2026-10-26 used Python 3.12.4, PyTorch 2.14.1+cu130, NVIDIA RTX 4070 Ti (12 GiB),
and Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The browser downloaded 9,342,816,694 bytes,
cancelled/resumed the transfer, and started verified local weights with CUDA/bfloat16. Real inference and all-layer
neuron capture passed. The three sequential outputs completed, followed by live activation highlighting.
GPU utilization reached 99%. After Stop, `nvidia-smi` reported 1,380 MiB used / 10,615 MiB free (including desktop
and CUDA context memory). Reference PyTorch kernels were used for Qwen's linear attention; optional custom CUDA
extensions are not required. macOS/MPS and Linux remain supported paths requiring their own hardware acceptance.

The final runtime was also started with `HF_HUB_OFFLINE=1`. It reached Ready with 184 hooks and
9,188,792,320 allocated CUDA bytes. Stopping during generation reduced the hook count to zero and allocation
to 8,519,680 bytes (backend buffers), releasing the model weights and model-dependent caches.
Restart restored Ready with exactly 184 hooks. Manual browser checks also covered safe reload/resume,
restoring the previous L19 N4445 selection on exit, cancellation, a keyboard-started tour, and a 390px help
drawer whose Escape key restored focus to Help. The final automated run passed 32 Python tests and 5 JavaScript tests.
