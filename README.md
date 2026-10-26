# neuron explorer

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

Install Python 3.12 (3.11+ supported), then run from the project directory:

```powershell
# Windows PowerShell
.\launch.ps1
# Explicit CPU mode: .\launch.ps1 -Cpu
# Repair dependencies / change backend: .\launch.ps1 -Repair
```

```bash
# macOS / Linux
bash launch.sh
# Explicit CPU mode: bash launch.sh --cpu
# Repair: bash launch.sh --repair
```

The first launch creates `.venv`, installs `uv.lock` dependencies, selects CUDA on NVIDIA or MPS on Apple Silicon,
and verifies the backend with an actual matrix multiplication. Windows/Linux CUDA uses the official
`torch==2.14.1+cu130` wheel as an explicit platform override; a current CUDA 13-compatible NVIDIA driver is required.
No driver is installed by the app. CPU must be selected explicitly and uses float32, so allow extra RAM and time.
Subsequent launches use `.venv` directly and preserve the verified backend. Avoid `uv run` / `uv sync` for normal
launches: they can replace the CUDA override. If the lock changes, setup runs again.

Open **http://localhost:8791**, choose **Model & setup → Download Qwen**, then **Start Qwen**. No weights download
or load during web startup. The interface and bundled reference data work offline. The first model download needs
about 8.70 GiB plus working space; its exact size and progress are measured. CUDA/MPS use supported accelerator
precision, with no silent fallback. The Ready state requires a real inference and neuron-hook check.

Model files use the Hugging Face cache under `HF_HOME` (default `.model-cache` for these launchers). The first
download pins a commit and records it in `HF_HOME/neuron-explorer/`. Cancellation keeps reusable partial data;
Retry / Resume retains the revision. A complete existing cache is reused and verified. Model start uses only
verified local files. Stop ends active generation cooperatively, waits for computation and releases model memory;
downloaded files remain. `EXPLORER_NO_MODEL=1` explicitly restricts the app to cached exploration. Launchers do not
set this flag; remove it from your shell if you want live execution.

**Help → Start / Resume guided demo** walks through a live, three-column steering comparison on L22 N2742 and
an activation run. Each comparison column is generated sequentially, up to 96 tokens, only after pressing Run.
No ChatGPT login is required. The tour preserves its step/results locally, never repeats inference after a reload,
and restores the previous selection and neuron conversation on exit. **Help & glossary** and contextual `?`
buttons explain the controls without a model or assistant connection.

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Explicitly starts Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **ChatGPT / Claude** (fourth neuron-panel tab, or the right-edge handle): a chat using the `codex` CLI with your
  ChatGPT login by default, or the `claude` CLI when selected, with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

### ChatGPT subscription chat

The assistant panel defaults to **ChatGPT**, with **ChatGPT (via Codex)** selected in
the provider menu. It uses the local Codex CLI signed in with your ChatGPT account,
and counts toward your plan's Codex usage limits. No API key is needed; API-key
logins are rejected for this option. Claude remains available in the provider menu.

Install the [Codex CLI](https://learn.chatgpt.com/docs/cli) and start the explorer.
Open the **AI assistant** panel (or the right-edge handle), click **Sign in with ChatGPT**,
and finish signing in in the browser tab. The panel shows the connected account
and plan. If a popup is blocked, use **Continue sign-in**. Pending sign-ins can be
cancelled; connection failures show a retry button and missing CLI installations
show an installation link.

**Sign out** stops running ChatGPT conversations and removes this explorer's login.
It does not sign you out of the Codex desktop app or CLI. Credentials are stored
only in the explorer's Git-ignored `explorer/cache/agent/codex-home/`; the app no
longer imports your CLI login automatically. A login saved by the previous version
is retained, with any credential symlink detached before use.

This remains a personal, local application: browser tabs on the same explorer
server share its login. Separate users should run their own local instances.
ChatGPT can inspect the bundled cached data in cache-only mode; live model
experiments still require Qwen model weights and an enabled runtime.

### Model lifecycle and validation

`GET /api/model/status` reports lifecycle state, diagnostics, capabilities, measured download bytes, revision and
errors. `POST /api/model/download`, `/api/model/download/cancel`, `/api/model/start` (`{"device":"auto"}` or
explicit `cuda`, `mps`, `cpu`), and `/api/model/stop` control background operations. Poll status until complete.
`GET /api/status` remains supported. Browser and assistant live endpoints return `model_not_ready` and an
`open_model_setup` action when unavailable. Sweep/embedding availability is separate from model readiness.

Run `python -m unittest discover -s tests -v` in the project environment for regression tests. The CI lifecycle
tests use mocked downloads/inference and do not fetch weights. See [the real-model acceptance procedure](docs/model-acceptance.md)
for the separate hardware/browser checks.

Reference: [official Qwen Transformers model](https://huggingface.co/Qwen/Qwen3.5-4B),
[Hugging Face versioned downloads and cache](https://huggingface.co/docs/huggingface_hub/guides/download).

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.
