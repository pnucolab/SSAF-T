# Installation

SSAF-T is distributed as source on GitHub only; there is no PyPI or conda package. You build it
into a virtual environment of your own.

## Requirements

| | CPU install | GPU install (verified) |
|---|---|---|
| OS | Linux (x86_64 or ARM), macOS | Linux x86_64 |
| Python | 3.11 recommended | **3.11** exactly |
| CUDA | - | **13.x** driver |
| Build tools | a C++17 compiler; `meson`, `ninja` and `meson-python` are pulled in by the build | same |
| Package manager | [uv](https://docs.astral.sh/uv/) | uv |

The GPU stack (RAPIDS `cuml`/`cugraph`/`cuvs`, `rapids-singlecell`, `cupy`, `torch`, `triton`) is
only verified on **Python 3.11 + CUDA 13**. Other combinations may resolve, but they are not tested
and are not supported by the [agent skill](agent.md).

A CUDA GPU is effectively required for real datasets: the KDE, the mapping and the Leiden
clustering all run on it. The CPU install is useful for inspecting stores and for small crops.

## 1. Get the source

```bash
git clone --recurse-submodules https://github.com/pnucolab/SSAF-T.git
```

`--recurse-submodules` also fetches `SSAF-T/CeLLMan`, the annotation skill and agent. If you
cloned without it, run `git submodule update --init` inside `SSAF-T/`.

## 2. Create the environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

## 3. Install

Pick **one**.

::::{tab-set}

:::{tab-item} GPU, pinned (recommended)
Reproduces the environment the tutorials were run in. Every dependency is pinned in
`requirements-lock.txt`; SSAF-T itself is then installed without re-resolving.

```bash
uv pip install -r SSAF-T/requirements-lock.txt \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps ./SSAF-T
```
:::

:::{tab-item} GPU, ranges
Resolves the compatible ranges declared in `pyproject.toml`. Use this when SSAF-T has to share the
environment with other packages.

```bash
uv pip install './SSAF-T[gpu]' \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128
```
:::

:::{tab-item} CPU only
```bash
uv pip install ./SSAF-T
```
:::

::::

### CPUs without AVX-512

The C extension is built with AVX-512 enabled on every x86_64 host without checking the CPU. On a
machine that lacks it the first call into the extension dies with *illegal instruction*. Check:

```bash
lscpu | grep -o avx512f | head -1     # prints nothing -> no AVX-512
```

If it prints nothing (most consumer Ryzen and Core CPUs, every Intel Mac), append
`--config-settings=setup-args=-Dsimd=disabled` to the install command you chose. ARM hosts need
nothing.

### Editable install (development)

```bash
uv pip install --no-build-isolation -e './SSAF-T[dev]'
```

`--no-build-isolation` is required: meson-python rebuilds the extension on import, and it can only
do so when the build tools live in the same environment.

## 4. Verify

```bash
python - <<'PY'
import ssaft, ssaft.utils
print("ssaft", ssaft.__version__)
try:
    import torch, rapids_singlecell
    print("CUDA", torch.cuda.is_available(), torch.version.cuda, "| GPUs", torch.cuda.device_count())
except ImportError:
    print("GPU stack not installed (CPU install)")
PY
```

```{note}
CUDA device numbering follows PCI bus order in PyTorch but may differ from the order printed by
`nvidia-smi`. Check `torch.cuda.get_device_name(i)` before choosing `gpu_device='cuda:1'`.
```

## 5. Register CeLLMan with your LLM host

The annotation stage is run by an LLM host (Claude Code or Codex), not by Python. Copy the skill and
the expert agent from the submodule:

::::{tab-set}

:::{tab-item} Claude Code
```bash
mkdir -p .claude/skills .claude/agents
cp -r SSAF-T/CeLLMan/skills/annotate-mode .claude/skills/
cp SSAF-T/CeLLMan/agents/annotate-expert.md .claude/agents/
```
:::

:::{tab-item} Codex
```bash
mkdir -p .agents/skills .codex/agents
cp -r SSAF-T/CeLLMan/skills/annotate-mode .agents/skills/
cp SSAF-T/CeLLMan/agents/annotate-expert.toml .codex/agents/
printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> .codex/config.toml
```
:::

::::

The annotator reads the Cell-Ontology knowledge graph from its public server
(`https://cellman.pnucolab.com`); no local database is needed. Keep exactly one CeLLMan clone in
the directory tree the host runs from: the skill locates its helper by searching for
`*CeLLMan/scripts/helper.py`.
