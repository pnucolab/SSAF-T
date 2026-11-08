# Installation

SSAF-T is distributed as source on GitHub only; no PyPI or conda package is provided. It is built
into a dedicated virtual environment.

## Requirements

| | CPU install | GPU install (verified) |
|---|---|---|
| OS | Linux (x86_64 or ARM), macOS | Linux x86_64 |
| Python | 3.11 recommended | **3.11** exactly |
| CUDA | - | **13.x** driver |
| Build tools | a C++17 compiler; `meson`, `ninja` and `meson-python` are pulled in by the build | same |
| Package manager | [uv](https://docs.astral.sh/uv/) | uv |

The GPU stack (RAPIDS `cuml`/`cugraph`/`cuvs`, `rapids-singlecell`, `cupy`, `torch`, `triton`) has
been verified only on **Python 3.11 + CUDA 13**. Other combinations may resolve but are untested and
are not supported.

A CUDA GPU is effectively required for datasets of realistic size: the kernel density estimate, the
mapping and the Leiden clustering are all performed on it. The CPU installation is intended for
inspecting stores and for small crops.

## 1. Get the source

```bash
git clone --recurse-submodules https://github.com/pnucolab/SSAF-T.git
```

`--recurse-submodules` also fetches `SSAF-T/CeLLMan`, the annotation skill and agent. If the
repository was cloned without it, run `git submodule update --init` inside `SSAF-T/`.

## 2. Create the environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

## 3. Install

Select **one**.

::::{tab-set}

:::{tab-item} GPU, pinned (recommended)
Reproduces the environment in which the tutorials were run. Every dependency is pinned in
`requirements-lock.txt`, and SSAF-T itself is then installed without re-resolving.

```bash
uv pip install -r SSAF-T/requirements-lock.txt \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps ./SSAF-T
```
:::

:::{tab-item} GPU, ranges
Resolves the compatible ranges declared in `pyproject.toml`. This option is appropriate when SSAF-T
must share the environment with other packages.

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

The C extension is built with AVX-512 enabled on every x86_64 host, without checking the CPU. On a
machine that lacks AVX-512, the first call into the extension terminates with *illegal instruction*.
This is checked as follows:

```bash
lscpu | grep -o avx512f | head -1     # prints nothing -> no AVX-512
```

If nothing is printed (most consumer Ryzen and Core CPUs, and every Intel Mac), append
`--config-settings=setup-args=-Dsimd=disabled` to the chosen install command. ARM hosts require no
such adjustment.

### Editable install (development)

This is applied on top of an environment that already carries the dependencies (for example after
the GPU install above); the `[dev]` extra adds only the test tools, not the GPU stack. The build
backend must be present in that environment first, because `--no-build-isolation` has meson-python
rebuild the extension on import from the tools found there:

```bash
uv pip install meson-python meson ninja
uv pip install --no-build-isolation -e './SSAF-T[dev]'
```

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
CUDA device numbering in PyTorch follows PCI bus order and may differ from the order printed by
`nvidia-smi`. Verify `torch.cuda.get_device_name(i)` before selecting `gpu_device='cuda:1'`.
```

## 5. Register CeLLMan with the language-model host

The annotation stage is run by a language-model host (Claude Code or Codex) rather than by Python.
Copy the skill and the expert agent from the submodule:

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

The annotator reads the knowledge graph from its public server (`https://cellman.pnucolab.com`), so
no local database is required. Exactly one CeLLMan clone must be present in the directory tree from
which the host runs, because the skill locates its helper by searching for
`*CeLLMan/scripts/helper.py`.
