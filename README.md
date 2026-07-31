# SSAF-T

**Spatial Segmentation-free Agentic Framework - Transcriptomics**

Assigning cell types from spatial transcriptomics usually depends on cell segmentation, which is
difficult to achieve accurately. SSAF-T removes that dependency: transcripts are turned into a
kernel-density vector field and cell identity is inferred from mRNA signals alone — no segmentation,
no neighborhood aggregation.

The annotation step is agentic and ships as its own repository,
[**CeLLMan**](https://github.com/pnucolab/CeLLMan), linked here as the [`CeLLMan/`](CeLLMan)
submodule: one specialized agent per organism and organ mines PubMed into a knowledge graph of
cell-type marker genes and infers cell identity from the sampled mRNA signals.

<p align="center">
  <img src="ssaft.png" alt="SSAF-T overview" width="900">
</p>

---

## Installation

```bash
git clone --recurse-submodules https://github.com/pnucolab/SSAF-T.git
uv venv .venv --python 3.11
source .venv/bin/activate
```

Then run **one** of the two commands below. 

```bash
# CPU only
uv pip install ./SSAF-T
```

```bash
# same as above, plus GPU acceleration.
uv pip install './SSAF-T[gpu]'
```

Either command builds the C extension, and the default build enables AVX-512 on any x86_64 host
without checking whether the CPU supports it — a machine that lacks it aborts with an illegal
instruction the first time the extension runs. Append `--config-settings=setup-args=-Dsimd=disabled`
to whichever command you chose if either of these applies:

- **x86_64 Linux with no AVX-512** — `lscpu | grep -o avx512f | head -1` prints nothing (most
  consumer Ryzen and Core chips)
- **an Intel Mac** — no Mac has ever shipped with AVX-512

Apple Silicon and other ARM hosts need nothing extra: SIMD is not enabled there to begin with.

The package is imported as `ssaft`:

```python
import ssaft
```

Then register the annotation skill and the expert agent with your LLM host. Both live in the
`CeLLMan/` submodule — if you cloned without `--recurse-submodules`, run
`git submodule update --init` first.

```bash
# claude code
mkdir -p .claude/skills .claude/agents
cp -r SSAF-T/CeLLMan/skills/annotate-mode .claude/skills/
cp SSAF-T/CeLLMan/agents/annotate-expert.md .claude/agents/
```

```bash
# codex
mkdir -p .agents/skills .codex/agents
cp -r SSAF-T/CeLLMan/skills/annotate-mode .agents/skills/
cp SSAF-T/CeLLMan/agents/annotate-expert.toml .codex/agents/
printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> .codex/config.toml
```

---

## Documentation

Full documentation is in preparation and will be published on Read the Docs.

Until then, `notebooks/` holds one worked example per data modality:

| notebook | modality |
|---|---|
| [`notebooks/visiumhd_small_intestine.ipynb`](notebooks/visiumhd_small_intestine.ipynb) | array-based spatial transcriptomics |
| [`notebooks/xenium_kidney.ipynb`](notebooks/xenium_kidney.ipynb) | imaging-based spatial transcriptomics |

For scRNA-seq — where no spatial mapping is involved and CeLLMan is used on its own — see
[`CeLLMan/notebooks/TS_lung.ipynb`](CeLLMan/notebooks/TS_lung.ipynb).

For how to phrase the prompt that starts an annotation run, see
[`CeLLMan/CALLER_PROMPT_GUIDE.md`](CeLLMan/CALLER_PROMPT_GUIDE.md).

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
