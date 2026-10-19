# Annotate (CeLLMan)

The clusters are named by [CeLLMan](https://github.com/pnucolab/CeLLMan)'s `annotate-mode`, which
runs inside an LLM host (Claude Code or Codex), not in Python. An `annotate-expert` agent loads the
Cell-Ontology module for the organ and judges every cluster: a label is accepted only when the
term's own discriminating markers are enriched, the deepest such term is chosen, and a cluster that
is not one identity is re-clustered on its ontology children's markers and judged again.

Installation of the skill and agent: [Installation, step 5](../installation.md#5-register-cellman-with-your-llm-host).

## Starting it

In the host, from the directory the skill was installed into:

```text
/annotate-mode Annotate the h5ad file below using the annotate-expert agent.

- input          : /abs/path/outputs/sample/mouse_kidney_cls.h5ad
- tissue         : mouse, kidney, normal
- cluster column : leiden
- virtual env    : /abs/path/.venv
- out            : /abs/path/outputs/sample/annotation_out
```

The full prompt reference, including several datasets at once and reference-restricted runs
(`/annotate-reference-mode`), is `CeLLMan/CALLER_PROMPT_GUIDE.md`.

## What it writes

In `out`:

| file | used by |
|---|---|
| `<input>_annotated.h5ad` with `obs['cellman_annotation']` | `map_celltypes(h5ad=...)` |
| `celltype_markers.yaml`: every emitted label and its ancestors, each with its defining / required / supporting markers | `map_celltypes(marker_genes_path=...)` |
| `expert_judgment_annotation.csv`: one row per cluster with label, markers and rationale | review |

A cluster that fits no term is `Unknown` and is kept apart per cluster on the map
(`unknown1..N`); `Background` clusters are dropped.

## Parameters

### tissue

Default
: none
Applies to
: all
Asked
: yes (from [species](load.md#species) and [organ](load.md#organ))

`species, organ[, condition]`. Free text is resolved to one ontology module; the annotator asks
before starting when the match is not exact.

### disease

Default
: none (normal tissue)
Applies to
: all
Asked
: yes

Disease context, e.g. `glioblastoma`. A hint, not a filter: disease and state terms are always
considered.

### reference

Default
: none
Applies to
: all
Asked
: yes

An expected set of cell types (a label list, or an annotated h5ad + column). Switches the run to
`/annotate-reference-mode`: identity is judged the same way, but only answer-key terms can become
labels and the rest becomes `Unknown`.

### reference_scope

Default
: `strict`
Applies to
: reference runs
Asked
: yes, when a reference is given

`strict` labels only with the answer-key's own terms; `refine` also allows their ontology
descendants.

### note

Default
: none
Applies to
: all
Asked
: no

Free-text instructions forwarded to the agent, e.g. the GPU to use.
