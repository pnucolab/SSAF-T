# Annotate (CeLLMan)

The clusters are annotated by [CeLLMan](https://github.com/pnucolab/CeLLMan)'s `annotate-mode`,
which runs inside a language-model host (Claude Code or Codex) rather than in Python. An
`annotate-expert` agent loads the knowledge-graph module for the organ and judges every cluster: a
label is assigned only when the term's own discriminating markers are elevated above the dataset as
a whole, the deepest such term is chosen, and a cluster that is not a single identity is re-clustered
on the markers of its ontology children and judged again.

The skill and agent are installed as described in
[Installation, step 5](../installation.md#5-register-cellman-with-the-language-model-host).

## Starting it

Within the host, from the directory into which the skill was installed:

```text
/annotate-mode Annotate the h5ad file below using the annotate-expert agent.

- input          : /abs/path/outputs/sample/mouse_kidney_cls.h5ad
- tissue         : mouse, kidney, normal
- cluster column : leiden
- virtual env    : /abs/path/.venv
- out            : /abs/path/outputs/sample/annotation_out
```

The complete prompt reference, covering multiple datasets in one run and reference-restricted runs
(`/annotate-reference-mode`), is given in `CeLLMan/CALLER_PROMPT_GUIDE.md`.

## What it writes

In `out`:

| file | used by |
|---|---|
| `<input>_annotated.h5ad` with `obs['cellman_annotation']` | `map_celltypes(h5ad=...)` |
| `celltype_markers.yaml`: every emitted label and its ancestors, each with its defining / required / supporting markers | `map_celltypes(marker_genes_path=...)` |
| `expert_judgment_annotation.csv`: one row per cluster with label, markers and rationale | review |

A cluster that matches no term is labelled `Unknown` and retained separately per cluster on the map
(`unknown1..N`); `Background` clusters are discarded.

## Reference mode

When an expected set of cell types is already available — a published annotation, a label list, or a
column in a reference `h5ad` — the run is started with `/annotate-reference-mode` in place of
`/annotate-mode`. Identity is still judged by the same test against the full ontology, but the label
space is restricted to the answer key, and any cluster the answer key cannot explain is labelled
`Unknown`. Only the reference's labels are read; its expression never serves as evidence for the
clusters under annotation.

```text
/annotate-reference-mode Annotate the h5ad file below using the annotate-expert agent.

- input            : /abs/path/outputs/sample/mouse_kidney_cls.h5ad
- tissue           : mouse, kidney, normal
- cluster column   : leiden
- reference h5ad   : /abs/path/reference.h5ad
- reference column : cell_type
- reference scope  : strict
- virtual env      : /abs/path/.venv
- out              : /abs/path/outputs/sample/annotation_out
```

`reference scope` sets the depth of the label space. Under `strict`, the default, a cluster is
labelled only with the answer key's own terms. Under `refine`, the ontology descendants of those
terms also become labelable, so that a lumped reference term such as *lymphocyte* is resolved into
its T, B and NK children where the data support it. A reference run additionally writes
`reference_coverage.json`, recording how much of the answer key was reproduced and what was left
unexplained.

## The interactive phase

The autonomous run constitutes the complete annotation, but the host session does not end there.
After finalize, the host presents the label table with an explanation and offers either to conclude
or to continue under the user's direction. This is the single point at which the result is directed,
and every request is answered by the same method as the autonomous run — a targeted re-clustering
followed by re-judgment, not a manual edit of labels. The test is unchanged, so a cell type that is
asked after is reported absent when its markers are not elevated above background, rather than forced
onto a cluster. Representative directions and how they are answered:

- whether a given type is truly absent — the leaf carrying that type's signal is re-clustered against
  it and its nearest sibling, and either a real sub-leaf is labelled or the type is reported
  unenriched;
- resolving one population into its subtypes — those paths are subclustered against the subtype
  candidates, the discriminative sub-leaves taking the finer label and the remainder retaining the
  parent;
- a cell state — the run's state evidence is read and the leaves where a program is genuinely
  enriched are tagged, for example `CD8 T cell, proliferating`. States are applied only here.

A refinement made in the interactive phase is written to `obs['cellman_free_annotation']`, leaving
the autonomous `obs['cellman_annotation']` intact, so that the directed result is kept alongside the
reproducible one rather than over it.
