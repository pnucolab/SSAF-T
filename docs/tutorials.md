# Tutorials

Two worked examples, end to end. The spatial pipeline differs by **modality**, so one example is
provided per modality:

- **[Imaging-based example](tutorials/imaging.md)** (imaging-based data — Xenium, MERSCOPE,
  CosMx), on a Xenium mouse kidney section in three dimensions.
- **[Array-based example](tutorials/array.md)** (array-based data — Visium HD, Stereo-seq), on a
  Visium HD mouse small intestine section in two dimensions.

The two examples share their central steps (cluster → CeLLMan → map) and differ at the front
(load / field / downsample) and at the end (segment / viewer). The per-parameter reference for each
stage is given under [Pipeline](pipeline/index.md); the tutorials link to it rather than reproduce
it. The annotation step is documented on its own in [Annotate](pipeline/annotate.md), which also
covers reference mode and the interactive phase.

```{note}
The figures on these pages are rendered from the notebooks held under `docs/notebooks/`; the
corresponding notebook may be opened to reproduce the example.
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/imaging
tutorials/array
```

## Datasets

| notebook | modality | platform | tissue | map |
|---|---|---|---|---|
| `docs/notebooks/tutorial_xenium_kidney_3d.ipynb` | imaging | Xenium | mouse kidney (GEO GSE269719) | 3D |
| `docs/notebooks/tutorial_visiumhd_small_intestine.ipynb` | array | Visium HD | mouse small intestine (10x Genomics) | 2D |
