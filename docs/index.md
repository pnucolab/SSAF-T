# SSAF-T

**A spatial segmentation-free agentic framework for transcriptomics**

SSAF-T infers cell types from spatial transcriptomics data without using cell boundaries to decide
identity. Expression is estimated as a continuous gene expression vector field — computed from
transcript coordinates for imaging-based data and from binned counts for array-based data — and cell
identity is settled on that field by [CeLLMan](https://github.com/pnucolab/CeLLMan), a large language
model agent that queries a curated, literature-grounded knowledge graph in place of a reference
dataset. The vector field is sampled at representative positions within a tissue mask, the sampled
vectors are clustered and annotated, and the labels are extended to every voxel of the field by a
semi-supervised classifier, producing a cell-type map. Because identity is settled on the field,
cells are not required at any point: where single cells are wanted they are recovered from the
finished map, so that segmentation becomes the last step rather than the first.

```{image} ../SSAF-T.png
:alt: SSAF-T overview
:width: 100%
```

## Where to start

- Installation and a first run: [Installation](installation.md), followed by the
  [Quick start](tldr.md).
- Processing a dataset: the [pipeline stages](pipeline/index.md) document each step; the platform is
  selected in each stage's tabs, and every function's parameters are given in the
  [API reference](api.md).
- Browsing without a local installation: the [SSAF-T viewer](https://ssaf.pnucolab.com/T) presents
  the cell-type maps and the [CeLLMan KG Explorer](https://cellman.pnucolab.com) the knowledge graph;
  each portal documents its own use on its About page.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
tldr
tutorials
```

```{toctree}
:maxdepth: 2
:caption: Pipeline

pipeline/index
```

```{toctree}
:maxdepth: 1
:caption: API

api
```

```{toctree}
:maxdepth: 1
:caption: Reference

outputs
troubleshooting
```
