# Concepts

## The vector field

For every position on a regular lattice SSAF-T computes a vector with one entry per gene: the
kernel density of that gene's molecules around the position. Each vector is normalised to a fixed
L1 size (`size_after_normalization`, default 10) and `log1p`-transformed. Its norm before
normalisation, `vf_norm`, is the local mRNA density and is kept separately.

A cell type is a direction in that space, not a region on a segmentation. Nothing about the
field depends on where cell boundaries are.

Two regimes produce the field:

| regime | platforms | how the field is built | when |
|---|---|---|---|
| **transcript** | Xenium, MERSCOPE, CosMx | point KDE of individual molecules (GPU) | lazily, per chunk, inside `downsample_vector` (at anchors) and `map_celltypes` (whole lattice); discarded unless `save_vf=True` |
| **count** | Visium HD, Stereo-seq | kernel blur of the binned count raster (GPU; Gaussian by default, or tophat, Epanechnikov, exponential, linear, cosine) | once, by `compute_vf`, stored as `vf_normalized` |

The regime is detected from the store and the SpatialData; you never set it.

## Anchors

The field has hundreds of millions of voxels, too many to cluster. **Anchors** are a sample of
positions drawn inside the tissue mask (an Otsu threshold of a morphology image), at a density of
`n_samples` per `block_um` x `block_um` block. Their vectors form an AnnData (`anchors x genes`)
that is clustered like single cells.

## 2D and 3D

Transcript platforms record a `z` for every molecule and can run in 3D.

- **2D** (`z=2`): one tissue mask from one 2D image; molecules' `z` is ignored; the map is one
  plane.
- **3D** (`z=3`): every plane of a morphology **z-stack** is thresholded and sampled on its own,
  each anchor carries a 3D KDE, and the map is a volume. This needs the z-stack as a SpatialData
  image element, built with [`Analysis.zstack_image`](pipeline/zstack.md).

The dimension is fixed at downsampling and recorded in the store (`attrs['n_dims']`); mapping,
filtering and segmentation follow it. Count platforms are 2D only.

## The store

`ssaft.Dataset(path)` opens a zarr **store** that accumulates every result of one analysis:
anchors, (count regime) the field, the cell-type maps, their scores and the palette. Stages can be
run in separate processes: re-open the store and continue. One store per sample.

The SpatialData you loaded is a separate object. Results that belong next to the raw data (the
map as `labels`, the cells as `shapes` and `tables`) are written into it; see [Outputs](outputs.md).

## Annotation happens outside Python

Clustering writes an h5ad. CeLLMan, running inside an LLM host, judges each cluster against the
Cell-Ontology knowledge graph and writes back `obs['cellman_annotation']` plus
`celltype_markers.yaml`. Mapping then reads both. See [Annotation](pipeline/annotate.md).

## Tiling

Large transcript datasets can be cut into square SpatialData tiles with
`slice_spatialdata.py` and streamed by `downsample_vector(chunked=True)`. Tiling bounds memory
during downsampling only; mapping always reads the whole SpatialData, chunk by chunk. It is
optional: small sections run without it.

## Stage order

```text
                           transcript platforms          count platforms
load (spatialdata_io)            *                              *
z-stack image                    3D only                        -
tile                             optional                       -
gene panel + vector field        -                              *
downsample (anchors)             *                              *
cluster                          *                              *
annotate (CeLLMan)               *                              *
map                              *                              *
filter                           *                              *
segment                          optional (needs nuclei)        optional (needs nuclei)
```
