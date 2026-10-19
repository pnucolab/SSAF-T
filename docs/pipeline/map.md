# Map

Assigns a cell type to every voxel of the vector field by scoring it against a dictionary built from
the annotated anchors.

```python
ds = ssaft.Dataset("outputs/sample/ssaft_store")          # re-open after the annotation
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

sdata = analysis.map_celltypes(
    sdata, method="lda",
    h5ad="outputs/sample/annotation_out/mouse_kidney_cls_annotated.h5ad",
    marker_genes_path="outputs/sample/annotation_out/celltype_markers.yaml",
    kernel="gaussian", bandwidth=2.5, sampling_distance=1.0, gpu_device="cuda")
```

- **transcript regime**: the KDE is computed chunk by chunk over the whole lattice and scored on the
  fly (2D or 3D, as recorded by downsampling). `sdata` must hold the transcripts.
- **count regime**: the stored `vf_normalized` is read; no KDE, and `kernel` / `bandwidth` are
  ignored.

Written to the store: `celltype_maps` (voxel -> class id), the score volume for the method
(`max_probabilities` for `lda`/`gnb`, `max_correlations` for `corr`), `vf_norm`, the class names
and colours. Written to `sdata`: `tables['ssaft_downsampled_vectors']`, the anchors with the
rebuilt labels and palette. Rasters are added by [filter](filter.md).

This is the longest stage on large sections (hours in 3D).

## Parameters

### h5ad

Default
: none (**required** on the first mapping of a store)
Applies to
: all
Asked
: no (the annotation output)

The annotated anchors. Must be row-aligned with the anchors in the store: do not subset or reorder
the clustering h5ad before annotating it.

### marker_genes_path

Default
: `None` (whole panel)
Applies to
: all
Asked
: no (the annotation output)

CeLLMan's `celltype_markers.yaml` (preferred), `expert_judgment_annotation.csv`, or a
`{celltype: [genes]}` dict. With `method='lda'` it provides the axis that
`filter_celltypemaps(hsm_f=...)` scores; with `corr` / `gnb` it also restricts the genes used.

### method

Default
: `'lda'`
Applies to
: all
Asked
: yes
Verified
: `'lda'`

`'lda'` scores each voxel's likelihood against per-cell-type factors learned from the anchors, with
a spatial anchor prior; cell types without ontology markers still get a discriminating row.
`'gnb'` is a Gaussian naive Bayes classifier; `'corr'` takes the argmax Pearson correlation to
centroids and tends to drop rare types whose signal overlaps a larger one.

### kernel

Default
: `'gaussian'`
Applies to
: transcript
Asked
: no (**must equal** the downsampling [kernel](downsample.md#kernel))

### bandwidth

Default
: `2.5`
Applies to
: transcript
Asked
: no (**must equal** the downsampling [bandwidth](downsample.md#bandwidth))

### sampling_distance

Default
: `1.0`
Applies to
: transcript
Asked
: no (**must equal** the downsampling value)

### column

Default
: `'cellman_annotation'`
Applies to
: all
Asked
: no

Label column in the annotated h5ad.

### leiden_column

Default
: `'leiden'`
Applies to
: all
Asked
: no

Cluster column, used to split `Unknown` into `unknown1..N`.

### cmap

Default
: `None` (qualitative palette sized to the class count)
Applies to
: all
Asked
: no

Palette fixed at mapping time and reused by every later view. Change it afterwards with
`Analysis.set_celltype_palette` without re-mapping.

### map_kwargs

Default
: `None`
Applies to
: all
Asked
: no

Scorer options. `lda`: `cand_nats` (`1.0`, how far behind the best log-likelihood a class can be and
still be a candidate), `prior_weight` (`0.5`), `prior_floor` (`0.02`). `gnb`: `var_smoothing`,
`row_normalize`.

### save_vf

Default
: `False`
Applies to
: transcript
Asked
: no

Also write the transcript field to the store as `vf_normalized`. Large on disk; only when you need
the field itself.

### gpu_device / force_cpu

Default
: `'cuda'` / `False`
Applies to
: all
Asked
: gpu_device yes (one GPU choice for the run)

### gpu_target_bytes / query_subchunk_size

Default
: `1e9` / automatic
Applies to
: transcript, GPU
Asked
: no

Size of the GPU work unit. Lower `gpu_target_bytes` when the GPU runs out of memory.

### overwrite

Default
: `True`
Applies to
: all
Asked
: no

Replace an earlier map in the store.
