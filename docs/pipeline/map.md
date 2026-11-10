# Map

Each voxel of the vector field is assigned a cell type by scoring it against a dictionary built from
the annotated representative vectors. In this way the labels of the annotated vectors are extended to
every voxel of the field by a semi-supervised classifier, producing a cell-type map.

```python
ds = ssaft.Dataset("outputs/sample/ssaft_store")          # re-open after the annotation
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

sdata = analysis.map_celltypes(
    sdata, method="lda",
    h5ad="outputs/sample/annotation_out/sample_cls_annotated.h5ad",
    marker_genes_path="outputs/sample/annotation_out/celltype_markers.yaml",
    kernel="gaussian", bandwidth=2.5, sampling_distance=1.0, gpu_device="cuda")
```

- **imaging-based data**: the kernel density estimate is computed chunk by chunk over the whole
  lattice and scored as it is produced (in two or three dimensions, as recorded at downsampling);
  `sdata` must hold the transcripts.
- **array-based data**: the stored `vf_normalized` is read, so no kernel density estimate is
  computed and `kernel` and `bandwidth` are ignored.

Written to the store: `celltype_maps` (voxel to class id), the score volume for the method
(`max_probabilities` for `lda`/`gnb`, `max_correlations` for `corr`), `vf_norm`, and the class names
and colours. Written to `sdata`: `tables['ssaft_downsampled_vectors']`, the representative vectors
with their rebuilt labels and palette, which is usually left in memory because the store
holds the same vectors, names and colours. The raster images are added by [filter](filter.md).

This is the most time-consuming stage on large sections, requiring hours in three dimensions.

## Parameters

Every parameter is documented under {py:meth}`~ssaft.Analysis.map_celltypes` in the
[API reference](../api.md). Several stage-level constraints are not evident from the signature:

- `h5ad` and `marker_genes_path` are the [annotation](annotate.md) outputs; the h5ad must remain
  row-aligned with the representative vectors in the store, so the clustering h5ad must not be subset
  or reordered before annotation.
- For imaging-based data only, `kernel`, `bandwidth` and `sampling_distance` **must equal** the
  values used at [downsample](downsample.md) (kernel choices: [Kernels](kernels.md)); for array-based
  data they are ignored.
- The palette is fixed at this step and reused by every subsequent view; it may be changed afterwards
  with {py:meth}`~ssaft.Analysis.set_celltype_palette` without re-mapping.
