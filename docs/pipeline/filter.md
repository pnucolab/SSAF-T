# Filter

Voxels at which there is no cell are removed, and the map is written into the SpatialData. The step
completes in seconds, so several settings may be evaluated against one mapping.

```python
sdata = analysis.filter_celltypemaps(sdata)                 # defaults
sdata = analysis.filter_celltypemaps(sdata, norm_f=0.5)     # stricter density cut
```

Written to the store: `filtered_celltype_maps` (`-1` = background). Written to `sdata`:

- `labels['ssaft_celltype_map']`: pixel value = class id + 1 (0 = background); a three-dimensional
  map retains every layer as `(z, y, x)`;
- `images['ssaft_celltype_map_rgb']`: an RGB rendering (the middle layer in three dimensions).

The class names for the pixel values are
`sdata['ssaft_downsampled_vectors'].obs['cellman_annotation'].cat.categories[pixel - 1]`.

Two independent questions arise, with one parameter for each:

| question | parameter |
|---|---|
| Is there a cell at this voxel? | `norm_f` (per-cell-type density cut), optionally `min_norm` |
| Is the label right? | `min_p` (lda/gnb) or `min_r` (corr) |

## Parameters

Every parameter is documented under {py:meth}`~ssaft.Analysis.filter_celltypemaps` in the
[API reference](../api.md). In practice, `norm_f` (the per-cell-type density cut) is the principal
control; `norm_depth=True` should be retained in three dimensions so that the outer planes are not
emptied wholesale, and `show_unknown=True` retains the annotation's `unknown1..N` classes on the map.
