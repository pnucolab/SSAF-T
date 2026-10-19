# Pipeline stages

Each page documents one stage: what it does, its inputs and outputs, and every parameter under its
own heading so it can be linked directly.

Every parameter section starts with the same four fields:

Default
: the value the function uses when you pass nothing.

Applies to
: the regime (transcript / count), dimension (2D / 3D) or platforms the parameter is read in.
  Elsewhere it is ignored or rejected.

Asked
: whether the [agent skill](../agent.md) asks for it before a run (**yes**), or leaves it at the
  platform default unless you override it (**no**).

Verified
: the values the tutorials and benchmark datasets were run with, where they differ from the default.

```{toctree}
:maxdepth: 1

load
zstack
tile
panel_vf
downsample
cluster
annotate
map
filter
segment
```

## Stages per platform

| stage | Xenium | MERSCOPE | CosMx | Visium HD | Stereo-seq |
|---|---|---|---|---|---|
| [load](load.md) | yes | yes | yes | yes | yes |
| [z-stack](zstack.md) | 3D only | 3D only | 3D only | - | - |
| [tile](tile.md) | optional | optional | optional | - | - |
| [gene panel + field](panel_vf.md) | - | - | - | yes | yes |
| [downsample](downsample.md) | yes | yes | yes | yes | yes |
| [cluster](cluster.md) | yes | yes | yes | yes | yes |
| [annotate](annotate.md) | yes | yes | yes | yes | yes |
| [map](map.md) | yes | yes | yes | yes | yes |
| [filter](filter.md) | yes | yes | yes | yes | yes |
| [segment](segment.md) | optional | optional | optional | optional | optional |

## Resuming

Every stage persists its result, so a run can stop between any two stages and continue in a new
process:

| after | what is on disk | resume with |
|---|---|---|
| load | `spatialdata.zarr` | `sd.read_zarr(...)` |
| tile | `<crops>/r00_c00.zarr ...` | pass the directory as `sdata_dir` |
| field | `vf_normalized`, `vf_norm` in the store | `ssaft.Dataset(store)` |
| downsample | anchors in the store | `Dataset(store)`; the AnnData can be rebuilt from the store |
| cluster | `<name>_cls.h5ad` | the annotation prompt |
| annotate | `annotation_out/<name>_cls_annotated.h5ad`, `celltype_markers.yaml` | `map_celltypes(h5ad=..., marker_genes_path=...)` |
| map | `celltype_maps` + score volume in the store | `filter_celltypemaps` (seconds; re-run freely) |
| filter | `filtered_celltype_maps` in the store; `labels` / `images` in `sdata` | `segment_cells` |
