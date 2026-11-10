# Pipeline stages

SSAF-T proceeds through a sequence of stages, each documented on its own page: the operation it
performs, its inputs and outputs, the platform-specific calls, and the considerations that are not
evident from a function signature. The **parameter reference is given in the
[API reference](../api.md)**, where every `ssaft` function's parameters are taken from its docstring;
the stage pages refer to it rather than reproduce it.

Two stages are driven by something other than an `ssaft` function and retain their own parameter
notes: [tile](tile.md) (the `slice_spatialdata.py` command-line interface) and
[annotate](annotate.md) (the CeLLMan skill prompt fields). These notes record whether each field is
requested interactively (**Asked**) and the values used in the tutorials (**Verified**). [Load](load.md) and [cluster](cluster.md) rely on external readers and clustering
tools; they present the calls in context and leave those tools' parameters to their respective
documentation.

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
kernels
```

The final page, [kernels](kernels.md), describes a concept common to the field stages rather than a
stage of its own.

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
| [segment](segment.md) | yes | yes | yes | yes | yes |

## Resuming

Every stage persists its result, so a run may be interrupted between any two stages and continued in
a new process:

| after | what is on disk | resume with |
|---|---|---|
| load | `ssaft_sdata.zarr` | `sd.read_zarr(...)` |
| tile | `<crops>/r00_c00.zarr ...` | pass the directory as `sdata_dir` |
| field | `vf_normalized`, `vf_norm` in the store | `ssaft.Dataset(store)` |
| downsample | anchors in the store | `Dataset(store)`; the AnnData can be rebuilt from the store |
| cluster | `<name>_cls.h5ad` | the annotation prompt |
| annotate | `annotation_out/<name>_cls_annotated.h5ad`, `celltype_markers.yaml` | `map_celltypes(h5ad=..., marker_genes_path=...)` |
| map | `celltype_maps` + score volume in the store | `filter_celltypemaps` (seconds; re-run freely) |
| filter | `filtered_celltype_maps` in the store; `labels` / `images` in `sdata` | `segment_cells` |
