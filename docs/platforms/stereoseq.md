# Stereo-seq

STOmics Stereo-seq (SAW output). Count regime; 2D. **Verified** (parameters): adult mouse brain (MOSTA).

## Input

SAW output read with `spatialdata_io.stereoseq`, passed to SSAF-T as it is:

```python
import spatialdata_io
import ssaft

sdata = spatialdata_io.stereoseq(path, dataset_id="<chip>", read_square_bin=True)

analysis = ssaft.Analysis(ssaft.Dataset("outputs/sample/ssaft_store"), ncores=30)
analysis.compute_vf(sdata, bin_um=0.5, bandwidth_um=8, sampling_distance=1.0,
                    kernel="gaussian", panel=panel)
```

`spatialdata_io.stereoseq` (0.7) expects the SAW directory layout `02.count/`, `03.register/`,
`04.tissuecut/` (`*.tissuecut.gef`), `041.cellcut/` (`*.cellbin.gef`) and `051.cellcluster/`
(`*cell.cluster.h5ad`). SAW releases that write every file into one directory (`*.tissue.gef`,
`*_regist.tif`, ...) are not read by it.

| element | key | used by |
|---|---|---|
| bin-1 counts | `tables['bin1_table']`, annotating `points['bin1_genes']` (DNB coordinates) | panel, vector field, segment (counts per cell) |
| registered ssDNA | `images['<chip>_regist']` (1 pixel = 1 DNB) | tissue mask |
| SAW cell segmentation | `shapes['cells_polygons']` | segment (seeds) |

Bins and image are in DNB units in `'global'`. SSAF-T converts them in memory, without modifying the
SpatialData: bin centres at DNB x 0.5 µm, the ssDNA at 0.5 µm per pixel (registered image and bins
share the DNB grid), tissue mask not inverted (`*_HE_regist` images are treated as H&E). The map and
the cells are written back into `'global'`, on the DNB grid.

## Stages

| stage | call |
|---|---|
| [load](../pipeline/load.md) | `spatialdata_io.stereoseq(path, dataset_id=..., read_square_bin=True)` |
| [gene panel](../pipeline/panel_vf.md) | HVGs on 8 µm pseudo-bins: `grid = ssaft.GridDataset.from_sdata(sdata)`, then `aggregate_to_grid(grid.raw, bin_um=8.0, origin=grid.origin)` |
| [vector field](../pipeline/panel_vf.md) | `compute_vf(sdata, bin_um=0.5, bandwidth_um=8, sampling_distance=1.0, kernel="gaussian", panel=panel)` |
| [downsample](../pipeline/downsample.md) | `downsample_vector(sdata, z=2, n_samples_per_block=5, block_um=10, vf_norm_percentile=25)` |
| [cluster](../pipeline/cluster.md) | PCA 50, 15 neighbours, Leiden 1.0 |
| [map](../pipeline/map.md) | `method="lda"` |
| [filter](../pipeline/filter.md) | defaults |
| [segment](../pipeline/segment.md) | `nuclei="cells_polygons"` (SAW cells' centroids), or nuclei segmented on the ssDNA image |

## Platform defaults

| parameter | value |
|---|---|
| `bin_um` | `0.5` |
| `bandwidth_um` | `8` (sparse capture needs a wide kernel) |
| `kernel` | `'gaussian'` (faster than Epanechnikov at this width) |
| HVG pseudo-bin | `8.0` µm |
| `hvg_n` | `5000` |
| `n_samples_per_block` | `5` |
