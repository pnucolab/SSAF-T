# Quick start (tldr)

The complete SSAF-T pipeline on a single page, with a Xenium dataset as the example. Each step links
to its full reference. Installation precedes a first run ([Installation](installation.md)); every
result persists to one store, so a run may be interrupted and resumed between any two steps.

```{note}
Illustrative end-to-end listing; the worked [notebooks](tutorials.md) are being rewritten. In
ordinary use only the paths and the tissue string are changed.
```

## 0. Load + vector field

The platform output is read once and written to a SpatialData zarr, and the gene expression vector
field is then built into a SSAF-T store. See [load](pipeline/load.md) and
[field](pipeline/panel_vf.md).

```python
import spatialdata as sd, spatialdata_io, ssaft

sdata = spatialdata_io.xenium("path/to/xenium_output")
sdata.write("outputs/sample/spatialdata.zarr")
sdata = sd.read_zarr("outputs/sample/spatialdata.zarr")

ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
analysis.compute_vf(sdata, panel=None, bandwidth_um=8.0, kernel="gaussian", device="cuda")
```

## 1. Downsample

The vector field is sampled at representative positions within the tissue mask. See
[downsample](pipeline/downsample.md).

```python
adata = analysis.downsample_vector(sdata, z=2, n_samples=5, block_um=10,
                                   bandwidth=2.5, sampling_distance=1.0,
                                   kernel="gaussian", gpu_device="cuda")
```

## 2. Cluster

The sampled vectors are clustered with Leiden. See [cluster](pipeline/cluster.md).

```python
import rapids_singlecell as rsc

rsc.pp.pca(adata, n_comps=50)
rsc.pp.neighbors(adata, n_neighbors=15)
rsc.tl.leiden(adata, resolution=1.0)
adata.write_h5ad("outputs/sample/sample_cls.h5ad")
```

## 3. CeLLMan (annotate)

Each cluster is annotated by CeLLMan, run from an agent session; the knowledge graph is browsed in
the [CeLLMan KG Explorer](https://cellman.pnucolab.com). See [annotate](pipeline/annotate.md).

```text
/annotate-mode Annotate the h5ad file below using the annotate-expert agent.

- input          : /abs/outputs/sample/sample_cls.h5ad
- tissue         : <species>, <organ>, normal
- cluster column : leiden
- out            : /abs/outputs/sample/annotation_out
```

This writes `annotation_out/sample_cls_annotated.h5ad` and `celltype_markers.yaml`.

## 4. Map

The labels are extended to every voxel of the vector field, producing a cell-type map. See
[map](pipeline/map.md).

```python
ds = ssaft.Dataset("outputs/sample/ssaft_store")           # re-open after annotation
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

sdata = analysis.map_celltypes(
    sdata, method="lda",
    h5ad="outputs/sample/annotation_out/sample_cls_annotated.h5ad",
    marker_genes_path="outputs/sample/annotation_out/celltype_markers.yaml",
    kernel="gaussian", bandwidth=2.5, sampling_distance=1.0, gpu_device="cuda")

sdata = analysis.filter_celltypemaps(sdata)                # clean the map
```

## 5. Segment

Single cells are recovered last, constrained by the cell-type map. See
[segment](pipeline/segment.md).

```python
sdata = analysis.segment_cells(sdata, nuclei="nucleus_boundaries")
cells = sdata.tables["ssaft_cells"]
```

## 6. 3D viewer

The result is explored in the browser with the [SSAF-T viewer](https://ssaf.pnucolab.com/T).

---

See also the step-by-step [spatial examples](tutorials.md), and the annotation step in
[Annotate](pipeline/annotate.md), which covers reference mode and the interactive phase.
