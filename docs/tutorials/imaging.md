# Imaging-based example — Xenium mouse kidney (3D)

**Imaging-based** platforms — Xenium, MERSCOPE, CosMx — record single molecules and carry a staining
image, so the section can be reconstructed in **3D**. This example applies the full pipeline in
three dimensions to one Xenium section of a healthy mouse kidney: the sample is loaded and the
z-stack is built, the section is tiled, the vector field is sampled in 3D, the sampled vectors are
clustered and annotated by CeLLMan, the labels are mapped and filtered, and single cells are
recovered from the map.

The central stages (cluster → CeLLMan → map) are shared with the [array-based example](array.md);
the preceding stages (load / z-stack / downsample) and the final one (segment) are specific to
imaging-based data. Each parameter below is documented per platform under
[Pipeline](../pipeline/index.md), to which this page refers rather than reproduce it.

```{admonition} Dataset
:class: note
Sham-operated right kidney (male) of GEO series
[GSE269719](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269719), sample
[GSM8325616](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8325616), from Xuanyuan, Wu,
Sundaramoorthi et al., *Multimodal spatial transcriptomic characterization of mouse kidney injury
and repair*, Nat Commun 16, 7567 (2025). A CUDA GPU is required for the kernel density estimation,
the mapping and `rapids_singlecell`. Notebook: `docs/notebooks/tutorial_xenium_kidney_3d.ipynb`.
```

```python
from pathlib import Path
import ssaft

DATA = Path("data/xenium_kidney/ShamR-output-XETG00063__0005295__Region_6__20230717__191520")
OUT  = Path("outputs/xenium_kidney")
ZARR  = OUT / "spatialdata.zarr"      # Xenium outputs + the morphology z-stack
CROPS = OUT / "spatialdata_crops"     # tiles for chunked downsampling
STORE = OUT / "ssaft_store"           # SSAF-T store: anchors, vector-field norms, cell-type maps
CLS   = OUT / "mouse_kidney_cls.h5ad" # clustered anchors, the input of annotate-mode
ANNOTATION = OUT / "annotation_out"   # written by annotate-mode
```

## 1. Load and add the morphology z-stack

`spatialdata_io.xenium` loads the transcripts (with their `z`), the nucleus boundaries and the two
**2D** morphology images. A 2D image carries no depth and therefore cannot locate the tissue along z.
The full stack, `morphology.ome.tif`, comprises 11 DAPI planes 3 µm apart.
[`zstack_image`](../pipeline/zstack.md) wraps that file as a 3D image element `(c, z, y, x)` without
reading it into memory; `like=` places it on the pixel grid of `morphology_focus` so that it
overlays the transcripts exactly. It must be written with the rest of the SpatialData, because
[`downsample_vector`](../pipeline/downsample.md) reads its planes from `sdata.images`.

```python
import spatialdata as sd
import spatialdata_io

sdata = spatialdata_io.xenium(DATA)
ssaft.Analysis.zstack_image(   # -> sdata.images["morphology_zstack"]
    DATA / "morphology.ome.tif", sdata=sdata, write=False,
    like=sdata.images["morphology_focus"])
sdata.write(ZARR)
sdata = sd.read_zarr(ZARR)
```

```{figure} ../_static/tutorials/xenium_kidney_staining.png
:width: 100%
:alt: DAPI morphology of the Xenium kidney section

The DAPI morphology, cropped to the region mapped in this example. The 6 µm FFPE section lies in
the middle planes of the stack, where the transcripts are; planes outside the transcripts' z range
are skipped during downsampling, so that no representative position is placed where there are no
molecules.
```

Refs: [load](../pipeline/load.md) (Xenium tab), [z-stack](../pipeline/zstack.md).

## 2. Tile the section

`slice_spatialdata.py` divides the SpatialData into square tiles (in `global` units, Xenium pixels
here) so that `downsample_vector(chunked=True)` can stream them. `--image-key` must retain the
z-stack; the transcripts retain their `z` column.

```python
import subprocess, sys

SLICE = Path(ssaft.__file__).parent / "scripts" / "slice_spatialdata.py"
subprocess.run([sys.executable, str(SLICE),
                "--sdata-path", str(ZARR), "--tile-size", "10000",
                "--output-dir", str(CROPS),
                "--image-key", "morphology_focus", "morphology_zstack",
                "--keep-points", "transcripts",
                "--drop-labels", "--drop-shapes", "--drop-tables"], check=True)
```

Ref: [tile](../pipeline/tile.md).

## 3. Downsample in 3D

With `z=3` every plane of `morphology_zstack` is treated independently: the plane's tissue mask is
its Otsu threshold scaled by `otsu_mult`; `n_samples` representative positions are drawn per
`block_um` × `block_um` block on that plane's z; and each representative vector is the **3D** kernel
density estimate of the transcripts at that position (`bandwidth` µm, Gaussian). Positions whose
kernel reaches almost no transcripts are discarded. DAPI presents bright nuclei on a dark background,
hence `revert_image=False`. The `kernel` must match the one supplied to
[`map_celltypes`](../pipeline/map.md) below.

```python
ds = ssaft.Dataset(str(STORE))
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

adata = analysis.downsample_vector(
    z=3, image_key="morphology_zstack",
    chunked=True, sdata_dir=str(CROPS),
    n_samples=10, block_um=10, otsu_mult=0.6, revert_image=False,
    bandwidth=2.5, sampling_distance=1.0, kernel="gaussian",
    gpu_device="cuda")
```

```{figure} ../_static/tutorials/xenium_kidney_anchors_on_staining.png
:width: 100%
:alt: Sampled representative positions overlaid on the morphology

The representative positions sampled across the section (thinned for display), each carrying a 3D
kernel density estimate of the local transcripts. They cover the tissue mask on every plane that
falls within the transcripts' z range.
```

Ref: [downsample](../pipeline/downsample.md) (Xenium tab).

## 4. Cluster

The representative vectors are clustered as single cells would be. `rapids_singlecell` runs PCA, the
kNN graph and Leiden on the GPU.

```python
import rapids_singlecell as rsc

rsc.pp.pca(adata, n_comps=50)
rsc.pp.neighbors(adata, n_neighbors=15)
rsc.tl.leiden(adata, resolution=1)
adata.write_h5ad(CLS)
```

Ref: [cluster](../pipeline/cluster.md).

## 5. Annotate with CeLLMan

The clusters are annotated by CeLLMan's **annotate-mode** skill, which runs in Claude Code or Codex
rather than in the notebook kernel. It is invoked with a prompt of the form below, with the absolute
paths filled in. It writes `mouse_kidney_cls_annotated.h5ad` with `obs['cellman_annotation']`, the
rationale for every cluster, and `celltype_markers.yaml` — the knowledge-graph markers behind each
label, which the mapping uses next.

```text
/annotate-mode Annotate the h5ad file below using the annotate-expert agent.

- input          : /abs/outputs/xenium_kidney/mouse_kidney_cls.h5ad
- tissue         : mouse, kidney, normal
- cluster column : leiden
- virtual env    : /abs/path/to/env-with-ssaft
- out            : /abs/outputs/xenium_kidney/annotation_out
```

```{figure} ../_static/tutorials/xenium_kidney_anchors_celltype.png
:width: 100%
:alt: Representative positions coloured by CeLLMan cell type

The same representative positions, coloured by the cell type CeLLMan assigned to their cluster. The
knowledge graph behind these assignments — terms, markers and PubMed evidence — is presented in the
[CeLLMan KG Explorer](https://cellman.pnucolab.com).
```

Ref: [annotate](../pipeline/annotate.md).

## 6. Map cell types in 3D

The store recorded `n_dims = 3` at downsampling, so `map_celltypes` computes the vector field on a
3D lattice and assigns a cell type to every voxel. `method='lda'` scores voxels against a dictionary
built from the annotated representative vectors; `marker_genes_path` supplies each cell type's own
knowledge-graph markers.

```python
ds = ssaft.Dataset(str(STORE))               # re-open: annotation ran outside the kernel
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

sdata = analysis.map_celltypes(
    sdata, method="lda", kernel="gaussian", bandwidth=2.5, sampling_distance=1.0,
    h5ad=str(ANNOTATION / "mouse_kidney_cls_annotated.h5ad"),
    marker_genes_path=str(ANNOTATION / "celltype_markers.yaml"),
    gpu_device="cuda")
```

Ref: [map](../pipeline/map.md).

## 7. Filter

`filter_celltypemaps` removes voxels that contain no cell. Under the defaults, each cell type is cut
at `norm_f = 0.3` of its own density peak, after each plane's density is normalised by that plane's
median (`norm_depth=True`), because density declines away from the middle of the section. The
filtered map is written back to the store and into `sdata` as `labels['ssaft_celltype_map']` (every
layer, `(z, y, x)`) and `images['ssaft_celltype_map_rgb']` (an RGB rendering of the middle layer).

```python
sdata = analysis.filter_celltypemaps(sdata)
```

Colour is a presentation choice, which SSAF-T stores once so that every view is consistent. The
palette used in the paper was matched by cell-type name against the reference annotation, so that the
SSAF-T and published panels share one legend. It is applied with
{py:meth}`~ssaft.Analysis.set_celltype_palette` before any figure is drawn.

```{figure} ../_static/tutorials/xenium_kidney_celltype_map.png
:width: 100%
:alt: Filtered cell-type map of the kidney section

The filtered cell-type map (middle plane). Every retained voxel carries a CeLLMan label; the
background is where no cell type cleared its density floor.
```

Refs: [filter](../pipeline/filter.md), [map](../pipeline/map.md),
{py:meth}`~ssaft.Analysis.set_celltype_palette`.

## 8. Segment into single cells

`segment_cells` recovers single cells from the map: one watershed per cell type, seeded by nuclei and
confined to that cell type's territory. The store is 3D, so the watershed runs over every layer of
the filtered map. Each nucleus centroid serves as one seed on the map's middle layer and takes the
map's label there; the flood proceeds per cell type, in 3D, within that type's voxels. It writes the
3D cell labels, the per-layer and middle-layer boundaries, and `tables['ssaft_cells']` (cell × gene
counts, with each cell's volume and layer count).

```python
sdata = analysis.segment_cells(sdata, nuclei="nucleus_boundaries")
```

```{figure} ../_static/tutorials/xenium_kidney_cells.png
:width: 100%
:alt: Segmented single cells on the middle layer

The recovered cells on the map's middle layer, coloured by cell type. Cells of different types divide
where the map changes type; cells of the same type divide midway between their nuclei.
```

Ref: [segment](../pipeline/segment.md).

## 9. Explore in the viewer

The 3D map and the recovered cells are presented in the browser by the
[SSAF-T viewer](https://ssaf.pnucolab.com/T/explore/xenium-mouse-kidney-sham-region6/). The
downloadable per-dataset store is the same SpatialData zarr this notebook writes.

Return to the [tutorials index](../tutorials.md).
