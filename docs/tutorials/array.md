# Array-based example — Visium HD mouse small intestine (2D)

**Array-based** platforms — Visium HD, Stereo-seq — measure counts in fixed bins rather than single
molecules, and carry no morphology z-stack, so the map is **2D**. This example applies the full
pipeline to a Visium HD section of mouse small intestine: the sample is loaded, the gene panel is
selected, the gene expression vector field is computed, the field is sampled, the sampled vectors are
clustered and annotated by CeLLMan, the labels are mapped and filtered, and single cells are
recovered from the map, seeded by nuclei from a separate segmentation.

The central stages (cluster → CeLLMan → map) are shared with the
[imaging-based example](imaging.md). The difference lies at the front: the per-bin counts are too
sparse for a kernel density estimate of single molecules, so a **gene panel** is selected first and
the field is formed by smoothing the binned counts; the final stages carry no z-stack, and
segmentation is seeded by externally segmented nuclei.

```{admonition} Dataset
:class: note
10x Genomics
[Visium HD mouse small intestine (FFPE)](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-intestine)
(CC BY 4.0): the binned outputs (all bin levels) and the full-resolution H&E `.btf`. Nuclei come
from [ENACT](https://github.com/Sanofi-Public/enact-pipeline) (StarDist on the same H&E; Kamel et
al., Bioinformatics 2025). A CUDA GPU is required. Notebook:
`docs/notebooks/tutorial_visiumhd_small_intestine.ipynb`.
```

```python
from pathlib import Path
import ssaft

DATA    = Path("data/visiumhd_small_intestine")
FULLRES = DATA / "Visium_HD_Mouse_Small_Intestine_tissue_image.btf"
ENACT   = Path("data/enact_output/mouse_small_intestine")
OUT     = Path("outputs/visiumhd_small_intestine")
ZARR    = OUT / "spatialdata.zarr"
STORE   = OUT / "ssaft_store"
CLS     = OUT / "mouse_small_intestine_cls.h5ad"
ANNOTATION = OUT / "annotation_out"

SPECIES, ORGAN = "mouse", "small_intestine"       # Cell-Ontology module
ONTO_API = "https://cellman.pnucolab.com"         # public knowledge-graph server
BIN_UM, BANDWIDTH_UM, SAMPLING_DIST, HVG_N = 2, 2.5, 2.0, 5000
```

## 1. Load Visium HD

`load_visiumhd` reads the 2 µm bins and the full-resolution H&E with `spatialdata_io.visium_hd`,
retains the in-tissue bins, converts bin coordinates to µm, and fits the bin-lattice-to-pixel affine.
The result holds the counts in `tables['table']` and the H&E in `images['staining']`.

```python
import spatialdata as sd

sdata = ssaft.load_visiumhd(DATA, species=SPECIES, organ=ORGAN,
                            bin_um=BIN_UM, fullres_image_file=str(FULLRES))
sdata.write(ZARR)
grid = ssaft.GridDataset.from_sdata(sdata)
```

```{figure} ../_static/tutorials/visiumhd_small_intestine_staining.png
:width: 100%
:alt: H&E of the Visium HD small-intestine section

The full-resolution H&E, cropped to the region mapped in this example.
```

Ref: [load](../pipeline/load.md) (Visium HD tab).

## 2. Gene panel

The field is built on highly variable genes together with every marker the knowledge graph lists for
the organ. Low-variance marker genes are retained so that they remain available to the annotation.

```python
panel, onto_markers = ssaft.select_gene_panel(
    grid.raw, SPECIES, ORGAN, hvg_n=HVG_N, server_url=ONTO_API)
```

Ref: [gene panel + field](../pipeline/panel_vf.md), {py:func}`~ssaft.select_gene_panel`.

## 3. Vector field

`compute_vf` rasterises the bin counts on a `bin_um` grid and smooths every gene with a Gaussian of
standard deviation `bandwidth_um`. Each voxel is normalised and log-transformed, then stored as
`vf_normalized` with its norm `vf_norm`. The stored grid is decimated by
`round(sampling_distance / bin_um)`, so that it may be coarser than the bins but never finer; with
2 µm bins the field remains at 2 µm, which is why `sampling_distance` is set to `bin_um` here.

```python
ds = ssaft.Dataset(str(STORE))
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)

analysis.compute_vf(sdata, bin_um=BIN_UM, panel=panel, sampling_distance=SAMPLING_DIST,
                    bandwidth_um=BANDWIDTH_UM, kernel="gaussian", device="cuda")
```

Ref: [field](../pipeline/panel_vf.md).

## 4. Downsample

`z=2`: a single tissue mask is taken from the H&E in `sdata.images`, inverted before Otsu because the
H&E shows dark tissue on a bright background. Representative positions are sampled per `block_um`
block and their vectors are looked up in the stored field; `vf_norm_percentile` discards the
positions in the lowest-density quarter.

```python
adata = analysis.downsample_vector(sdata, z=2, n_samples_per_block=5,
                                   block_um=10, vf_norm_percentile=25)
```

```{figure} ../_static/tutorials/visiumhd_small_intestine_anchors_on_staining.png
:width: 100%
:alt: Sampled representative positions overlaid on the H&E

The representative positions sampled across the tissue mask (thinned for display); each carries its
looked-up field vector.
```

Ref: [downsample](../pipeline/downsample.md) (Visium HD tab).

## 5. Cluster

```python
import scanpy as sc
import rapids_singlecell as rsc

# PCA on the CPU: the anchors x panel matrix does not fit on a 24 GB GPU next to the vector field;
# the kNN graph and Leiden run on the GPU
sc.pp.pca(adata, n_comps=50, svd_solver="randomized", random_state=0)
rsc.pp.neighbors(adata, n_neighbors=15)
rsc.tl.leiden(adata, resolution=1)
adata.write_h5ad(CLS)
```

Ref: [cluster](../pipeline/cluster.md).

## 6. Annotate with CeLLMan

As in the imaging-based example, **annotate-mode** runs in Claude Code or Codex, annotates the
clusters, and writes `mouse_small_intestine_cls_annotated.h5ad` with `obs['cellman_annotation']`
together with `celltype_markers.yaml`.

```text
/annotate-mode Annotate the h5ad below using the annotate-expert agent.

- input          : /abs/outputs/visiumhd_small_intestine/mouse_small_intestine_cls.h5ad
- tissue         : mouse, small intestine
- cluster column : leiden
- virtual env    : /abs/path/to/env-with-ssaft
- out            : /abs/outputs/visiumhd_small_intestine/annotation_out
```

```{figure} ../_static/tutorials/visiumhd_small_intestine_anchors_celltype.png
:width: 100%
:alt: Representative positions coloured by CeLLMan cell type

The representative positions coloured by their CeLLMan cell type. The knowledge graph behind these
assignments is presented in the [CeLLMan KG Explorer](https://cellman.pnucolab.com).
```

Ref: [annotate](../pipeline/annotate.md).

## 7. Map and filter

The store holds `vf_normalized`, so `map_celltypes` reads the field rather than computing a kernel
density estimate. It scores every voxel against the annotated representative vectors (`method='lda'`),
using each cell type's knowledge-graph markers from `celltype_markers.yaml`. `filter_celltypemaps`,
under its defaults, then cuts each cell type at 0.3 of its own density peak.

```python
sdata = analysis.map_celltypes(sdata, method="lda", kernel="gaussian",
                               h5ad=str(ANNOTATION / "mouse_small_intestine_cls_annotated.h5ad"),
                               marker_genes_path=str(ANNOTATION / "celltype_markers.yaml"),
                               gpu_device="cuda")
sdata = analysis.filter_celltypemaps(sdata)
```

The shared palette is applied with {py:meth}`~ssaft.Analysis.set_celltype_palette` before any figure
is drawn, so that the RGB map, the cells table and every plot are consistent.

```{figure} ../_static/tutorials/visiumhd_small_intestine_celltype_map.png
:width: 100%
:alt: Filtered cell-type map of the small-intestine section

The filtered 2D cell-type map. The villus architecture of the small intestine is resolved without
segmenting any cell first.
```

Refs: [map](../pipeline/map.md), [filter](../pipeline/filter.md).

## 8. Nuclei and segmentation

`segment_cells` requires nuclei as watershed seeds. Here they are taken from ENACT's StarDist
segmentation of the H&E (`nuclei_df.csv`, one WKT polygon per nucleus). ENACT crops the image to the
bounding box of the in-tissue bins before segmenting, so its coordinates are expressed in that crop's
pixel frame; adding the crop origin returns them to the full-resolution image, where they adopt the
H&E's transformation and land in the map's µm frame. Without the origin shift the nuclei fall about
40 µm from their H&E position.

```python
import pandas as pd, geopandas as gpd
from shapely import from_wkt
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation

tp = pd.read_parquet(DATA / "binned_outputs/square_002um/spatial/tissue_positions.parquet")
tp = tp[tp.in_tissue == 1].astype({"pxl_col_in_fullres": int, "pxl_row_in_fullres": int})
tp = tp[(tp.pxl_col_in_fullres >= 0) & (tp.pxl_row_in_fullres >= 0)]
x_min, y_min = int(tp.pxl_col_in_fullres.min()), int(tp.pxl_row_in_fullres.min())

df = pd.read_csv(ENACT / "nuclei_df.csv", usecols=["geometry", "id"])
nuclei = gpd.GeoDataFrame(
    geometry=gpd.GeoSeries(from_wkt(df["geometry"].to_numpy())).translate(x_min, y_min).to_numpy(),
    index=pd.Index(df["id"].to_numpy(), name="nucleus_id"))
sdata.shapes["enact_nucleus_boundaries"] = ShapesModel.parse(
    nuclei, transformations=dict(get_transformation(sdata.images["staining"], get_all=True)))

sdata = analysis.segment_cells(sdata, nuclei="enact_nucleus_boundaries")
```

```{figure} ../_static/tutorials/visiumhd_small_intestine_cells.png
:width: 100%
:alt: Segmented single cells

The recovered cells, coloured by cell type. Each cell is one nucleus flooded within its cell type's
territory of the map.
```

Ref: [segment](../pipeline/segment.md) (array platforms seed segmentation from external nuclei).

## 9. Explore in the viewer

The 2D map and the recovered cells are presented in the
[SSAF-T viewer](https://ssaf.pnucolab.com/T/explore/visiumhd-mouse-small-intestine-ffpe/); the
downloadable per-dataset store is the same SpatialData zarr this notebook writes.

Return to the [tutorials index](../tutorials.md).
