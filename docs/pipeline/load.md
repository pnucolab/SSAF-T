# Load

SSAF-T does not read vendor output directly. The raw data are loaded into a
[SpatialData](https://spatialdata.scverse.org) object with
[`spatialdata_io`](https://spatialdata.scverse.org/projects/io), written to zarr once, and read
back at every later stage.

```python
import spatialdata as sd
import spatialdata_io

sdata = spatialdata_io.xenium("path/to/xenium_output")   # one reader per platform, below
sdata.write("outputs/sample/spatialdata.zarr")
sdata = sd.read_zarr("outputs/sample/spatialdata.zarr")   # always continue from the zarr
```

Reading back from zarr is necessary because the vendor readers return lazily-parsed elements whose
later access can be considerably slower than the zarr copy.

## Per-platform loading

Select the platform below; the selection is retained across the pipeline pages (downsample, segment,
and the rest).

::::{tab-set}
:class: platform-tabs

:::{tab-item} Xenium
:sync: xenium

10x Genomics Xenium, imaging-based; two or three dimensions. **Verified:** mouse kidney, 3D.

```python
sdata = spatialdata_io.xenium("path/to/xenium_output")
```

Input: the Xenium Onboard Analysis output directory (`experiment.xenium`, `transcripts.parquet`,
`morphology_focus/` or `morphology_focus.ome.tif`, and, for 3D, `morphology.ome.tif`).

| element | key | used by |
|---|---|---|
| transcripts (`x`, `y`, `z` µm, `feature_name`) | `points['transcripts']` | downsample, map, segment |
| 2D morphology (DAPI + stains) | `images['morphology_focus']` | 2D mask; `like=` for the z-stack |
| nuclei | `shapes['nucleus_boundaries']` | segment |
:::

:::{tab-item} MERSCOPE
:sync: merscope

Vizgen MERSCOPE / MERFISH, imaging-based; two or three dimensions. **Verified:** mouse liver, 3D.

```python
sdata = spatialdata_io.merscope(path, z_layers=3)   # z_layers selects the 2D plane(s) loaded
```

Input: the region directory with `detected_transcripts.csv` and `images/mosaic_<stain>_z<N>.tif`.

| element | key | used by |
|---|---|---|
| transcripts (`x`, `y` µm, `z` = plane index, `gene`) | `points['<dataset_id>_transcripts']` | downsample, map, segment |
| mosaic images (one per z layer, stains as channels) | `images['<dataset_id>_z3']` | 2D mask; `like=` for the z-stack |

The stain determines the mask: DAPI yields a nuclear mask and polyT a fuller tissue mask. The vendor
`cell_boundaries.parquet` delineate whole cells rather than nuclei; DAPI is therefore segmented to
obtain nuclei for the segmentation seeds.
:::

:::{tab-item} CosMx
:sync: cosmx

NanoString CosMx SMI, imaging-based; two or three dimensions. **Each field of view (FOV) is
processed as a separate run.**

```python
sdata = spatialdata_io.cosmx(path, dataset_id="<dataset_id>", transcripts=True)
```

Input: the flat-file export (`*_tx_file.csv`, `*_fov_positions_file.csv`, `*_exprMat_file.csv`,
`*_metadata_file.csv`, `CellComposite/`, `CellLabels/`). The reader returns **one element per FOV**
(`points['<fov>_points']` in pixels with `z_raw`, `images['<fov>_image']`, `labels['<fov>_labels']`).

Each FOV constitutes a separate SSAF-T run. The SpatialData for each FOV is built with transcripts in
µm, together with a transform back to pixels, as Xenium stores them:

```python
import spatialdata as sd
from spatialdata.models import PointsModel, Image2DModel
from spatialdata.transformations import Identity, Scale

PX_UM = 0.12028          # µm/pixel; 0.18 on older instruments — check the export metadata
fov = "12"
df = sdata.points[f"{fov}_points"].compute().reset_index(drop=True)
df["x"] *= PX_UM; df["y"] *= PX_UM
coords = {"x": "x", "y": "y"}   # 3D: df["z"] = df["z_raw"] * Z_STEP_UM; coords["z"] = "z"
fov_sdata = sd.SpatialData(
    points={"transcripts": PointsModel.parse(df, coordinates=coords, feature_key="target",
        transformations={"global": Scale([1/PX_UM, 1/PX_UM], axes=("x", "y"))})},
    images={"composite": Image2DModel.parse(sdata.images[f"{fov}_image"].data, dims=("c","y","x"),
        transformations={"global": Identity()})})
fov_sdata.write(f"outputs/sample/fov{fov}/spatialdata.zarr")
```
:::

:::{tab-item} Visium HD
:sync: visiumhd

10x Genomics Visium HD, array-based; two dimensions. **Verified:** mouse small intestine.

```python
sdata = spatialdata_io.visium_hd(path, bin_size=2, load_all_images=True,
                                 fullres_image_file="path/to/tissue_image.btf")
```

Input: the Space Ranger `outs/` with `binned_outputs/square_002um/`, plus (recommended) the
full-resolution microscope image (`.btf` / `.tif`).

| element | key | used by |
|---|---|---|
| 2 µm bin counts (`array_row`/`array_col`/`in_tissue`, `obsm['spatial']`) | `tables['<dataset_id>_square_002um']` | panel, field, segment |
| full-resolution H&E | `images['<dataset_id>_full_image']` | tissue mask |

The bins are located at H&E pixels in a coordinate system named `<dataset_id>`; SSAF-T places the
in-tissue bins at `(array_col, array_row) x 2` µm and fits the pixel -> µm affine in memory (see the
array-platform note below).
:::

:::{tab-item} Stereo-seq
:sync: stereoseq

STOmics Stereo-seq (SAW output), array-based; two dimensions. **Verified** (parameters): adult mouse brain (MOSTA).

```python
sdata = spatialdata_io.stereoseq(path, dataset_id="<chip>", read_square_bin=True)
```

Input: the SAW layout `02.count/`, `03.register/`, `04.tissuecut/` (`*.tissuecut.gef`),
`041.cellcut/` (`*.cellbin.gef`), `051.cellcluster/` (`*cell.cluster.h5ad`). SAW releases that write
everything into one directory (`*.tissue.gef`, `*_regist.tif`) are not read.

| element | key | used by |
|---|---|---|
| bin-1 counts | `tables['bin1_table']` / `points['bin1_genes']` (DNB coords) | panel, field, segment |
| registered ssDNA (1 px = 1 DNB) | `images['<chip>_regist']` | tissue mask |
| SAW cell segmentation | `shapes['cells_polygons']` | segment seeds |

The bins and the image are in DNB units in `'global'`; SSAF-T converts them in memory (bin centres at
DNB x 0.5 µm, ssDNA at 0.5 µm/px). The map and the cells are written back into `'global'` on the DNB
grid.
:::

::::

### Array-based platforms (Visium HD, Stereo-seq)

The SpatialData returned by the reader is passed directly to `compute_vf`, `downsample_vector`,
`map_celltypes`, `filter_celltypemaps` and `segment_cells`. SSAF-T recognises it from
`sdata.attrs['spatialdata_io_reader']` and builds its µm lattice in memory; the SpatialData is not
modified until results are added to it. `ssaft.GridDataset.from_sdata(sdata)` provides the same
converted view, for example to select the gene panel. `compute_vf` records the coordinate system and
the µm -> pixel affine in the store, so that the map and the cells are written into that system
(`<dataset_id>` for Visium HD, `'global'` for Stereo-seq) and overlay the reader's image and bins,
including when later stages run in a new process on a re-read SpatialData.
