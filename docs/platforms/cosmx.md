# CosMx

NanoString CosMx SMI. Transcript regime; 2D or 3D. **Run per field of view (FOV).**

```{warning}
**Experimental.** No CosMx dataset has been run end to end with the current pipeline.
```

## Input

The flat-file export: `<dataset_id>_tx_file.csv`, `<dataset_id>_fov_positions_file.csv`,
`<dataset_id>_exprMat_file.csv`, `<dataset_id>_metadata_file.csv`, and the `CellComposite/` and
`CellLabels/` folders.

```python
import spatialdata_io
sdata = spatialdata_io.cosmx(path, dataset_id="<dataset_id>", transcripts=True)
```

`spatialdata_io.cosmx` returns **one element per FOV**, each in its own coordinate system named after
the FOV (plus `global`, which places it on the slide):

| element | key | notes |
|---|---|---|
| transcripts | `points['<fov>_points']` | `x`, `y` = `x_local_px`, `y_local_px` (**pixels**); the plane index is kept as `z_raw` |
| composite image | `images['<fov>_image']` | `CellComposite`, RGB, same pixel frame |
| vendor cell labels | `labels['<fov>_labels']` | same pixel frame |

## One run per FOV

Every FOV is a separate SSAF-T run: its own SpatialData zarr, store, clustering h5ad and annotation.
FOVs are small, so tiling is not needed.

The transcripts must be in µm (bandwidth, sampling distance and block size are µm) with a
transformation back to image pixels, as Xenium stores them. Build each FOV's SpatialData like this:

```python
import spatialdata as sd
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, Scale

PX_UM = 0.12028          # µm per pixel; 0.18 on older instruments. Check the export's metadata.
fov = "12"

df = sdata.points[f"{fov}_points"].compute().reset_index(drop=True)
df["x"] = df["x"] * PX_UM
df["y"] = df["y"] * PX_UM
coords = {"x": "x", "y": "y"}
# 3D only: df["z"] = df["z_raw"].astype("float64") * Z_STEP_UM; coords["z"] = "z"

fov_sdata = sd.SpatialData(
    points={"transcripts": PointsModel.parse(
        df, coordinates=coords, feature_key="target",
        transformations={"global": Scale([1 / PX_UM, 1 / PX_UM], axes=("x", "y"))})},
    images={"composite": sd.models.Image2DModel.parse(
        sdata.images[f"{fov}_image"].data, dims=("c", "y", "x"),
        transformations={"global": Identity()})},
)
fov_sdata.write(f"outputs/sample/fov{fov}/spatialdata.zarr")
```

Then run the [pipeline](../pipeline/index.md) on `fov{fov}/` as for Xenium. For a whole slide,
loop over the FOVs; annotations are independent per FOV, so the same cell type can carry a
different cluster id (but the same ontology label) in each.

## Platform defaults

| parameter | value |
|---|---|
| `image_key` | `'composite'` |
| `intensity` | `'max'` (fluorescence composite; luminance would suppress a blue DAPI signal) |
| `revert_image` | `False` |
| tiling | no |
| `bandwidth` / `sampling_distance` / `kernel` | `2.5` / `1.0` / `'gaussian'` (as Xenium; not tuned for CosMx) |

## 3D

The transcript file keeps a plane index (`z_raw`). A 3D run needs it converted to µm and a per-plane
morphology stack for the FOV; the flat-file export does not include one. Use the instrument's raw
per-FOV morphology z-stacks with [`zstack_image`](../pipeline/zstack.md), with `like=` the FOV's
composite image and `z_step` the plane spacing in µm.

## Pitfalls

- **Pixel size.** `PX_UM` differs between instrument generations; a wrong value rescales every µm
  parameter.
- **Low-copy markers.** CosMx panels detect some lineage markers (e.g. `FOXP3`, `CTLA4`) near
  background; populations defined only by such genes may not be resolvable.
- **Nuclei.** Seed segmentation from a DAPI segmentation of the FOV, or from the centroids of
  `labels['<fov>_labels']` (vendor cells, so centroids are approximate nucleus positions).
