# MERSCOPE / MERFISH

Vizgen MERSCOPE. Transcript regime; 2D or 3D. **Verified:** mouse liver, 3D.

## Input

The region directory holding `detected_transcripts.csv` and `images/mosaic_<stain>_z<N>.tif`.

```python
import spatialdata_io
sdata = spatialdata_io.merscope(path, z_layers=3)
```

| element | key | used by |
|---|---|---|
| transcripts (`x`, `y` in µm, `z` = the plane index from `global_z`, `gene`) | `points['<dataset_id>_transcripts']` | downsample, map, segment |
| mosaic images, one element per loaded z layer, all stains as channels | `images['<dataset_id>_z3']` | 2D tissue mask; `like=` for the z-stack |

`z_layers` selects which planes are loaded as 2D images (default: the middle plane, 3).

## Stages

| stage | 2D | 3D |
|---|---|---|
| [load](../pipeline/load.md) | `spatialdata_io.merscope` | same, then convert transcript z to µm (below) |
| [z-stack](../pipeline/zstack.md) | - | `zstack_image(path/"images", like=sdata.images["<dataset_id>_z3"], z_step=1.5)`, or a list of planes with their real spacing |
| [tile](../pipeline/tile.md) | optional | optional; verified with `--tile-size 20000` |
| [downsample](../pipeline/downsample.md) | `z=2, image_key="<dataset_id>_z3"` | `z=3, images="<zstack key>"` |
| [cluster](../pipeline/cluster.md) / [map](../pipeline/map.md) / [filter](../pipeline/filter.md) | as Xenium | as Xenium |
| [segment](../pipeline/segment.md) | nuclei from a DAPI segmentation | same |

## Platform defaults

| parameter | value |
|---|---|
| `revert_image` | `False` (DAPI / polyT) |
| `n_samples` | `10` per plane in 3D (verified) |
| `bandwidth` / `sampling_distance` / `kernel` | `2.5` / `1.0` / `'gaussian'` |

## Pitfalls

- **z is a plane index, not µm.** The reader copies `global_z` into `z`; it counts planes (0-6), 1.5 µm apart. For 3D, multiply
  the transcripts' z by the plane spacing before writing the zarr, and give `zstack_image` a
  `z_step` in µm:

  ```python
  from spatialdata.models import PointsModel
  from spatialdata.transformations import get_transformation

  key = next(k for k in sdata.points if k.endswith("_transcripts"))
  pts = sdata.points[key]
  df = pts.compute().reset_index(drop=True)
  df["z"] = df["z"].astype("float64") * 1.5
  sdata.points[key] = PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "z"},
                                        transformations=dict(get_transformation(pts, get_all=True)))
  ```

  Without the conversion the lattice is compressed 1.5x along z. 2D runs ignore z.
- **Stain choice.** The mosaic image carries several stains as channels. DAPI gives a nuclear mask;
  polyT covers cytoplasm and usually gives a fuller tissue mask.
- **Vendor cells are not nuclei.** `cell_boundaries.parquet` outlines whole cells; segment DAPI for
  nuclei seeds.
