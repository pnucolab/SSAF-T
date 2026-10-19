# Xenium

10x Genomics Xenium. Transcript regime; 2D or 3D. **Verified:** mouse kidney, 3D
([tutorial](../tutorials.md)).

## Input

The Xenium Onboard Analysis output directory (the one holding `experiment.xenium`,
`transcripts.parquet`, `morphology_focus/` or `morphology_focus.ome.tif`, and, for 3D,
`morphology.ome.tif`).

```python
import spatialdata_io
sdata = spatialdata_io.xenium(path)
```

| element | key | used by |
|---|---|---|
| transcripts (`x`, `y`, `z` in µm, `feature_name`) | `points['transcripts']` | downsample, map, segment |
| 2D morphology (DAPI, plus boundary/interior stains in newer bundles) | `images['morphology_focus']` | 2D tissue mask; `like=` for the z-stack |
| nuclei | `shapes['nucleus_boundaries']` | segment |

## Stages

| stage | 2D | 3D |
|---|---|---|
| [load](../pipeline/load.md) | `spatialdata_io.xenium` | same |
| [z-stack](../pipeline/zstack.md) | - | `zstack_image(path/"morphology.ome.tif", like=sdata.images["morphology_focus"])`; `z_step` is read from the OME metadata |
| [tile](../pipeline/tile.md) | optional | optional; verified with `--tile-size 10000` |
| [downsample](../pipeline/downsample.md) | `z=2, image_key="morphology_focus"` | `z=3, images="morphology_zstack"` |
| [cluster](../pipeline/cluster.md) | PCA 50, 15 neighbours, Leiden 1.0 | same |
| [map](../pipeline/map.md) | `method="lda"` | same |
| [filter](../pipeline/filter.md) | defaults | defaults (`norm_depth=True`) |
| [segment](../pipeline/segment.md) | `nuclei="nucleus_boundaries"` | same |

## Platform defaults

| parameter | value |
|---|---|
| `revert_image` | `False` (bright nuclei on a dark background) |
| `intensity` | `'luminance'` |
| `n_samples` | `10` per plane in 3D (verified); `5` in 2D |
| `otsu_mult` | `0.6` (verified in 3D) |
| `bandwidth` / `sampling_distance` / `kernel` | `2.5` / `1.0` / `'gaussian'` |

## Pitfalls

- **Coordinate units.** Xenium's `global` coordinate system is in morphology **pixels**
  (0.2125 µm), while transcripts and the SSAF-T store are in µm. `--tile-size` is in pixels.
  SSAF-T's raster outputs carry the transcripts' µm -> pixel transformation, so they overlay the
  image in `global`.
- **Out-of-range planes.** Each field of view covers 10 of the 11 planes; the stack is zero where a
  field of view was not imaged, and planes outside the transcripts' z range are skipped.
- **3D output size.** The z-stack and its tiles take tens of GB (about 60 GB for the tutorial
  section).
