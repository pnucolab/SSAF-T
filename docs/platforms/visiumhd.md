# Visium HD

10x Genomics Visium HD. Count regime; 2D. **Verified:** mouse small intestine
([tutorial](../tutorials.md)).

## Input

The Space Ranger `outs/` directory with `binned_outputs/square_002um/`, plus, recommended, the
full-resolution microscope image (`.btf` / `.tif`). Read it with `spatialdata_io` and pass the result
to SSAF-T as it is:

```python
import spatialdata_io
import ssaft

sdata = spatialdata_io.visium_hd(path, bin_size=2, load_all_images=True,
                                 fullres_image_file="path/to/tissue_image.btf")

analysis = ssaft.Analysis(ssaft.Dataset("outputs/sample/ssaft_store"), ncores=30)
analysis.compute_vf(sdata, bin_um=2, bandwidth_um=2.5, sampling_distance=1.0, panel=panel)
```

| element | key | used by |
|---|---|---|
| 2 µm bin counts, with `obs['array_row' / 'array_col' / 'in_tissue']` and the full-resolution pixel position in `obsm['spatial']` | `tables['<dataset_id>_square_002um']` | panel, vector field, segment (counts per cell) |
| full-resolution H&E | `images['<dataset_id>_full_image']` | tissue mask |

SSAF-T works on a µm lattice; the reader's bins sit at H&E pixels in a coordinate system named after
the dataset (`<dataset_id>`), with no µm contract. The first stage that receives the SpatialData
converts it in memory, without modifying it: the in-tissue bins are placed at
`(array_col, array_row) x 2` µm, and the pixel -> µm affine is fitted from the bins' pixel positions.
The frame is recorded in the store, so the map and the cells are written back into
`<dataset_id>` and overlay the H&E and the bins there, also when later stages run in another process.

## Stages

| stage | call |
|---|---|
| [load](../pipeline/load.md) | `spatialdata_io.visium_hd(path, bin_size=2, load_all_images=True, fullres_image_file=...)` |
| [gene panel](../pipeline/panel_vf.md) | `select_gene_panel(ssaft.GridDataset.from_sdata(sdata).raw, species, organ, hvg_n=5000)` on the 2 µm bins |
| [vector field](../pipeline/panel_vf.md) | `compute_vf(sdata, bin_um=2, bandwidth_um=2.5, sampling_distance=1.0, panel=panel)` |
| [downsample](../pipeline/downsample.md) | `downsample_vector(sdata, z=2, n_samples_per_block=5, block_um=10, vf_norm_percentile=25)` |
| [cluster](../pipeline/cluster.md) | CPU PCA (the dense matrix is large), GPU neighbours and Leiden |
| [map](../pipeline/map.md) | `method="lda"` |
| [filter](../pipeline/filter.md) | defaults |
| [segment](../pipeline/segment.md) | nuclei segmented on the H&E |

## Platform defaults

| parameter | value |
|---|---|
| `bin_um` | `2` |
| `bandwidth_um` | `2.5` |
| `kernel` | `'gaussian'` (default; a compact kernel such as `'epanechnikov'` is slightly faster at this width) |
| `hvg_n` | `5000` |
| `n_samples_per_block` | `5` |
| tissue mask polarity | H&E, inverted (automatic) |

## Pitfalls

- **Nuclei frames.** Add nuclei as a shapes element with the same transformation as the H&E
  (`<dataset_id>` coordinate system). Nuclei segmented on a cropped H&E (ENACT crops to the in-tissue
  bounding box) are in the crop's pixel frame: add the crop origin back first, or they land about
  40 µm off.
- **Anchor count vs the tutorial.** µm per pixel is taken from the fitted affine (0.27383); the
  tutorial used Space Ranger's `microns_per_pixel` (0.27376), so the number of sampled anchors differs
  very slightly.

