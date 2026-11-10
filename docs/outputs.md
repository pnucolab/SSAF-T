# Outputs

## Store (`ssaft.Dataset(path)`)

| array / attr | shape | written by |
|---|---|---|
| `vf_normalized`, `vf_normalized_genes` | genes x X x Y (x Z) | `compute_vf` (count); `map_celltypes(save_vf=True)` (transcript) |
| `vf_norm` | X x Y (x Z) | `compute_vf`, `map_celltypes` |
| `normalized_vectors`, `downsampled_vectors`, `genes`, `coordinate_offset` | anchors x genes, ... | `downsample_vector` |
| `attrs['n_dims']` | 2 or 3 | `downsample_vector` |
| `centroids`, `cluster_labels` | classes x genes, anchors | `map_celltypes(h5ad=...)` |
| `celltype_maps` | X x Y (x Z), class id | `map_celltypes` |
| `max_probabilities` / `max_correlations` | X x Y (x Z) | `map_celltypes` (`lda`,`gnb` / `corr`) |
| `celltype_marker_score` | X x Y (x Z) | `map_celltypes(method='lda', marker_genes_path=...)` |
| `attrs['mapping_method']` | str | `map_celltypes` |
| `filtered_celltype_maps` | X x Y (x Z), `-1` = background | `filter_celltypemaps` |

Arrays are indexed `(x, y[, z])`, the transpose of image order. The class names and colours are
given by `ds.celltype_names_colors()`.

## SpatialData elements

| element | kind | written by | notes |
|---|---|---|---|
| `ssaft_downsampled_vectors` | table | `map_celltypes` | anchors with `obs['cellman_annotation']` and `uns['cellman_annotation_colors']`; the categories name the map's pixel values. Usually left in memory, since the store holds the same vectors, names and colours |
| `ssaft_celltype_map` | labels | `filter_celltypemaps` | pixel = class id + 1, 0 = background; `(z, y, x)` in 3D |
| `ssaft_celltype_map_rgb` | image | `filter_celltypemaps` | RGB rendering; middle layer in 3D |
| `ssaft_cell_boundaries_2d` | shapes | `segment_cells` | one polygon per cell (on its seed layer in 3D) |
| `ssaft_cell_boundaries_3d` | shapes | `segment_cells`, 3D | one polygon per cell and layer, with `cell_id`, `z_index`, `z_um` |
| `ssaft_cell_labels_3d` | labels | `segment_cells`, 3D | `(z, y, x)`, value = `label_id` |
| `ssaft_cells` | table | `segment_cells` | cells x genes counts; `obs`: `cellman_annotation`, `area_um2` or `volume_um3`, `n_z`; `obsm['spatial']` in µm |

All elements carry transformations into the same `global` coordinate system as the input image and
the transcripts, so that they overlay directly:

```python
import spatialdata_plot
sdata.pl.render_images("ssaft_celltype_map_rgb").pl.show()
sdata.pl.render_shapes("ssaft_cell_boundaries_2d", color="cellman_annotation",
                       table_name="ssaft_cells").pl.show()
```

`sdata` is modified in memory; the elements to be retained are written explicitly. An
element the zarr already holds cannot be overwritten in place, so a second run has to
`sdata.delete_element_from_disk(name)` before writing it again:

```python
sdata.write_element(["ssaft_celltype_map", "ssaft_celltype_map_rgb",
                     "ssaft_cell_boundaries_2d", "ssaft_cells"])
sdata.tables["ssaft_cells"].write_h5ad("outputs/sample/ssaft_cells.h5ad")
```

## Annotation (`annotation_out/`)

See [Annotate](pipeline/annotate.md#what-it-writes).
