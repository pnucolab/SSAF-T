# Segment

Cells are recovered from the filtered map by a cell-type-constrained watershed seeded by nuclei,
each cell type's own territory being flooded one type at a time. Cells of different types are
separated where the map changes type, and cells of the same type are separated midway between their
nuclei. Nuclei are required as segmentation seeds (see the platform-specific source below).

```python
sdata = analysis.segment_cells(sdata, nuclei="nucleus_boundaries")
cells = sdata.tables["ssaft_cells"]
```

- **Two-dimensional** store (or `z` given): a single plane; writes
  `shapes['ssaft_cell_boundaries_2d']` and `tables['ssaft_cells']`.
- **Three-dimensional** store: the watershed runs in three dimensions through every layer. Each
  nucleus is seeded on a single layer — its own z (`nuclei_z`) when known, otherwise the layer
  selected by `seed_rule` (default `'majority'`). Writes `labels['ssaft_cell_labels_3d']` and
  `shapes['ssaft_cell_boundaries_3d']` (one row per cell and layer), together with
  `shapes['ssaft_cell_boundaries_2d']`, which holds one outline per cell on its own seed layer.

`tables['ssaft_cells']` is cells x genes (transcripts or bins counted per cell), carrying
`obs['cellman_annotation']`, the area or volume, and `obsm['spatial']`.

The nuclei must fall on the map. An overlay should be inspected before segmenting: an offset of a
few tens of micrometres — a crop origin not added back, or a transposed axis — assigns every nucleus
to a neighbouring cell type.

## Parameters

Every parameter is documented under {py:meth}`~ssaft.Analysis.segment_cells` in the
[API reference](../api.md). In three dimensions, `seed_rule` (default `'majority'`) selects each
nucleus's seed layer when its own z is not known, and `nuclei_z` supplies that z when it is.
