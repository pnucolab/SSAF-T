# Segment (optional)

Cuts the filtered map into single cells with a cell-type-constrained watershed seeded by nuclei.
Cells of different types are separated where the map changes type; cells of the same type are
separated midway between their nuclei. Skip this stage if you only need the map, or if no nuclei are
available.

```python
sdata = analysis.segment_cells(sdata, nuclei="nucleus_boundaries")
cells = sdata.tables["ssaft_cells"]
```

- **2D** store (or `z` given): one plane; writes `shapes['ssaft_cell_boundaries_2d']` and
  `tables['ssaft_cells']`.
- **3D** store: the watershed runs through every layer, seeded on the middle layer; additionally
  writes `labels['ssaft_cell_labels_3d']` and `shapes['ssaft_cell_boundaries_3d']` (one row per cell
  and layer).

`tables['ssaft_cells']` is cells x genes (transcripts or bins counted per cell), with
`obs['cellman_annotation']`, area or volume, and `obsm['spatial']`.

## Nuclei per platform

| platform | nuclei |
|---|---|
| Xenium | `shapes['nucleus_boundaries']` from `spatialdata_io.xenium` |
| MERSCOPE | vendor cell boundaries are cells, not nuclei; segment DAPI (e.g. Cellpose) and add them as shapes |
| CosMx | per FOV: a DAPI segmentation, or centroids of `labels['<fov>_labels']` |
| Visium HD | nuclei segmented on the H&E (e.g. StarDist / ENACT), added as shapes with the H&E's transformation |
| Stereo-seq | SAW `shapes['<chip>_polygons']`, or nuclei segmented on the ssDNA image |

Nuclei must land on the map. Check an overlay before segmenting: an offset of a few tens of µm
(a crop origin not added back, a transposed axis) assigns every nucleus to a neighbouring cell type.

## Parameters

### segment

Default
: no
Applies to
: all
Asked
: yes

Whether to run this stage.

### nuclei

Default
: none (**required**)
Applies to
: all
Asked
: yes (the agent lists shapes / points elements it found)

A shapes element key (polygon centroids are used), a points element with `x` / `y`, or an `(N, 2)`
array in the map's µm frame.

### nuclei_frame

Default
: automatic (the frame that lands on the map)
Applies to
: all
Asked
: no

Force `'intrinsic'` or `'global'`.

### min_area_um2

Default
: `5.0`
Applies to
: all
Asked
: no

Drop cells smaller than this (3D: by their largest layer).

### max_area_um2

Default
: `None` (no cap)
Applies to
: all (3D: per layer, i.e. the cross-section)
Asked
: no

Cap on one cell's area, **off unless you set it**. A nucleus with no neighbouring nucleus of its own
type floods the whole region the map gives it -- there is nothing else for the watershed to stop
against -- so at a tissue edge or for a sparse cell type single "cells" grow to hundreds of µm². The
cap keeps each cell's voxels in order of their distance to its own nucleus and gives the rest back to
the background: the water was poured on the nucleus, so the nucleus is always kept and the cell is
trimmed from the outside inwards. It is a per-dataset correction rather than a default, because
otherwise it also trims the cells whose size the map and the neighbouring nuclei had right; `100`
(about an 11 µm disc) is a reasonable value where the flooding is a real problem.

### simplify

Default
: `None` (exact)
Applies to
: all
Asked
: no

Douglas-Peucker tolerance in µm for the polygons.

### z

Default
: `None`
Applies to
: 2D; on a 3D store, segments that one layer in 2D
Asked
: no

### transcripts / bins

Default
: the only points element / the count table
Applies to
: transcript / count
Asked
: no

What is counted into `tables['ssaft_cells']`.

### tile / pad_um

Default
: `1024` / `64.0`
Applies to
: 3D
Asked
: no

xy tile size (voxels) and overlap (µm) of the parallel 3D watershed.

### shapes_key / shapes_3d_key / labels_3d_key / table_key / column

Default
: `'ssaft_cell_boundaries_2d'` / `'ssaft_cell_boundaries_3d'` / `'ssaft_cell_labels_3d'` /
  `'ssaft_cells'` / `'cellman_annotation'`
Applies to
: all
Asked
: no
