# Filter

Removes voxels where there is no cell, and writes the map into the SpatialData. It takes seconds,
so try several settings on the same mapping.

```python
sdata = analysis.filter_celltypemaps(sdata)                 # defaults
sdata = analysis.filter_celltypemaps(sdata, norm_f=0.5)     # stricter density cut
```

Written to the store: `filtered_celltype_maps` (`-1` = background). Written to `sdata`:

- `labels['ssaft_celltype_map']`: pixel value = class id + 1 (0 = background); a 3D map keeps every
  layer as `(z, y, x)`;
- `images['ssaft_celltype_map_rgb']`: an RGB rendering (the middle layer in 3D).

Class names for the pixel values:
`sdata['ssaft_downsampled_vectors'].obs['cellman_annotation'].cat.categories[pixel - 1]`.

There are two independent questions and one parameter for each:

| question | parameter |
|---|---|
| Is there a cell at this voxel? | `norm_f` (per-cell-type density cut), optionally `min_norm` |
| Is the label right? | `min_p` (lda/gnb) or `min_r` (corr) |

## Parameters

### norm_f

Default
: `0.3`
Applies to
: all
Asked
: yes
Verified
: `0.3`

Per-cell-type density cut: a voxel is kept when its `vf_norm` is at least `norm_f` times the density
peak of its own cell type (half-sample mode of `log10(vf_norm)`). Density differs between cell types,
so one global cut removes whole sparse types; this does not. On the 3D mouse-kidney map `0.3`
removed 8.3% of mapped voxels and `0.6` removed 28.1%, emptying the medulla between tubules. `None`
turns it off.

### min_p

Default
: `0` (keep every mapped voxel)
Applies to
: `lda` / `gnb` maps
Asked
: no

Posterior threshold. The posterior answers "which of these cell types", not "is there a cell", and
sibling subtypes sharing markers split it between them, so the same `0.9` removed 12% of voxels on
one panel and 56% on another. Raise it only for unambiguous calls.

### min_r

Default
: `0.6`
Applies to
: `corr` maps
Asked
: no

### min_norm

Default
: `0`
Applies to
: all
Asked
: no

Global density floor. A string selects a scikit-image `threshold_*` method (options via
`filter_params`).

### norm_depth

Default
: `True`
Applies to
: 3D
Asked
: no

Divide each plane's density by its median before the per-cell-type cut. Density falls off away from
the middle of the section; without it the outer planes are removed wholesale.

### hsm_f

Default
: `None` (off)
Applies to
: `lda` maps with `marker_genes_path`
Asked
: no

Per-cell-type cut on the voxel's own-marker score, as a fraction of that type's peak. Reaches only
cell types with markers; around `0.4` is the useful range.

### show_unknown

Default
: `True`
Applies to
: all
Asked
: yes

Keep the annotation's `unknown1..N` classes on the map (where the annotation abstained). `False`
sends them to background; they are never reassigned to a runner-up.

### fill_blobs / min_blob_area

Default
: `True` / `0`
Applies to
: all
Asked
: no

Fill enclosed holes in kept regions; drop regions smaller than `min_blob_area` voxels.

### z

Default
: middle layer
Applies to
: 3D
Asked
: no

Layer rendered to the RGB image. The labels always keep every layer.

### background

Default
: `'white'`
Applies to
: all
Asked
: no

Colour of removed voxels in the RGB image.

### labels_key / image_key

Default
: `'ssaft_celltype_map'` / `'ssaft_celltype_map_rgb'`
Applies to
: all
Asked
: no
