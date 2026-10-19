# Downsample (anchors)

Samples anchor positions inside the tissue and returns their vectors as an AnnData
(`anchors x genes`, `obsm['spatial']` in µm, `obs['vf_norm']`), ready for clustering. The anchors are
also written to the store.

```python
ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
```

::::{tab-set}

:::{tab-item} Transcript, 2D
```python
adata = analysis.downsample_vector(
    sdata, z=2, image_key="morphology_focus", revert_image=False,
    n_samples=5, block_um=10, bandwidth=2.5, sampling_distance=1.0,
    kernel="gaussian", gpu_device="cuda")
```
:::

:::{tab-item} Transcript, 3D, tiled
```python
adata = analysis.downsample_vector(
    z=3, images="morphology_zstack", chunked=True, sdata_dir="outputs/sample/spatialdata_crops",
    n_samples=10, block_um=10, otsu_mult=0.6, revert_image=False,
    bandwidth=2.5, sampling_distance=1.0, kernel="gaussian", gpu_device="cuda")
```
:::

:::{tab-item} Count (Visium HD, Stereo-seq)
```python
# after compute_vf on the same store
adata = analysis.downsample_vector(sdata, z=2, n_samples_per_block=5, block_um=10,
                                   vf_norm_percentile=25)
```
:::

::::

How the tissue mask is made: the image is collapsed to grey (`intensity`), inverted if the tissue is
dark on a bright background (`revert_image`), and thresholded by Otsu (optionally scaled by
`otsu_mult`, or replaced by `otsu_threshold`). In 3D every plane gets its own threshold; with tiles,
the threshold is the median over tiles.

## Parameters common to both regimes

### z

Default
: none (**required**)
Applies to
: all
Asked
: yes, first question for Xenium, MERSCOPE and CosMx; fixed to `2` for count platforms

`2` or `3`. Recorded in the store; mapping, filtering and segmentation follow it.

### block_um

Default
: `10`
Applies to
: all
Asked
: no
Verified
: `10`

Edge of the sampling block, µm. Sampling density is `n_samples` (or `n_samples_per_block`) per
`block_um` x `block_um`.

### gpu_device

Default
: `'cuda'`
Applies to
: all
Asked
: yes (one GPU choice for the whole run)

### auto_gpu

Default
: `True`
Applies to
: transcript
Asked
: no

Route the KDE to the GPU when CUDA, torch and triton are available; `False` forces the CPU path.

### otsu_threshold

Default
: `None` (Otsu)
Applies to
: all
Asked
: no

A fixed tissue threshold instead of Otsu. Cannot be combined with `otsu_mult`.

### seed

Default
: `42` (transcript), `0` (count)
Applies to
: all
Asked
: no

## Transcript regime

### n_samples

Default
: `5`
Applies to
: transcript
Asked
: yes
Verified
: 3D `10` (per plane, Xenium kidney and MERFISH liver)

Anchors per `block_um` block of tissue mask. In 3D this is per plane, so the total grows with the
number of planes. More anchors resolve rarer cell types but cost clustering and annotation time.

### image_key

Default
: `'he'`
Applies to
: transcript, 2D
Asked
: yes (the agent lists the image elements it found)

Key of the 2D image in `sdata.images` (or in every tile) the tissue mask is made from, e.g.
`morphology_focus` (Xenium). The default rarely exists in a vendor SpatialData; set it.

### images

Default
: `None`
Applies to
: transcript, 3D (required)
Asked
: no (the z-stack element built in the [z-stack stage](zstack.md))

Key of the 3D image element `(c, z, y, x)`.

### revert_image

Default
: `True`
Applies to
: transcript
Asked
: yes
Verified
: `False` for DAPI / polyT / fluorescence

`True` for a bright background with dark tissue (H&E): the image is inverted before thresholding.
`False` for a dark background with bright tissue (DAPI, polyT, most morphology stains). A wrong
value inverts the mask and anchors land outside the tissue.

### intensity

Default
: `'luminance'`
Applies to
: transcript
Asked
: no (set by platform)

How an RGB image becomes grey. `'luminance'` for brightfield; `'max'` (per-pixel channel maximum)
for fluorescence composites such as CosMx CellComposite, where the signal can sit in one channel
that luminance weights down.

### otsu_mult

Default
: `None` (Otsu as computed)
Applies to
: transcript
Asked
: yes
Verified
: Xenium kidney 3D `0.6`

Scales the Otsu threshold. Below 1 widens the mask: on the mouse-kidney Xenium section `0.6` took
the threshold from 59.0 to 35.4, the mask from 12.0% to 23.3% of the lattice and the anchor count
from 258,582 to 502,339. Use it when sparse tissue (interstitium, fibrous regions) falls outside the
default mask.

### bandwidth

Default
: `2.5`
Applies to
: transcript
Asked
: yes
Verified
: `2.5`

KDE σ in µm. The value given to `map_celltypes` must be the same.

### sampling_distance

Default
: `1.0`
Applies to
: transcript
Asked
: no

Lattice spacing in µm (x, y and, in 3D, z). The value given to `map_celltypes` must be the same.

### kernel

Default
: `'gaussian'`
Applies to
: transcript
Asked
: yes
Verified
: `'gaussian'`

KDE kernel: `'gaussian'`, `'tophat'`, `'epanechnikov'`, `'exponential'`, `'linear'` or `'cosine'`,
on the GPU and the CPU alike. `bandwidth` means the same width for all six: each kernel's own scale
is matched to the Gaussian's variance. **Must match the `kernel` given to `map_celltypes`**: the
anchors become the mapper's dictionary, and scoring one kernel's field against another's dictionary
is inconsistent. The compact kernels (tophat, Epanechnikov, linear, cosine) stop at their support
instead of at 4.3 bandwidths and run 2-3x faster; the smooth ones give near-identical fields to the
Gaussian.

### chunked

Default
: `False`
Applies to
: transcript
Asked
: follows the [tile](tile.md) choice

`True` reads the tiles in `sdata_dir` instead of `sdata`.

### sdata_dir

Default
: `None`
Applies to
: transcript, `chunked=True`
Asked
: no

Directory holding the `*.zarr` tiles.

### prune_coefficient / size_after_normalization / log_transform

Default
: `4.3` / `10` / `True`
Applies to
: transcript
Asked
: no

As in [compute_vf](panel_vf.md#prune_coefficient).

## Count regime

The field already exists, so anchors are looked up in `vf_normalized`. The kernel was fixed by
`compute_vf`; `kernel`, `bandwidth` and `otsu_mult` are not used (`otsu_mult` raises).

### n_samples_per_block

Default
: `3`
Applies to
: count
Asked
: yes
Verified
: `5`

Anchors per `block_um` block of tissue mask. `n_samples` is accepted as an alias.

### vf_norm_percentile

Default
: `25.0`
Applies to
: count
Asked
: no
Verified
: `25`

Drops anchors whose `vf_norm` is below this percentile of the non-zero field, i.e. the
lowest-density quarter, where blur halos and empty tissue sit. `None` uses the analytic
threshold (`norm_threshold_mult`).

### norm_threshold_mult

Default
: `2.0`
Applies to
: count, `vf_norm_percentile=None`
Asked
: no

### lookup_n_workers / lookup_ram_gb

Default
: `ncores` / `32.0`
Applies to
: count
Asked
: no

Parallelism and RAM budget of the tile lookup. Lower `lookup_ram_gb` on small machines.
