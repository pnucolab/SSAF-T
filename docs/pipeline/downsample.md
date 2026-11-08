# Downsample (anchors)

The gene expression vector field is sampled at representative positions within the tissue, and their
vectors are returned as an AnnData (`anchors x genes`, `obsm['spatial']` in µm, `obs['vf_norm']`) for
clustering. The representative vectors are also written to the store.

```python
ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
```

Select the platform below; the selection is retained across the pipeline pages.

::::{tab-set}
:class: platform-tabs

:::{tab-item} Xenium
:sync: xenium

Imaging-based. The tissue mask is taken from the morphology image.

```python
# 2D
adata = analysis.downsample_vector(
    sdata, z=2, image_key="morphology_focus", revert_image=False,
    n_samples=5, block_um=10, bandwidth=2.5, sampling_distance=1.0,
    kernel="gaussian", gpu_device="cuda")

# 3D (tiled z-stack): z=3, otsu_mult=0.6 widens the mask over sparse interstitium
adata = analysis.downsample_vector(
    z=3, images="morphology_zstack", chunked=True, sdata_dir="outputs/sample/spatialdata_crops",
    n_samples=10, block_um=10, otsu_mult=0.6, revert_image=False,
    bandwidth=2.5, sampling_distance=1.0, kernel="gaussian", gpu_device="cuda")
```
:::

:::{tab-item} MERSCOPE
:sync: merscope

Imaging-based. The mask is taken from a mosaic stain (DAPI or polyT); `image_key` is the loaded
z-plane element, for example `<dataset_id>_z3`.

```python
# 2D
adata = analysis.downsample_vector(
    sdata, z=2, image_key="<dataset_id>_z3", revert_image=False,
    n_samples=5, block_um=10, bandwidth=2.5, sampling_distance=1.0,
    kernel="gaussian", gpu_device="cuda")

# 3D: z=3 on the z-stack built in the z-stack stage; n_samples=10 per plane (verified, mouse liver)
adata = analysis.downsample_vector(
    z=3, images="<zstack key>", chunked=True, sdata_dir="outputs/sample/spatialdata_crops",
    n_samples=10, block_um=10, revert_image=False,
    bandwidth=2.5, sampling_distance=1.0, kernel="gaussian", gpu_device="cuda")
```
:::

:::{tab-item} CosMx
:sync: cosmx

Imaging-based, **one run per FOV** (no tiling). The mask is taken from the `composite` image;
`intensity="max"` prevents a single-channel DAPI signal from being weighted down.

```python
adata = analysis.downsample_vector(
    sdata, z=2, image_key="composite", intensity="max", revert_image=False,
    n_samples=5, block_um=10, bandwidth=2.5, sampling_distance=1.0,
    kernel="gaussian", gpu_device="cuda")
```
:::

:::{tab-item} Visium HD
:sync: visiumhd

Array-based. The field already exists (`compute_vf`); representative positions are looked up in
`vf_normalized`, and `kernel`, `bandwidth` and `otsu_mult` are not used.

```python
adata = analysis.downsample_vector(sdata, z=2, n_samples_per_block=5, block_um=10,
                                   vf_norm_percentile=25)
```
:::

:::{tab-item} Stereo-seq
:sync: stereoseq

Array-based; the same call as Visium HD (representative positions looked up in the existing field).

```python
adata = analysis.downsample_vector(sdata, z=2, n_samples_per_block=5, block_um=10,
                                   vf_norm_percentile=25)
```
:::

::::

The tissue mask is constructed as follows: the image is collapsed to grey (`intensity`), inverted
where the tissue is dark on a bright background (`revert_image`), and thresholded by Otsu's method
(optionally scaled by `otsu_mult`, or replaced by `otsu_threshold`). In three dimensions each plane
receives its own threshold; with tiles, the threshold is the median over tiles.

## Parameters

See {py:meth}`~ssaft.Analysis.downsample_vector` in the [API reference](../api.md) for every
parameter. For imaging-based data only, the `kernel`, `bandwidth` and `sampling_distance` set here
are carried into [map](map.md) and **must match** the values given there; for array-based data the
representative positions are looked up in the stored field and these are ignored.

By default the `kernel` is `gaussian`; `tophat`, `epanechnikov`, `exponential`, `linear` and
`cosine` are also available (with the same meaning of `bandwidth`, several of them faster) — see
[Kernels](kernels.md).
