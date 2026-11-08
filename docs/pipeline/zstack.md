# Z-stack image (3D only)

**This stage is omitted for a 2D run.**

A 3D run samples representative positions plane by plane and therefore requires the morphology stack
as a 3D SpatialData image element `(c, z, y, x)`. The 2D morphology images the vendor readers load
carry no depth. `Analysis.zstack_image` wraps the stack lazily, reading nothing into memory, and
places it on the pixel grid of an existing 2D image.

Only the imaging-based platforms provide a morphology stack; the array-based platforms (Visium HD,
Stereo-seq) are two-dimensional and omit this stage.

::::{tab-set}
:class: platform-tabs

:::{tab-item} Xenium
:sync: xenium
A single ZYX stack file; `z_step` is read from the OME metadata.

```python
sdata.images["morphology_zstack"] = ssaft.Analysis.zstack_image(
    "path/to/morphology.ome.tif", like=sdata.images["morphology_focus"])
sdata.write("outputs/sample/spatialdata.zarr")
```
:::

:::{tab-item} MERSCOPE
:sync: merscope
A folder of per-plane `*_z<N>.tif` files; `z_step` is supplied in µm (the planes are 1.5 µm apart, so
loading only `z0` and `z3` requires `z_step=4.5`).

```python
sdata.images["morphology_zstack"] = ssaft.Analysis.zstack_image(
    "path/to/images", like=sdata.images["<dataset_id>_z3"], z_step=1.5)
sdata.write("outputs/sample/spatialdata.zarr")
```
:::

:::{tab-item} CosMx
:sync: cosmx
The flat-file export contains no morphology stack; the instrument's raw per-FOV z-stacks are used
instead, with `like=` set to the FOV's composite image and `z_step` the plane spacing in µm.
:::

::::

The element must be written with the rest of the SpatialData, because `downsample_vector` reads
images only from `sdata.images` (or from the tiles, which copy it).

Plane `k` is placed at `z_origin + k * z_step` in the transcripts' z units. Planes outside the
transcripts' z range are skipped during downsampling.

## Parameters

See {py:meth}`~ssaft.Analysis.zstack_image` in the [API reference](../api.md) for every parameter.
