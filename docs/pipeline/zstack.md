# Z-stack image (3D only)

**Skip this stage for a 2D run.**

A 3D run samples anchors plane by plane, so it needs the morphology stack as a 3D SpatialData image
element `(c, z, y, x)`. The 2D morphology images the vendor readers load carry no depth.
`Analysis.zstack_image` wraps the stack lazily (nothing is read into memory) and places it on the
pixel grid of an existing 2D image.

```python
sdata.images["morphology_zstack"] = ssaft.Analysis.zstack_image(
    "path/to/morphology.ome.tif", like=sdata.images["morphology_focus"])
sdata.write("outputs/sample/spatialdata.zarr")
```

The element must be written with the rest of the SpatialData: `downsample_vector` reads images only
from `sdata.images` (or from the tiles, which copy it).

Plane `k` is placed at `z_origin + k * z_step` in the transcripts' z units. Planes outside the
transcripts' z range are skipped during downsampling.

## Parameters

### source

Default
: none
Applies to
: 3D
Asked
: yes (the agent proposes the file it finds)

The stack. One of:

- a ZYX stack file (Xenium `morphology.ome.tif`);
- a folder of per-plane `*_z<N>.tif` files (MERSCOPE `images/mosaic_DAPI_z0.tif` ...);
- a glob pattern or a list of files.

### like

Default
: none (identity x/y transformation)
Applies to
: 3D
Asked
: no

The 2D image element whose pixel grid the planes share, e.g. `sdata.images['morphology_focus']`.
Its transformations place the stack in every coordinate system, and its size is checked against the
planes. Always pass it.

### z_step

Default
: read from OME `PhysicalSizeZ` for a stack file; **required** for per-plane files
Applies to
: 3D
Asked
: yes, when it cannot be read from the file

Distance between consecutive planes, in the transcripts' z units. It must describe the planes you
actually pass: MERSCOPE planes are 1.5 µm apart, so passing only `z0` and `z3` needs
`z_step=4.5`.

### z_origin

Default
: `0.0`
Applies to
: 3D
Asked
: no

z of plane 0 in the transcripts' z units.

### scale_factors

Default
: `[2, 2, 2]`
Applies to
: 3D
Asked
: no

x/y pyramid factors of the element. z is never downscaled.

### chunk

Default
: `2048`
Applies to
: 3D
Asked
: no

Dask chunk edge, pixels.
