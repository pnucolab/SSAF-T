# Tile (optional)

Imaging-based platforms only (excluding CosMx, which is processed per FOV). The SpatialData is cut
into square tiles so that `downsample_vector(chunked=True, sdata_dir=...)` streams them rather than
holding the whole section in memory. Tiling affects downsampling only; mapping always reads the whole
SpatialData chunk by chunk.

Tiling is used when the section's transcripts and images do not fit comfortably in memory. Both
verified 3D runs (Xenium kidney, MERFISH liver) were tiled; small sections and crops are processed
untiled.

The script is provided within the package:

```python
import subprocess, sys
from pathlib import Path
import ssaft

SLICE = Path(ssaft.__file__).parent / "scripts" / "slice_spatialdata.py"
subprocess.run([sys.executable, str(SLICE),
                "--sdata-path", "outputs/sample/spatialdata.zarr",
                "--tile-size", "10000",
                "--output-dir", "outputs/sample/spatialdata_crops",
                "--image-key", "morphology_focus", "morphology_zstack",
                "--keep-points", "transcripts",
                "--drop-labels", "--drop-shapes", "--drop-tables"], check=True)
```

One `r{row:02d}_c{col:02d}.zarr` is written per tile.

## Parameters

### tile

Default
: no tiling
Applies to
: imaging-based platforms
Asked
: yes

Whether tiling is performed.

### --tile-size

Default
: none (required)
Applies to
: tiling
Asked
: yes, when tiling
Verified
: Xenium kidney `10000` (Xenium pixels, about 2.1 mm); MERFISH liver `20000`

Tile edge **in the units of the chosen coordinate system**. For Xenium's `global` system these are
pixels rather than µm; the element extent should be inspected before a value is chosen.

### --image-key

Default
: keep all images
Applies to
: tiling
Asked
: no (set from the image choices made for downsampling)

Images to retain in every tile. It must include the images from which the downsampling thresholds are
computed: the 2D image (`image_key`) in two dimensions, and the z-stack (`images`) in three — see
{py:meth}`~ssaft.Analysis.downsample_vector`.

### --keep-points

Default
: keep all
Applies to
: tiling
Asked
: no

Points elements to retain; the transcripts key is supplied.

### --min-tile-size

Default
: half of `--tile-size`
Applies to
: tiling
Asked
: no

A trailing row or column narrower than this is merged into its neighbour.

### --coordinate-system

Default
: `global`
Applies to
: tiling
Asked
: no

### --drop-labels / --drop-shapes / --drop-tables

Default
: keep
Applies to
: tiling
Asked
: no

These drop elements that downsampling does not read; all three are recommended.

### --drop-z

Default
: keep z
Applies to
: tiling
Asked
: no

Drops the transcripts' z column. **It must not be used for a 3D run.**
