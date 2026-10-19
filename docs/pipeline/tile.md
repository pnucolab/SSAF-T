# Tile (optional)

Transcript platforms only (not CosMx, which runs per FOV). Cuts the SpatialData into square tiles so that
`downsample_vector(chunked=True, sdata_dir=...)` streams them instead of holding the whole section.
Tiling only affects downsampling; mapping always reads the whole SpatialData chunk by chunk.

Tile when the section's transcripts and images do not fit comfortably in RAM. Both verified 3D runs
(Xenium kidney, MERFISH liver) were tiled; small sections and crops run untiled.

The script ships inside the package:

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

It writes `r{row:02d}_c{col:02d}.zarr` per tile.

## Parameters

### tile

Default
: no tiling
Applies to
: transcript platforms
Asked
: yes

Whether to tile at all.

### --tile-size

Default
: none (required)
Applies to
: tiling
Asked
: yes, when tiling
Verified
: Xenium kidney `10000` (Xenium pixels, about 2.1 mm); MERFISH liver `20000`

Tile edge **in the units of the chosen coordinate system**. These are pixels for Xenium's `global`
system, not µm. Check the element extent before choosing.

### --image-key

Default
: keep all images
Applies to
: tiling
Asked
: no (set from the image choices made for downsampling)

Images to keep in every tile. It must include the image downsampling thresholds: the 2D image
([`image_key`](downsample.md#image_key)) for 2D, and the z-stack
([`images`](downsample.md#images)) for 3D.

### --keep-points

Default
: keep all
Applies to
: tiling
Asked
: no

Points elements to keep; give the transcripts key.

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

Drop elements downsampling does not read. Recommended: all three.

### --drop-z

Default
: keep z
Applies to
: tiling
Asked
: no

Drops the transcripts' z column. **Never use it for a 3D run.**
