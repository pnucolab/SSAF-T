# Load

SSAF-T does not read vendor output itself. Raw data is loaded into a
[SpatialData](https://spatialdata.scverse.org) object with
[`spatialdata_io`](https://spatialdata.scverse.org/projects/io), written to zarr once, and read
back for every later stage.

```python
import spatialdata as sd
import spatialdata_io

sdata = spatialdata_io.xenium("path/to/xenium_output")   # one reader per platform, below
sdata.write("outputs/sample/spatialdata.zarr")
sdata = sd.read_zarr("outputs/sample/spatialdata.zarr")   # always continue from the zarr
```

Reading back from zarr matters: the vendor readers return lazily-parsed elements whose later
access can be far slower than the zarr copy.

## Readers

| platform | reader | what SSAF-T needs from the result |
|---|---|---|
| Xenium | `spatialdata_io.xenium(path)` | `points['transcripts']` (`x`, `y`, `z`, `feature_name`); a 2D morphology image (`morphology_focus`); `shapes['nucleus_boundaries']` for segmentation |
| MERSCOPE | `spatialdata_io.merscope(path)` | the transcripts points element; a mosaic image (DAPI or polyT) |
| CosMx | `spatialdata_io.cosmx(path, dataset_id=...)` | per FOV: `points['<fov>_points']` converted to µm, `images['<fov>_image']`; one run per FOV |
| Visium HD | `spatialdata_io.visium_hd(path, bin_size=2, load_all_images=True, fullres_image_file=...)` | the 2 µm bin table + full-resolution H&E |
| Stereo-seq | `spatialdata_io.stereoseq(path, dataset_id=..., read_square_bin=True)` on SAW output | `tables['bin1_table']` + the registered ssDNA |

Platform-specific details (keys, units, pitfalls) are on each [platform page](../platforms/xenium.md).

## Count platforms (Visium HD, Stereo-seq)

Pass the SpatialData the reader returned straight to `compute_vf`, `downsample_vector`,
`map_celltypes`, `filter_celltypemaps` and `segment_cells`. SSAF-T recognises it from
`sdata.attrs['spatialdata_io_reader']` and builds its µm lattice in memory; the SpatialData is not
modified until results are added to it. `ssaft.GridDataset.from_sdata(sdata)` gives the same
converted view, e.g. to choose the gene panel.

`compute_vf` records the reader's coordinate system and the µm -> pixel affine in the store. The
map (`labels`, `images`) and the cells (`shapes`, `tables`) are written into that coordinate system
(`<dataset_id>` for Visium HD, `'global'` for Stereo-seq), so they overlay the reader's own image and
bins, including when the later stages run in a new process on a re-read SpatialData.

## Parameters

### species

Default
: none
Applies to
: all
Asked
: yes

Species of the sample, `human` or `mouse`. Selects the Cell-Ontology module together with
[`organ`](#organ).

### organ

Default
: none
Applies to
: all
Asked
: yes

Organ of the sample as a Cell-Ontology module name (`kidney`, `brain`, `small_intestine`, `liver`,
`thymus`, ...). Used by the gene panel (count platforms) and by the annotation. The agent resolves
free text ("mouse cortex") to a module and asks when it is ambiguous.

### bin_um (Visium HD)

Default
: `2`
Applies to
: `spatialdata_io.visium_hd(bin_size=...)`
Asked
: no

Which Space Ranger bin level to load. 2 µm is the native resolution and the only one the pipeline
is verified on.

### fullres_image_file (Visium HD)

Default
: none (Space Ranger's own images)
Applies to
: `spatialdata_io.visium_hd`
Asked
: yes, if a full-resolution microscope image exists next to the outputs

The full-resolution H&E (`.btf` / `.tif`). Without it the tissue mask comes from the downscaled
Space Ranger image.
