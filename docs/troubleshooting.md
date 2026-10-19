# Troubleshooting

## Installation

**`Illegal instruction` on the first call.** The C extension was built with AVX-512 on a CPU without
it. Reinstall with `--config-settings=setup-args=-Dsimd=disabled`
([details](installation.md#cpus-without-avx-512)).

**`import rapids_singlecell` fails with a `cuml` import error.** `rapids-singlecell` and `cuml` are out
of step. Use the pinned lock file on Python 3.11 + CUDA 13.

**Source edits have no effect.** The package was installed non-editably, or the editable install
cannot rebuild. Reinstall with `uv pip install --no-build-isolation -e ./SSAF-T`.

## Downsampling

**No anchors, or anchors outside the tissue.** `revert_image` does not match the stain: `True` for
H&E (dark tissue on bright), `False` for DAPI, polyT, ssDNA.

**Sparse tissue missing from the anchors.** The Otsu mask is too tight; lower
[`otsu_mult`](pipeline/downsample.md#otsu_mult) (e.g. `0.6`).

**`no anchor survived on any of the N planes` (3D).** The z-stack's z transformation does not put
the planes inside the transcripts' z range. Check `z_step`, and for MERSCOPE convert transcript z
from plane index to µm.

**`ValueError: the count backend field is a 2D blur raster`.** Count platforms are 2D; pass `z=2`.

**`otsu_mult is supported for the transcript backend only`.** Use `otsu_threshold` on count
platforms.

## Mapping and filtering

**The map looks shifted against the image.** The image-to-molecule registration is off. Count
platforms: check the overlay in the coordinate system the reader used (`<dataset_id>` for Visium HD,
`'global'` for Stereo-seq), not in another one.

**`spatialdata_io.stereoseq` cannot find its files.** It reads the directory layout of older SAW
releases (`03.register/`, `04.tissuecut/`, ...); see [Stereo-seq](platforms/stereoseq.md#input).

**A rare cell type is missing from the map.** Try `method='lda'` (the default) rather than `corr`,
lower `norm_f`, and check the annotation actually emitted it.

**Most voxels removed.** Lower [`norm_f`](pipeline/filter.md#norm_f); keep `min_p` at 0 unless you
need unambiguous calls only.

**GPU out of memory.** Lower `gpu_target_bytes` (mapping), `chunk_w` (compute_vf), or run PCA on the
CPU (clustering).

## Other

**`print(sdata)` crashes on a large store.** The SpatialData repr walks every dask graph; the data is
intact. Print element keys instead: `list(sdata.images)`, `list(sdata.points)`.

**Large numpy allocations stall for minutes with high system CPU.** Transparent huge page compaction.
Set `NUMPY_MADVISE_HUGEPAGE=0` before starting Python.
