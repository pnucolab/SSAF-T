# Troubleshooting

## Installation

**`Illegal instruction` on the first call.** The C extension was built with AVX-512 on a CPU without
it. Reinstall with `--config-settings=setup-args=-Dsimd=disabled`
([details](installation.md#cpus-without-avx-512)).

**`import rapids_singlecell` fails with a `cuml` import error.** `rapids-singlecell` and `cuml` are
mismatched. Install from the pinned lock file on Python 3.11 + CUDA 13.

**Source edits have no effect.** The package was installed non-editably, or the editable install
cannot rebuild. Reinstall with `uv pip install --no-build-isolation -e ./SSAF-T`.

## Downsampling

**No anchors, or anchors outside the tissue.** `revert_image` does not match the stain: `True` for
H&E (dark tissue on bright), `False` for DAPI, polyT, ssDNA.

**Sparse tissue missing from the anchors.** The Otsu mask is too tight; reduce `otsu_mult` (for
example `0.6`) on {py:meth}`~ssaft.Analysis.downsample_vector`.

**`no anchor survived on any of the N planes` (3D).** The z-stack's z transformation does not place
the planes within the transcripts' z range. Verify `z_step`, and for MERSCOPE convert the transcript
z from plane index to micrometres.

**`ValueError: the count backend field is a 2D blur raster`.** Array-based platforms are
two-dimensional; pass `z=2`.

**`otsu_mult is supported for the transcript backend only`.** Use `otsu_threshold` on array-based
platforms.

## Mapping and filtering

**The map appears shifted against the image.** The image-to-molecule registration is incorrect. For
array-based platforms, inspect the overlay in the coordinate system used by the reader
(`<dataset_id>` for Visium HD, `'global'` for Stereo-seq) rather than in another.

**`spatialdata_io.stereoseq` cannot find its files.** The reader expects the directory layout of
older SAW releases (`03.register/`, `04.tissuecut/`, ...); see the Stereo-seq tab on
[load](pipeline/load.md).

**A rare cell type is missing from the map.** Use `method='lda'` (the default) rather than `corr`,
reduce `norm_f`, and confirm that the annotation emitted the type.

**Most voxels removed.** Reduce `norm_f` on {py:meth}`~ssaft.Analysis.filter_celltypemaps`; keep
`min_p` at 0 unless only unambiguous calls are required.

**GPU out of memory.** Reduce `gpu_target_bytes` (mapping) or `chunk_w` (compute_vf), or run PCA on
the CPU (clustering).

## Other

**`print(sdata)` crashes on a large store.** The SpatialData repr traverses every dask graph; the
data is intact. Print the element keys instead: `list(sdata.images)`, `list(sdata.points)`.

**Large numpy allocations stall for minutes with high system CPU.** This is transparent huge page
compaction. Set `NUMPY_MADVISE_HUGEPAGE=0` before starting Python.
