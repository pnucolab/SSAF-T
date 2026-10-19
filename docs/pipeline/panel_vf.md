# Gene panel and vector field (count platforms)

Visium HD and Stereo-seq only. Transcript platforms skip this stage: their field is computed during
downsampling and mapping.

Whole-transcriptome platforms measure ~20,000 genes, most of them noise at bin resolution. The
field is built on a **panel**: highly variable genes plus every marker the Cell-Ontology lists for
the organ. Markers stay in even when their variance is low, so the annotation can still see them.
`compute_vf` then rasterises the bin counts, blurs every panel gene and stores the normalised field.

```python
grid = ssaft.GridDataset.from_sdata(sdata)

# Stereo-seq: estimate HVGs on coarse pseudo-bins; 0.5 um bins are too sparse
coarse = ssaft.aggregate_to_grid(grid.raw, bin_um=8.0, origin=grid.origin)
panel, onto_markers = ssaft.select_gene_panel(coarse, "mouse", "brain", hvg_n=5000)

ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
analysis.compute_vf(sdata, bin_um=0.5, panel=panel, sampling_distance=1.0,
                    bandwidth_um=8.0, kernel="gaussian", device="cuda")
```

Output in the store: `vf_normalized` (genes x X x Y) and `vf_norm` (X x Y).

## Gene panel parameters (`select_gene_panel`)

### hvg_n

Default
: `5000`
Applies to
: count platforms
Asked
: yes

Number of highly variable genes (scanpy `seurat_v3` on a subsample of bins, technical genes removed).
The ontology markers are added on top, so the panel is larger than this.

### hvg_bin_um

Default
: the loaded bin (no aggregation)
Applies to
: count platforms
Asked
: no
Verified
: Visium HD `2` (no aggregation); Stereo-seq `8.0`

Pseudo-bin edge for the HVG estimate, applied with `ssaft.aggregate_to_grid` before
`select_gene_panel`. Not a function argument: it decides what AnnData you pass in. At 0.5 µm nearly
every bin holds zero or one molecule and the variance estimate collapses; aggregate first.

### n_hvg_sub

Default
: `100000`
Applies to
: count platforms
Asked
: no

Bins subsampled for the HVG variance estimate.

### server_url

Default
: `CELLONTO_SERVER_URL` env, then `CELLONTO_API` env, then the public server
  `https://cellman.pnucolab.com`
Applies to
: count platforms
Asked
: no

Cell-Ontology REST server. `'none'` builds an HVG-only panel, which drops low-variance markers
and is not recommended.

### slots

Default
: `("R", "S", "N", "DF")`
Applies to
: count platforms
Asked
: no

Marker tiers taken from the ontology: required, supporting, negative, defining.

## Vector field parameters (`compute_vf`)

### bin_um

Default
: `2.0`
Applies to
: count platforms
Asked
: no (set by platform)
Verified
: Visium HD `2`; Stereo-seq `0.5`

Raster bin the blur is computed on. Use the platform's native pitch.

### bandwidth_um

Default
: equal to `bin_um` (`sigma_bin=1`)
Applies to
: count platforms
Asked
: yes
Verified
: Visium HD `2.5`; Stereo-seq `8.0`

Kernel width σ in µm. It sets how far a molecule's signal spreads: too small and sparse bins give a
noisy field, too large and neighbouring cells of different types blur together. Sparse capture
(Stereo-seq) needs a wider kernel than Visium HD. Give either this or `sigma_bin` (`bandwidth_um`
wins).

### sigma_bin

Default
: `1`
Applies to
: count platforms
Asked
: no

The same width in bin units (`bandwidth_um / bin_um`).

### sampling_distance

Default
: equal to `bin_um`
Applies to
: count platforms
Asked
: no
Verified
: `1.0`

Spacing of the stored field, µm. The raster is decimated by `round(sampling_distance / bin_um)`, so
the field can be coarser than the bins but never finer: with 2 µm bins and `1.0` it stays at 2 µm.

### kernel

Default
: `'gaussian'`
Applies to
: count platforms
Asked
: yes
Verified
: Visium HD `'epanechnikov'`; Stereo-seq `'gaussian'`

`'gaussian'`, `'tophat'`, `'epanechnikov'`, `'exponential'`, `'linear'` or `'cosine'`. All six have
the same width for the same bandwidth: each kernel's own scale is matched to the Gaussian's variance
(e.g. the Epanechnikov support is `bandwidth * sqrt(ndim + 4)`), and the smooth ones give
near-identical fields. On a count raster only the Gaussian is separable, so it is faster for wide
kernels: a compact kernel wins below `sigma_bin` about 6 and loses above it. The kernel given later
to `map_celltypes` must match.

### prune_coefficient

Default
: `4.3`
Applies to
: count platforms, Gaussian kernel (and the exponential, cut at the same relative height)
Asked
: no

Gaussian truncation radius in σ.

### size_after_normalization

Default
: `10.0`
Applies to
: all
Asked
: no

L1 size each voxel vector is scaled to before `log1p`.

### log_transform

Default
: `True`
Applies to
: all
Asked
: no

### device

Default
: `'cuda'`
Applies to
: count platforms
Asked
: yes (the GPU choice is asked once and used by every stage)

Torch device, e.g. `'cuda:1'`.

### chunk_h / chunk_w / store_chunk

Default
: `64` / `512` / `32`
Applies to
: count platforms
Asked
: no

GPU compute tile (rows x columns per blur pass) and the on-disk chunk of `vf_normalized`. Keep the
compute tile a multiple of `store_chunk`. Lower `chunk_w` if the GPU runs out of memory.
