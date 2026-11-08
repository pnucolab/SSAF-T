# Gene panel and vector field (array-based platforms)

Visium HD and Stereo-seq only. The imaging-based platforms omit this stage, as their field is
computed during downsampling and mapping.

Whole-transcriptome platforms measure approximately 20,000 genes, most of which are noise at bin
resolution. The gene expression vector field is therefore built on a **panel**: the highly variable
genes together with every marker the knowledge graph lists for the organ. The markers are retained
even where their variance is low, so that they remain available to the annotation. `compute_vf` then
rasterises the bin counts, blurs every panel gene with the kernel, and stores the normalised field.

Select the platform below; the selection is retained across the pipeline pages.

::::{tab-set}
:class: platform-tabs

:::{tab-item} Visium HD
:sync: visiumhd
The 2 µm bins are dense enough to estimate highly variable genes without aggregation; a narrow kernel
is used (`bandwidth_um=2.5`).

```python
grid = ssaft.GridDataset.from_sdata(sdata)
panel, onto_markers = ssaft.select_gene_panel(grid.raw, "<species>", "<organ>", hvg_n=5000)

ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
analysis.compute_vf(sdata, bin_um=2.0, panel=panel, sampling_distance=1.0,
                    bandwidth_um=2.5, kernel="gaussian", device="cuda")
```
:::

:::{tab-item} Stereo-seq
:sync: stereoseq
The 0.5 µm bins are too sparse for the variance estimate of the highly variable genes; they are first
aggregated to 8 µm pseudo-bins, and a wide kernel (`bandwidth_um=8`) accommodates the sparse capture.

```python
grid = ssaft.GridDataset.from_sdata(sdata)
coarse = ssaft.aggregate_to_grid(grid.raw, bin_um=8.0, origin=grid.origin)
panel, onto_markers = ssaft.select_gene_panel(coarse, "<species>", "<organ>", hvg_n=5000)

ds = ssaft.Dataset("outputs/sample/ssaft_store")
analysis = ssaft.Analysis(ds, ncores=30, verbose=True)
analysis.compute_vf(sdata, bin_um=0.5, panel=panel, sampling_distance=1.0,
                    bandwidth_um=8.0, kernel="gaussian", device="cuda")
```
:::

::::

Output in the store: `vf_normalized` (genes x X x Y) and `vf_norm` (X x Y).

## Parameters

The two functions' parameters are in the [API reference](../api.md):
{py:func}`~ssaft.select_gene_panel` for the gene panel and {py:meth}`~ssaft.Analysis.compute_vf` for
the vector field.

The convolution `kernel` is `gaussian` by default; `tophat`, `epanechnikov`, `exponential`, `linear`
and `cosine` are also available (same `bandwidth_um` meaning) — see [Kernels](kernels.md).
