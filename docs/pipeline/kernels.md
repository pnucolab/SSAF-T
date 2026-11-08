# Kernels (KDE / convolution)

The gene expression vector field is built by smoothing the data with a kernel — a kernel density
estimate of the transcripts for imaging-based data, and a convolution of the binned counts for
array-based data. The `kernel` parameter selects the kernel's shape and `bandwidth` (`bandwidth_um`
for `compute_vf`) sets its width. Six kernels are available (`ssaft._count.KDE_KERNELS`):

```text
gaussian (default), tophat, epanechnikov, exponential, linear, cosine
```

## Bandwidth means the same for every kernel

Changing the kernel does not change the width of the smoothing. Each kernel's own scale is derived
from `bandwidth` by **variance matching**, so that every kernel has the same spatial spread (second
moment) at a given `bandwidth`; `bandwidth` is therefore set once and the kernel may be changed
without re-tuning. The per-kernel scale (the `σ` in the figures below) consequently differs — in two
dimensions the tophat support is `bandwidth · √(d+2)`, the Epanechnikov support `bandwidth · √(d+4)`,
and so on — while the effective width is held constant. Panel **B** of each figure plots the
resulting weight profiles `K(r)`.

The kernels fall into two groups:

- **`gaussian`** has infinite support, truncated at `prune_coefficient · bandwidth`.
- **`tophat`, `epanechnikov`, `linear`, `cosine`** are **compact**: they reach exactly zero at their
  support and therefore touch fewer voxels per transcript.

## Which to choose

```{list-table}
:header-rows: 1
:widths: 18 82

* - kernel
  - when
* - **gaussian**
  - The default. Smooth and well characterised; the appropriate choice in the absence of a specific
    reason to prefer another.
* - **epanechnikov / linear / cosine**
  - Smooth and compact. The resulting field is **near-identical to the gaussian field** (median
    cosine similarity ≥ 0.99 and signal-mask Jaccard ≥ 0.99 on both Xenium and Visium HD), and on
    the imaging-based (KDE) path they touch fewer voxels per molecule and are therefore **2–3×
    faster** (Xenium GPU: gaussian 1142 ms against epanechnikov 382, linear 392). These are suitable
    replacements for the gaussian kernel where speed is required.
* - **tophat**
  - A hard box: uniform within its support and zero outside. Fastest on the imaging-based path, but
    it **differs most from the gaussian field structurally** — its difference map (panel C) reveals
    the box edges, and its cosine similarity falls to approximately 0.96–0.98.
* - **exponential**
  - Sharply peaked and heavy near the centre. It differs structurally in the manner of the tophat
    kernel and is the **slowest** everywhere (Xenium GPU 1666 ms; Visium HD CPU 8970 ms), so it is
    rarely advantageous.
```

In summary, the gaussian kernel is retained unless speed is required, in which case `epanechnikov`,
`linear` or `cosine` yield an almost identical field more quickly. The `tophat` and `exponential`
kernels alter the character of the field and should be selected only deliberately.

## Match the kernel across stages (imaging-based data)

On the imaging-based path the kernel is applied twice — once to the representative vectors at
[downsample](downsample.md) and once to the whole lattice at [map](map.md) — so the two **must use
the same `kernel`** (together with `bandwidth` and `sampling_distance`). For array-based data the
kernel is fixed in the stored field at [`compute_vf`](panel_vf.md); downsample and map then read that
field and ignore `kernel`.

## Benchmarks

The same six kernels applied to an imaging-based section (Xenium, KDE) and an array-based section
(Visium HD, convolution). Each figure shows: **A** the field (an RGB rendering of the top three
principal components) per kernel, with a zoom; **B** the kernel weight profiles; **C** the per-voxel
difference from the gaussian field (`1 − cosine similarity`); **D** GPU and CPU time; **E** the
median cosine similarity to the gaussian field; **F** the signal-mask Jaccard against the gaussian
field.

```{figure} ../_static/kernel/kernel_bench_xenium.png
:width: 100%
:alt: Six KDE kernels compared on a Xenium section

Imaging-based data (Xenium, mouse kidney). The compact smooth kernels (epanechnikov, linear, cosine)
reproduce the gaussian field almost exactly and run several times faster on the GPU; tophat and
exponential differ most, and exponential is the slowest.
```

```{figure} ../_static/kernel/kernel_bench_visiumhd.png
:width: 100%
:alt: Six convolution kernels compared on a Visium HD section

Array-based data (Visium HD, mouse small intestine). The same ordering holds: epanechnikov, linear
and cosine track the gaussian field (cosine similarity ≥ 0.99, Jaccard ≥ 0.99), whereas tophat and
exponential diverge, and exponential is markedly slow on the CPU.
```

Parameter reference: the `kernel` argument of {py:meth}`~ssaft.Analysis.compute_vf`,
{py:meth}`~ssaft.Analysis.downsample_vector` and {py:meth}`~ssaft.Analysis.map_celltypes` in the
[API reference](../api.md).
