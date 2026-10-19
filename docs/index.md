# SSAF-T

**Spatial Segmentation-free Agentic Framework - Transcriptomics**

SSAF-T assigns cell types in spatial transcriptomics without segmenting cells first. Transcripts
(or binned counts) are turned into a kernel-density **vector field**; representative positions of
that field are clustered; the clusters are named by the agentic annotator
[CeLLMan](https://github.com/pnucolab/CeLLMan); and the names are mapped back onto every voxel of
the field. Segmentation, if you want single cells at all, comes last and is constrained by the map.

```{image} ../SSAF-T.png
:alt: SSAF-T overview
:width: 100%
```

## Supported platforms

| platform | data kind | loaded with | 2D | 3D |
|---|---|---|---|---|
| [Xenium](platforms/xenium.md) | transcripts | `spatialdata_io.xenium` | yes | yes |
| [MERSCOPE / MERFISH](platforms/merscope.md) | transcripts | `spatialdata_io.merscope` | yes | yes |
| [CosMx](platforms/cosmx.md) | transcripts, per FOV | `spatialdata_io.cosmx` | yes (experimental) | yes (experimental) |
| [Visium HD](platforms/visiumhd.md) | 2 µm bins | `spatialdata_io.visium_hd` | yes | no |
| [Stereo-seq](platforms/stereoseq.md) | 0.5 µm DNB bins (SAW) | `spatialdata_io.stereoseq` | yes | no |

## Where to start

- New here: [Installation](installation.md), then [Concepts](concepts.md).
- Running a dataset: find your platform under **Platforms**, then follow the
  [pipeline stages](pipeline/index.md) it lists. Every parameter has its own anchor, e.g.
  [`n_samples`](pipeline/downsample.md#n_samples).
- Driving the run from an LLM agent: see [Running with an agent](agent.md).

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
concepts
tutorials
```

```{toctree}
:maxdepth: 1
:caption: Platforms

platforms/xenium
platforms/merscope
platforms/cosmx
platforms/visiumhd
platforms/stereoseq
```

```{toctree}
:maxdepth: 2
:caption: Pipeline

pipeline/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

outputs
api
agent
troubleshooting
```
