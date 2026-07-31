"""SSAF-T — Spatial Segmentation-free Agentic Framework for Transcriptomics.

Segmentation-free, multi-platform spatial cell-type mapping: transcript-level
KDE vector field, per-pixel cell-type assignment (``import ssaft``).

The agentic annotation component ships as a separate repository, CeLLMan
(https://github.com/pnucolab/CeLLMan), linked here as the ``CeLLMan/``
submodule. It is driven by an LLM host, not imported from this package.
"""

from ._analysis import Analysis
from ._dataset import Dataset

from ._griddataset import GridDataset
from ._panel import select_gene_panel, resolve_server, fetch_ontology_markers
from ._io import (aggregate_to_grid, load_stereoseq, load_visiumhd, load_strata,
                  load as load_count)

__version__ = "0.1.0"

try:
    import spatialdata
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False


def is_spatialdata_available():
    """Check if SpatialData is available."""
    return _SPATIALDATA_AVAILABLE
