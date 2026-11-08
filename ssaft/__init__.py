"""SSAF-T, the Spatial Segmentation-free Agentic Framework for Transcriptomics.

This package performs segmentation-free, multi-platform spatial cell-type mapping. It
builds a kernel density estimate of the transcript field, assigns a cell type to every
voxel of that field, and cuts cells out of the result. Import it as ``import ssaft``.

The agentic annotation component ships as a separate repository, CeLLMan
(https://github.com/pnucolab/CeLLMan), which is linked here as the ``CeLLMan/``
submodule. CeLLMan is driven by a large language model host and is not imported from
this package.
"""

from ._analysis import Analysis
from ._dataset import Dataset

from ._griddataset import GridDataset
from ._panel import select_gene_panel, resolve_server, fetch_ontology_markers
from ._io import (aggregate_to_grid, load_visiumhd, load_strata,
                  load as load_count)

__version__ = "0.1.0"

try:
    import spatialdata
    _SPATIALDATA_AVAILABLE = True
except ImportError:
    _SPATIALDATA_AVAILABLE = False


def is_spatialdata_available():
    """Report whether the optional ``spatialdata`` dependency could be imported.

    :returns: ``True`` when ``spatialdata`` is installed and importable, ``False`` otherwise.
    :rtype: bool
    """
    return _SPATIALDATA_AVAILABLE
