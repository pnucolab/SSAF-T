"""Sphinx configuration for the SSAF-T documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SSAF-T"
author = "PNU Computational Biology Lab"
copyright = "2026, PNU Computational Biology Lab"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "attrs_inline", "attrs_block"]
# Every heading down to ### gets a stable anchor. Parameter sections are ### headings named after
# the parameter, so a page can be linked as e.g. pipeline/downsample.html#n-samples.
myst_heading_anchors = 3

# ssaft is imported from the source tree without being built: the compiled extension, the GPU
# stack and the heavy scientific dependencies are mocked, so autodoc reads signatures and
# docstrings only.
autodoc_mock_imports = [
    "ssaft.utils",
    "numpy", "scipy", "pandas", "sklearn", "skimage", "matplotlib", "seaborn", "cv2",
    "zarr", "numcodecs", "dask", "xarray", "igraph", "leidenalg", "psutil", "packaging",
    "tqdm", "anndata", "scanpy", "spatialdata", "spatialdata_io", "spatialdata_plot",
    "PIL", "tifffile", "requests", "tensorstore", "torch", "triton", "cupy", "cuml",
    "cugraph", "rapids_singlecell", "geopandas", "shapely", "yaml", "numba",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "class"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}

html_theme = "furo"
html_title = "SSAF-T"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/pnucolab/SSAF-T",
    "source_branch": "main",
    "source_directory": "docs/",
}
