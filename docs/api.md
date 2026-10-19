# API reference

```{eval-rst}
.. currentmodule:: ssaft
```

## Analysis

```{eval-rst}
.. autoclass:: ssaft.Analysis
   :no-members:

.. automethod:: ssaft.Analysis.zstack_image
.. automethod:: ssaft.Analysis.compute_vf
.. automethod:: ssaft.Analysis.downsample_vector
.. automethod:: ssaft.Analysis.map_celltypes
.. automethod:: ssaft.Analysis.filter_celltypemaps
.. automethod:: ssaft.Analysis.segment_cells
.. automethod:: ssaft.Analysis.set_celltype_palette
```

## Dataset

```{eval-rst}
.. autoclass:: ssaft.Dataset
   :no-members:

.. automethod:: ssaft.Dataset.celltype_names_colors
.. automethod:: ssaft.Dataset.write_celltype_map
```

## Count data

```{eval-rst}
.. autoclass:: ssaft.GridDataset
   :no-members:

.. automethod:: ssaft.GridDataset.from_sdata
.. automethod:: ssaft.GridDataset.from_sdata_zarr
.. automethod:: ssaft.GridDataset.to_spatialdata

.. autofunction:: ssaft.aggregate_to_grid
.. autofunction:: ssaft.select_gene_panel
```
