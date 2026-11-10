# Cluster

The representative vectors are clustered as single cells are: by principal component analysis, a
k-nearest-neighbour graph, and the Leiden algorithm. SSAF-T does not wrap this step;
`rapids_singlecell` is used throughout. The result is written to an h5ad, which is the input of the
annotation.

```python
import rapids_singlecell as rsc

rsc.pp.pca(adata, n_comps=50)
rsc.pp.neighbors(adata, n_neighbors=15)
rsc.tl.leiden(adata, resolution=1.0)
adata.write_h5ad("outputs/sample/<species>_<organ>_cls.h5ad")
```

Genes must not be scaled, re-normalised or subset before the principal component analysis: the
vectors are already normalised, and the subsequent mapping assumes that the rows of the h5ad are
row-aligned with the representative vectors held in the store.

SSAF-T does not fix these settings; any clustering may be used. The tutorials apply principal
component analysis (`n_comps=50`), a k-nearest-neighbour graph (`n_neighbors=15`) and the Leiden
algorithm (`resolution=1.0`). The resolution is not critical, because the annotator re-clusters any
mixed cluster on the markers of its ontology children, so over-clustering is unnecessary and
under-clustering is recoverable at the cost of annotation time. The dense `anchors x panel` matrix
can be large — a Visium HD panel of approximately 5,000 genes with several hundred thousand
representative vectors occupies approximately 13 GB — so on a smaller device the vector field's
allocation has to be released before the principal component analysis, or `rsc.pp.pca` has to be
given `chunked=True`, which runs it incrementally. The cluster column (`leiden` by default) is the one passed to
the annotator and, subsequently, to `map_celltypes` as `leiden_column`.
