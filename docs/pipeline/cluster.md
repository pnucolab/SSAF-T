# Cluster

Anchors are clustered like single cells: PCA, a kNN graph, Leiden. SSAF-T does not wrap this step;
use scanpy on the CPU or `rapids_singlecell` on the GPU. Write the result to an h5ad; it is the
input of the annotation.

```python
import rapids_singlecell as rsc

rsc.pp.pca(adata, n_comps=50)
rsc.pp.neighbors(adata, n_neighbors=15)
rsc.tl.leiden(adata, resolution=1.0)
adata.write_h5ad("outputs/sample/<species>_<organ>_cls.h5ad")
```

Do not scale, re-normalise or subset genes before PCA: the vectors are already normalised, and the
mapping later assumes the anchors in the h5ad are row-aligned with the anchors in the store.

## Parameters

### n_comps

Default
: `50`
Applies to
: all
Asked
: no

### pca_backend

Default
: GPU (`rsc.pp.pca`)
Applies to
: all
Asked
: no

Use CPU PCA (`sc.pp.pca(adata, n_comps=50, svd_solver="randomized")`) when the dense
`anchors x panel` matrix does not fit on the GPU. A Visium HD panel of ~5,000 genes with a few
hundred thousand anchors is about 13 GB dense. The kNN graph and Leiden can stay on the GPU.

### n_neighbors

Default
: `15`
Applies to
: all
Asked
: yes
Verified
: `15`

Larger values smooth the graph and can merge rare populations into their neighbours.

### resolution

Default
: `1.0`
Applies to
: all
Asked
: yes
Verified
: `1.0`

Leiden resolution. The annotator splits mixed clusters itself (it re-clusters a cluster on its
ontology children's markers), so over-clustering is not needed; under-clustering is recoverable
too, at the cost of annotation time.

### cluster_key

Default
: `'leiden'`
Applies to
: all
Asked
: no

The `obs` column holding cluster ids. Passed to the annotator as the cluster column and to
`map_celltypes` as `leiden_column`.
