"""Gene-panel selection for count-based spatial data.

The vector field for a count platform is built over a chosen gene panel, and that
panel is the union of two sources. The first is data driven: the highly variable genes
found by the seurat_v3 method on a coarse-bin subsample, less a denylist of technical
genes. The second is knowledge driven: the marker genes of the Cell-Ontology module
for the sample's species and organ, fetched through the read-only slim-index REST API.

The knowledge graph is reached only through that REST endpoint, never through a raw
neo4j bolt connection. The server URL is resolved in a fixed order of precedence: the
``server_url`` argument, then the ``CELLONTO_SERVER_URL`` environment variable, then
the ``CELLONTO_API`` environment variable kept for backward compatibility, and finally
:data:`DEFAULT_ONTOLOGY_SERVER`. That is the same public read-only server, in the same
order, that ``CeLLMan/scripts/helper.py`` uses, so the panel and the annotator can
never disagree about whether the knowledge graph exists. Setting any of those sources
to an empty string, or to ``none``, disables the knowledge graph explicitly.

When the knowledge graph is disabled or unreachable the panel degrades to the highly
variable genes alone, so the pipeline never fails on a missing knowledge graph. That
degradation is worth noticing rather than ignoring: a marker gene missing from the
panel is off-panel at annotation time, which is exactly what taking the union of the
two sources exists to prevent.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData

DEFAULT_HVG_N = 5000
DEFAULT_HVG_SUB = 100_000

#: The single Cell-Ontology server we publish (public, read-only) — identical to
#: ``DEFAULT_ONTOLOGY_SERVER`` in ``Agentic-Annotation/scripts/helper.py``. Point
#: ``CELLONTO_SERVER_URL`` at your own deployment to override it.
DEFAULT_ONTOLOGY_SERVER = "https://cellman.pnucolab.com"
#: Values that explicitly turn the KG off (panel = HVG-only).
_SERVER_DISABLED = {"", "none", "off", "null", "false"}

_DENY_MOUSE_PREFIX = ("mt-", "Rps", "Rpl", "Hba", "Hbb")
_DENY_HUMAN_PREFIX = ("MT-", "RPS", "RPL", "MRPL", "MRPS", "HBA", "HBB")
_DENY_HUMAN_RE = re.compile(
    r"^(ENSG|LINC|MIR|SNOR)|(-AS\d*$)|(-DT$)|(-IT\d*$)|(-OT\d*$)"
    r"|(^RP\d+-)|(^A[CLP]\d{6})|(^Z\d{5})|(\.\d+$)"
)


def _denier(species: str):
    """Return a predicate that marks technical genes to drop from the panel.

    The returned predicate takes a gene symbol and returns ``True`` when that gene should
    be excluded. Which rules apply depends on the species: human symbols are matched
    against mitochondrial, ribosomal and haemoglobin prefixes, against ``MALAT1`` and
    against a regular expression covering unnamed and antisense loci, while mouse symbols
    are matched against the mouse equivalents of those prefixes, against predicted
    ``Gm`` genes and against ``Rik`` cDNA clone names.

    :param species: Species name. Anything that names human selects the human rules, and
        everything else selects the mouse rules.
    :type species: str
    :returns: A predicate taking a gene symbol and returning ``True`` when it is technical.
    :rtype: callable
    """
    sp = (species or "").lower()
    if sp.startswith("human") or sp == "hs" or sp == "homo_sapiens":
        def _denied(g: str) -> bool:
            """Report whether a human gene symbol is technical and should be dropped.

            :param g: The gene symbol to test.
            :type g: str
            :returns: ``True`` when the symbol is mitochondrial, ribosomal, haemoglobin,
                ``MALAT1``, or an unnamed or antisense locus.
            :rtype: bool
            """
            return (g.startswith(_DENY_HUMAN_PREFIX) or g == "MALAT1"
                    or bool(_DENY_HUMAN_RE.search(g)))
        return _denied
    def _denied(g: str) -> bool:
        """Report whether a mouse gene symbol is technical and should be dropped.

        :param g: The gene symbol to test.
        :type g: str
        :returns: ``True`` when the symbol is mitochondrial, ribosomal, haemoglobin, a
            predicted ``Gm`` gene, or a ``Rik`` cDNA clone name.
        :rtype: bool
        """
        return (g.startswith(_DENY_MOUSE_PREFIX) or bool(re.match(r"Gm\d", g))
                or g.endswith("Rik"))
    return _denied


def resolve_server(server_url: Optional[str] = None) -> Optional[str]:
    """Resolve the Cell-Ontology REST base URL in order of precedence.

    The order is the ``server_url`` argument, then the ``CELLONTO_SERVER_URL`` environment
    variable, then the ``CELLONTO_API`` environment variable, and finally
    :data:`DEFAULT_ONTOLOGY_SERVER`. Leaving every source unset therefore means the public
    server, which matches the behaviour of ``_resolve_server`` in ``helper.py``. The
    annotator connects by default, so the panel must connect by default too; otherwise the
    marker genes the annotator will later look for are silently absent from the vector
    field.

    ``CELLONTO_SERVER_URL`` is the canonical variable name and ``CELLONTO_API`` is kept for
    older setups. The first source that is set wins, which means an environment variable
    can be overridden by the argument and a value that is set but empty is not skipped.

    :param server_url: An explicit base URL, which takes priority over the environment
        variables. Pass ``'none'`` or an empty string to force a panel of highly variable
        genes only.
    :type server_url: str or None
    :returns: The resolved base URL with any trailing slash removed, or ``None`` when the
        first source that is set disables the knowledge graph.
    :rtype: str or None
    """
    for v in (server_url,
              os.environ.get("CELLONTO_SERVER_URL"),
              os.environ.get("CELLONTO_API")):
        if v is None:
            continue
        v = str(v).strip()
        return None if v.lower() in _SERVER_DISABLED else v.rstrip("/")
    return DEFAULT_ONTOLOGY_SERVER


def _clean(g: str) -> str:
    """Normalize a marker symbol so that it can match an entry in ``var_names``.

    Everything from the first hyphen or underscore onwards is discarded, which strips the
    suffixes that marker records sometimes carry.

    :param g: The marker symbol to normalize.
    :type g: str
    :returns: The symbol up to its first hyphen or underscore.
    :rtype: str
    """
    return g.split("-")[0].split("_")[0]


def fetch_ontology_markers(species: str, organ: str,
                           server_url: Optional[str] = None,
                           slots: Iterable[str] = ("R", "S", "N", "DF"),
                           timeout: float = 60.0,
                           verbose: bool = True) -> set:
    """Fetch the marker-gene set of one module from the slim-index REST API.

    Genes are collected from the requested term slots, which by default are the required,
    supporting, negative and defining slots. When the knowledge graph is disabled, is
    unreachable, or does not hold the requested module, the function warns and returns an
    empty set, and the caller then proceeds with highly variable genes alone. The default
    server resolved by :func:`resolve_server` is the public read-only one, so no
    configuration is needed for the knowledge graph to be used.

    :param species: Species name of the ontology module.
    :type species: str
    :param organ: Organ name of the ontology module.
    :type organ: str
    :param server_url: Base URL of the REST API. Resolved by :func:`resolve_server` when
        ``None``.
    :type server_url: str or None
    :param slots: Term slots to collect genes from, by default the required, supporting,
        negative and defining slots.
    :type slots: Iterable[str]
    :param timeout: Timeout of the HTTP request, in seconds.
    :type timeout: float
    :param verbose: Whether to print progress and warnings.
    :type verbose: bool
    :returns: The normalized marker symbols, or an empty set when the knowledge graph could
        not be consulted.
    :rtype: set
    """
    url = resolve_server(server_url)
    if url is None:
        if verbose:
            print("[panel][WARN] ontology server explicitly disabled "
                  "(server_url / CELLONTO_SERVER_URL set to none) → HVG-only panel: "
                  "KG markers absent from the panel become off-panel at annotation time")
        return set()
    try:
        import requests
        resp = requests.get(f"{url}/module/{species}/{organ}/slim-index",
                            timeout=timeout)
        resp.raise_for_status()
        onto = resp.json()
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[panel][WARN] ontology fetch failed ({url}, module {species}/{organ}) "
                  f"→ HVG-only panel; KG markers will be off-panel at annotation time: {e}")
        return set()

    markers = set()
    for t in onto.get("terms", {}).values():
        for slot in slots:
            for g in (t.get(slot) or []):
                markers.add(_clean(g))
    if verbose:
        print(f"[panel] slim-index {species}/{organ}: "
              f"n_terms={onto.get('n_terms', len(onto.get('terms', {})))} "
              f"version={onto.get('version')} → {len(markers)} marker genes")
    return markers


def select_hvg(adata: AnnData, n_top_genes: int = DEFAULT_HVG_N,
               n_sub: int = DEFAULT_HVG_SUB, species: str = "mouse",
               seed: int = 0, verbose: bool = True) -> set:
    """Select the top highly variable genes, less the technical-gene denylist.

    Selection uses the seurat_v3 method on a random subsample of the rows. The input
    should be a coarse-bin aggregation, because dense bins give a cleaner variance
    estimate than fine and therefore sparse ones. When the matrix holds no more genes than
    the number requested, every gene that survives the denylist is returned instead.

    :param adata: Bins used for selection. A coarse-bin aggregation is recommended.
    :type adata: AnnData
    :param n_top_genes: Number of highly variable genes to select.
    :type n_top_genes: int
    :param n_sub: Number of bins subsampled for the variance estimate.
    :type n_sub: int
    :param species: Species name, used to pick the technical-gene denylist.
    :type species: str
    :param seed: Seed of the random number generator used for the subsample.
    :type seed: int
    :param verbose: Whether to print the number of genes selected.
    :type verbose: bool
    :returns: The selected gene symbols.
    :rtype: set
    """
    denied = _denier(species)
    if adata.n_vars <= n_top_genes:
        return {g for g in adata.var_names if not denied(g)}
    rng = np.random.default_rng(seed)
    n = min(n_sub, adata.n_obs)
    sub = adata[rng.choice(adata.n_obs, n, replace=False)].copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=n_top_genes, flavor="seurat_v3")
    hvg = set(sub.var_names[sub.var["highly_variable"]].tolist())
    hvg = {g for g in hvg if not denied(g)}
    if verbose:
        print(f"[panel] HVG (seurat_v3 on {n} bins, denylist-filtered): {len(hvg)}")
    return hvg


def select_gene_panel(adata: AnnData, species: str, organ: str, *,
                      hvg: Optional[set] = None,
                      hvg_n: int = DEFAULT_HVG_N, n_hvg_sub: int = DEFAULT_HVG_SUB,
                      server_url: Optional[str] = None,
                      slots: Iterable[str] = ("R", "S", "N", "DF"),
                      seed: int = 0, verbose: bool = True
                      ) -> Tuple[List[str], List[str]]:
    """Build the gene panel as the union of highly variable genes and in-data markers.

    The marker side of the union is restricted to the ontology markers that actually occur
    in ``adata``, and the whole panel is intersected with ``adata.var_names``, so every
    returned symbol is present in the data.

    :param adata: Bins used for selecting highly variable genes. A coarse-bin aggregation
        is recommended. When ``hvg`` is given it is used as it stands and nothing is
        recomputed here.
    :type adata: AnnData
    :param species: Species name of the Cell-Ontology module to fetch markers for.
    :type species: str
    :param organ: Organ name of the Cell-Ontology module to fetch markers for.
    :type organ: str
    :param hvg: A precomputed set of highly variable genes.
    :type hvg: set, optional
    :param hvg_n: Number of highly variable genes to select when computing them here.
    :type hvg_n: int
    :param n_hvg_sub: Number of bins subsampled for the variance estimate.
    :type n_hvg_sub: int
    :param server_url: Base URL of the REST API, resolved by :func:`resolve_server`, so
        that the knowledge graph is used without any configuration. Pass ``'none'`` for a
        panel of highly variable genes only.
    :type server_url: str, optional
    :param slots: Term slots to collect marker genes from.
    :type slots: Iterable[str]
    :param seed: Seed of the random number generator used for the subsample.
    :type seed: int
    :param verbose: Whether to print progress.
    :type verbose: bool
    :returns: A pair. The first element is the sorted panel, being the union of the highly
        variable genes and the in-data ontology markers. The second is the sorted list of
        in-data ontology markers, which is a subset of the panel.
    :rtype: tuple of (list of str, list of str)
    """
    if hvg is None:
        hvg = select_hvg(adata, n_top_genes=hvg_n, n_sub=n_hvg_sub,
                         species=species, seed=seed, verbose=verbose)
    markers = fetch_ontology_markers(species, organ, server_url=server_url,
                                     slots=slots, verbose=verbose)
    full = set(map(str, adata.var_names))
    markers_in_data = markers & full
    panel = sorted((set(hvg) | markers_in_data) & full)
    if verbose:
        print(f"[panel] final panel: {len(panel)} genes "
              f"(HVG {len(hvg)} ∪ ontology-in-data {len(markers_in_data)})")
    return panel, sorted(markers_in_data)
