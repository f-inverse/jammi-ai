"""The verified citation map: one recipe ↔ one monograph equation ↔ one canon line.

This is the source of truth for the bridge chapters' citation table and the
verticals' bridge notes. Each :class:`CitationRow` pins a recipe to its monograph
reference(s) (Stanković et al. Parts I/II/III), its GNN- or conformal-canon
reference(s), and the exact ``jammi_ai`` verb it runs.

Two contracts hold this honest, both enforced by ``tests/test_citation_map.py``:

* every ``bib_keys`` entry exists in ``references.bib`` (no dangling ``@cite``);
* every ``jammi_call`` verb appears in the grounded API reference
  (``jammi_cookbook/_api_reference.md``) — the map cannot cite a verb that does
  not exist on the pinned engine.

Each citation was independently verified (author / year / venue) against the
primary source; the corrections found against the hand-off spec are recorded in
``EXECUTION-STATUS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationRow:
    """One row of the citation map: recipe ↔ theory ↔ canon ↔ engine call."""

    recipe: str
    monograph: str  # the Stanković et al. part + what it is there
    canon: str  # the GNN / conformal-canon reference(s), human-readable
    jammi_call: str  # the exact verb the recipe runs (must exist in the API reference)
    bib_keys: tuple[str, ...]  # every references.bib key this row cites


# The map covers every recipe in the verticals (tiers 01–04, both datasets) plus
# the conformal repair. The Jammi call is the verb actually run; where a recipe is
# theory-only on one axis (tier 04 has no monograph analogue) the monograph cell
# says so explicitly rather than inventing a reference.
CITATION_MAP: tuple[CitationRow, ...] = (
    CitationRow(
        recipe="01 construct — kNN similarity graph",
        monograph="Part I — graph construction, the Laplacian, spectral clustering",
        canon="(graph construction is the monograph's own; the GNN canon assumes a given graph)",
        jammi_call="build_neighbor_graph",
        bib_keys=("stankovic2019gsp1",),
    ),
    CitationRow(
        recipe="02 analyze — propagation (low-pass graph filter)",
        monograph="Part II — low-pass graph filtering / the graph Fourier transform",
        canon="SGC (Wu et al. 2019); APPNP (Gasteiger/Klicpera et al. 2019)",
        jammi_call="propagate_embeddings",
        bib_keys=("stankovic2019gsp2", "wu2019sgc", "gasteiger2019appnp"),
    ),
    CitationRow(
        recipe="03 learn — graph-supervised metric learning",
        monograph="Part III — graph embeddings, machine learning on graphs",
        canon="node2vec (Grover & Leskovec 2016); GraphSAGE (Hamilton et al. 2017)",
        jammi_call="fine_tune_graph",
        bib_keys=("stankovic2020gsp3", "grover2016node2vec", "hamilton2017graphsage"),
    ),
    CitationRow(
        recipe="02/03 retrieval — an edge is a self-search; a context set is a search/walk",
        monograph="Part I — kNN-graph construction as retrieval",
        canon="the retrieval-conditioned context of a Neural Process (Garnelo et al. 2018)",
        jammi_call="assemble_context",
        bib_keys=("stankovic2019gsp1", "garnelo2018cnp"),
    ),
    CitationRow(
        recipe="04 predict — context-conditioned posterior",
        monograph="(no monograph analogue — the tier the GSP series does not reach)",
        canon="CNP (Garnelo et al. 2018); ANP (Kim et al. 2019); TabPFN (Hollmann et al. 2022)",
        jammi_call="train_context_predictor",
        bib_keys=("garnelo2018cnp", "kim2019anp", "hollmann2022tabpfn"),
    ),
    CitationRow(
        recipe="04 predict — graph-conditioned prediction (BYOG)",
        monograph="(no monograph analogue)",
        canon="ANP attention over the context set (Kim et al. 2019)",
        jammi_call="predict_with_context_predictor",
        bib_keys=("kim2019anp",),
    ),
    CitationRow(
        recipe="04 quantify — marginal conformal + the graph break",
        monograph="(no monograph analogue)",
        canon="Vovk et al. 2005; Angelopoulos & Bates 2021; Barber et al. 2023; "
        "CF-GNN (Huang et al. 2023); NAPS (Clarkson 2023)",
        jammi_call="conformalize",
        bib_keys=("vovk2005conformal", "angelopoulos2021conformal", "barber2023beyond",
                  "huang2023cfgnn", "clarkson2023naps"),
    ),
    CitationRow(
        recipe="04 quantify — regression interval + the inline repair",
        monograph="(no monograph analogue)",
        canon="Tibshirani et al. 2019 (weighted conformal under covariate shift); "
        "Barber et al. 2023 (beyond exchangeability)",
        jammi_call="conformalize_interval",
        bib_keys=("tibshirani2019covariateshift", "barber2023beyond"),
    ),
)
