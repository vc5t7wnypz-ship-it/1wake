"""
Graph Probe Definitions
========================
Defines ``GraphProbeDefinition`` — a dataclass describing a graph-backed
interpretability probe — and instantiates the canonical set of probes used
by the WAKE analysis pipeline.

A *graph probe* maps a WakeToken (identified by its ``token_id``) to a
scalar or categorical signal derived from the Neo4j graph.  The signal is
used downstream to:

  1. Train linear probes on LLM residual-stream activations to detect
     whether the model's internal representation encodes the structural
     property described by the probe.
  2. Label axes of the superposition geometry discovered by sparse
     autoencoders (SAEs) run over Wake passage activations.
  3. Drive the anamnesis retrieval system: given an activation vector,
     find graph-adjacent tokens whose probe values predict the vector's
     direction.

Each probe carries a ``cypher_query`` that, when executed with the
parameter ``$token_id``, returns a single column named ``value`` whose
type matches the probe's ``return_type``.

Probe return types
------------------
- ``"integer"``  — discrete count or category index
- ``"float"``    — continuous score in [0, 1] unless noted
- ``"string"``   — categorical label
- ``"list"``     — list of strings (for multi-label probes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ReturnType = Literal["integer", "float", "string", "list"]


@dataclass
class GraphProbeDefinition:
    """
    Metadata descriptor for a single graph-backed probe.

    Attributes
    ----------
    name : str
        Machine-readable slug used as the probe identifier in training
        scripts and result tables (e.g. ``"etymology_depth"``)
    description : str
        Human-readable description of what property this probe measures.
    cypher_query : str
        Parameterised Cypher query.  Must accept ``$token_id`` and return
        a single column named ``value``.
    return_type : ReturnType
        The type of the ``value`` column returned by ``cypher_query``.
    expected_node_types : list[str]
        Neo4j node labels that this probe touches during graph traversal.
        Used for query-planning and schema-validation checks.
    used_by_lens : list[str]
        Lens ``lens_id`` values that consume this probe to modulate their
        foregrounded semantic fields.
    null_value : int | float | str | list
        The value to substitute when the Cypher query returns no rows
        (e.g. a token that has not been annotated).
    """

    name: str
    description: str
    cypher_query: str
    return_type: ReturnType
    expected_node_types: list[str]
    used_by_lens: list[str] = field(default_factory=list)
    null_value: int | float | str | list = 0  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Probe: etymology_depth_probe
# ---------------------------------------------------------------------------
# Measures the depth of the etymology chain reachable from a token:
# specifically, the maximum number of EtymRoot hops from the WakeToken
# to the deepest root reachable (following HAS_ROOT with confidence ≥ 0.7).
# A token with roots from 3 languages, each with a further sub-root,
# scores higher than a monoglot token.
#
# Query returns: count of distinct EtymRoot nodes reachable within 3 hops.
# ---------------------------------------------------------------------------
etymology_depth_probe = GraphProbeDefinition(
    name="etymology_depth",
    description=(
        "Number of distinct etymological root nodes reachable from a WakeToken "
        "within 3 HAS_ROOT hops (confidence ≥ 0.7). Measures the depth of the "
        "etymological superposition: a token that can be traced back through "
        "Latin → Proto-Indo-European → Proto-Semitic scores higher than one "
        "with a single English root."
    ),
    cypher_query="""
        MATCH (t:WakeToken {token_id: $token_id})-[r:HAS_ROOT*1..3]->(root:EtymRoot)
        WHERE all(rel IN r WHERE rel.confidence >= 0.7)
        RETURN count(DISTINCT root) AS value
    """,
    return_type="integer",
    expected_node_types=["WakeToken", "EtymRoot"],
    used_by_lens=["etymological", "vico"],
    null_value=0,
)


# ---------------------------------------------------------------------------
# Probe: multilingual_breadth_probe
# ---------------------------------------------------------------------------
# Counts the number of *distinct languages* present in all EtymRoot nodes
# directly attached to a WakeToken (1-hop HAS_ROOT).  This is the primary
# signal for multilingual superposition: a token with roots in 5 languages
# is expected to activate more polysemous directions in the residual stream
# than a monoglot token.
# ---------------------------------------------------------------------------
multilingual_breadth_probe = GraphProbeDefinition(
    name="multilingual_breadth",
    description=(
        "Count of distinct language codes across all direct EtymRoot nodes "
        "of a WakeToken (1-hop HAS_ROOT, any confidence). Core signal for "
        "detecting multilingual superposition in residual-stream activations. "
        "E.g. 'riverrun' scores 4 (en, it, non, la)."
    ),
    cypher_query="""
        MATCH (t:WakeToken {token_id: $token_id})-[:HAS_ROOT]->(r:EtymRoot)
        RETURN count(DISTINCT r.language) AS value
    """,
    return_type="integer",
    expected_node_types=["WakeToken", "EtymRoot"],
    used_by_lens=["etymological"],
    null_value=0,
)


# ---------------------------------------------------------------------------
# Probe: semantic_field_density_probe
# ---------------------------------------------------------------------------
# Counts the number of SemanticField nodes activated by a WakeToken.
# High density indicates a token that simultaneously foregrounds many
# thematic dimensions — a key structural property of Joycean polysemy.
# ---------------------------------------------------------------------------
semantic_field_density_probe = GraphProbeDefinition(
    name="semantic_field_density",
    description=(
        "Count of SemanticField nodes activated by a WakeToken via ACTIVATES "
        "relationships. High density indicates thematic superposition: the token "
        "is simultaneously operating in many semantic registers. Used to rank "
        "tokens as polysemy candidates for SAE feature analysis."
    ),
    cypher_query="""
        MATCH (t:WakeToken {token_id: $token_id})-[:ACTIVATES]->(s:SemanticField)
        RETURN count(s) AS value
    """,
    return_type="integer",
    expected_node_types=["WakeToken", "SemanticField"],
    used_by_lens=["vico", "kabbalah", "etymological", "geometric"],
    null_value=0,
)


# ---------------------------------------------------------------------------
# Probe: vico_cycle_probe
# ---------------------------------------------------------------------------
# Returns the Vico cycle number (1–4, or 0 if unassigned) for a WakeToken.
# The cycle is a categorical label that the model may encode implicitly:
# training a probe to detect Viconian phase from residual-stream activations
# tests whether the LLM has internalised cyclic historical structure from
# its pre-training on Joyce scholarship.
# ---------------------------------------------------------------------------
vico_cycle_probe = GraphProbeDefinition(
    name="vico_cycle",
    description=(
        "Vico historical cycle assigned to the WakeToken: "
        "1=Theocratic, 2=Aristocratic, 3=Democratic, 4=Ricorso, 0=unassigned. "
        "Returned as an integer category. Tests whether LLM residual-stream "
        "directions encode Viconian phase structure implicit in Joyce's text."
    ),
    cypher_query="""
        MATCH (t:WakeToken {token_id: $token_id})-[:BELONGS_TO_CYCLE]->(v:VicoCycle)
        RETURN v.cycle AS value
        UNION ALL
        MATCH (t:WakeToken {token_id: $token_id})
        WHERE NOT (t)-[:BELONGS_TO_CYCLE]->()
        RETURN 0 AS value
        LIMIT 1
    """,
    return_type="integer",
    expected_node_types=["WakeToken", "VicoCycle"],
    used_by_lens=["vico"],
    null_value=0,
)


# ---------------------------------------------------------------------------
# Probe: kabbalah_resonance_probe
# ---------------------------------------------------------------------------
# Returns the name of the Kabbalistic sefirah (or "none") that the
# WakeToken resonates with.  The sefirot form a hierarchy of 10 divine
# emanations: Kether (Crown) → Malkuth (Kingdom).  Tokens grounded in
# material, earthly language (like "Adam") map to Malkuth; tokens
# associated with divine or transcendent meaning map toward Kether.
# This categorical probe tests whether the model encodes ontological
# valence (divine ↔ material axis) as a residual-stream direction.
# ---------------------------------------------------------------------------
kabbalah_resonance_probe = GraphProbeDefinition(
    name="kabbalah_resonance",
    description=(
        "Kabbalistic sefirah name that a WakeToken resonates with via "
        "RESONATES_WITH relationship, or 'none' if unassigned. Returned as "
        "a string category. The 10 sefiroth form a hierarchy from Kether "
        "(Crown; pure transcendence) to Malkuth (Kingdom; material world). "
        "Tests whether LLM activations encode the divine-material valence "
        "axis that structures Joyce's kabbalistic imagery."
    ),
    cypher_query="""
        MATCH (t:WakeToken {token_id: $token_id})-[:RESONATES_WITH]->(k:KabbalahNode)
        RETURN k.sefirah AS value
        UNION ALL
        MATCH (t:WakeToken {token_id: $token_id})
        WHERE NOT (t)-[:RESONATES_WITH]->()
        RETURN 'none' AS value
        LIMIT 1
    """,
    return_type="string",
    expected_node_types=["WakeToken", "KabbalahNode"],
    used_by_lens=["kabbalah"],
    null_value="none",
)


# ---------------------------------------------------------------------------
# Registry: all probes in one place for import convenience
# ---------------------------------------------------------------------------

ALL_PROBES: list[GraphProbeDefinition] = [
    etymology_depth_probe,
    multilingual_breadth_probe,
    semantic_field_density_probe,
    vico_cycle_probe,
    kabbalah_resonance_probe,
]

PROBE_BY_NAME: dict[str, GraphProbeDefinition] = {p.name: p for p in ALL_PROBES}


def get_probe(name: str) -> GraphProbeDefinition:
    """
    Look up a probe by name.

    Parameters
    ----------
    name : str
        The probe name slug (e.g. ``"vico_cycle"``).

    Returns
    -------
    GraphProbeDefinition

    Raises
    ------
    KeyError
        If no probe with the given name is registered.
    """
    if name not in PROBE_BY_NAME:
        available = ", ".join(sorted(PROBE_BY_NAME.keys()))
        raise KeyError(f"No probe named {name!r}. Available probes: {available}")
    return PROBE_BY_NAME[name]
