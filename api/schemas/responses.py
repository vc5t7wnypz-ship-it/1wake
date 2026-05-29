"""
Pydantic v2 response schemas for the WAKE interpretability API.

These models define the exact JSON structure returned by every endpoint.
They are used both for serialisation and for OpenAPI documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# TokenPoint
# ---------------------------------------------------------------------------

class TokenPoint(BaseModel):
    """Interpretability snapshot for a single token position.

    Attributes
    ----------
    position:
        Zero-based token index within the passage.
    token:
        Surface string of the token (as tokenised by the model).
    entropy:
        Shannon entropy of the probe score distribution at this position.
        High entropy → genuine semantic superposition.
    superposition_score:
        1 − (entropy / max_entropy).  Near 1 = collapsed; near 0 = superposed.
    active_fields:
        List of semantic-field names whose probe score exceeds 0.5.
    top_field:
        Name of the semantic field with the highest probe score.
    """

    position: int = Field(..., description="Zero-based token index")
    token: str = Field(..., description="Token surface string")
    entropy: float = Field(..., description="Shannon entropy of probe distribution")
    superposition_score: float = Field(
        ..., description="1 − entropy/max_entropy; near 1 = collapsed, near 0 = superposed"
    )
    active_fields: List[str] = Field(
        default_factory=list,
        description="Fields with probe score > 0.5",
    )
    top_field: str = Field(..., description="Highest-scoring semantic field")

    model_config = {"json_schema_extra": {
        "example": {
            "position": 0,
            "token": "riverrun",
            "entropy": 2.78,
            "superposition_score": 0.12,
            "active_fields": ["water", "cyclic_return", "ricorso"],
            "top_field": "water",
        }
    }}


# ---------------------------------------------------------------------------
# AnamnesisResponse
# ---------------------------------------------------------------------------

class AnamnesisResponse(BaseModel):
    """Result of the anamnesis retrieval protocol.

    The anamnesis protocol searches the WAKE knowledge graph and corpus for
    passages, nodes, and conceptual connections that resonate with the
    current passage, then synthesises them into a narrative gloss.

    Attributes
    ----------
    retrieved_lenses:
        List of ``(lens_name, similarity_score)`` pairs, ranking which lenses
        from past analyses are most similar to the current passage.
    retrieved_nodes:
        List of graph-node dicts (WakeToken, SemanticField, EtymRoot, etc.)
        retrieved as contextually relevant.
    novel_connections:
        List of connection dicts describing relationships that are newly
        activated by this passage and were not prominent in prior analyses.
    narrative:
        A generated prose gloss summarising the anamnesis findings.
    """

    retrieved_lenses: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="(lens_name, similarity_score) pairs ranked by relevance",
    )
    retrieved_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant graph nodes retrieved from the knowledge graph",
    )
    novel_connections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Newly activated connections not prominent in prior analyses",
    )
    narrative: str = Field(
        default="",
        description="Generated prose gloss summarising anamnesis findings",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "retrieved_lenses": [("viconian", 0.91), ("mythological", 0.78)],
            "retrieved_nodes": [
                {"label": "WakeToken", "surface": "riverrun", "page": 3, "line": 1}
            ],
            "novel_connections": [
                {
                    "from": "riverrun",
                    "to": "Liffey",
                    "relation": "EVOKES",
                    "strength": 0.85,
                }
            ],
            "narrative": (
                "The opening word 'riverrun' initiates the Vichian ricorso, "
                "echoing ALP's return to the sea."
            ),
        }
    }}


# ---------------------------------------------------------------------------
# ContrastiveResponse
# ---------------------------------------------------------------------------

class ContrastiveResponse(BaseModel):
    """Full response for the contrastive lens-analysis endpoint.

    Attributes
    ----------
    passage:
        The raw passage that was analysed.
    page:
        Page number in Finnegans Wake.
    line:
        Line number on the page.
    tokens_by_lens:
        Dict mapping each lens name to the list of :class:`TokenPoint`
        objects for that lens's run.
    divergence_map:
        ``[seq_len × seq_len]`` attention-divergence matrix: entry ``[i][j]``
        is the mean symmetric KL divergence between the two most divergent
        lenses at attention positions (i, j).
    agreement_matrix:
        ``[n_lenses × n_lenses]`` matrix: entry ``[i][j]`` is the mean cosine
        similarity between the residual-stream activations of lens i and
        lens j, averaged over all token positions and captured layers.
    superposition_positions:
        List of token indices where the model is in superposition (holding
        multiple semantic fields simultaneously).
    lens_names:
        Ordered list of lens names (mirrors the keys of *tokens_by_lens*).
    anamnesis:
        Optional :class:`AnamnesisResponse` if the anamnesis protocol was run.
    """

    passage: str = Field(..., description="Raw Wake passage")
    page: int = Field(..., description="Page number")
    line: int = Field(..., description="Line number")
    tokens_by_lens: Dict[str, List[TokenPoint]] = Field(
        default_factory=dict,
        description="Per-lens token analysis results",
    )
    divergence_map: List[List[float]] = Field(
        default_factory=list,
        description="Attention-divergence matrix [seq × seq]",
    )
    agreement_matrix: List[List[float]] = Field(
        default_factory=list,
        description="Lens-agreement matrix [n_lenses × n_lenses]",
    )
    superposition_positions: List[int] = Field(
        default_factory=list,
        description="Token indices where superposition is detected",
    )
    lens_names: List[str] = Field(
        default_factory=list,
        description="Ordered list of lens names in the analysis",
    )
    anamnesis: Optional[AnamnesisResponse] = Field(
        default=None,
        description="Anamnesis retrieval result (if requested)",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "passage": "riverrun, past Eve and Adam's",
            "page": 3,
            "line": 1,
            "tokens_by_lens": {
                "viconian": [
                    {
                        "position": 0,
                        "token": "riverrun",
                        "entropy": 2.78,
                        "superposition_score": 0.12,
                        "active_fields": ["water", "cyclic_return"],
                        "top_field": "water",
                    }
                ]
            },
            "divergence_map": [[0.0, 0.12], [0.12, 0.0]],
            "agreement_matrix": [[1.0, 0.74], [0.74, 1.0]],
            "superposition_positions": [0, 3],
            "lens_names": ["viconian", "psychoanalytic"],
            "anamnesis": None,
        }
    }}


# ---------------------------------------------------------------------------
# GraphNodeResponse
# ---------------------------------------------------------------------------

class GraphNodeResponse(BaseModel):
    """Response for a single Wake-token graph node lookup.

    Attributes
    ----------
    surface:
        The surface form of the token.
    page:
        Page where this token appears in Finnegans Wake.
    line:
        Line on that page.
    roots:
        List of etymological-root dicts, each with keys
        ``language``, ``form``, ``gloss``, ``confidence``.
    semantic_fields:
        List of semantic-field dicts, each with at least ``name`` and
        optionally ``vico_cycle``.
    connections:
        List of relationship dicts describing outgoing edges from this node,
        each with keys ``type``, ``target``, ``properties``.
    """

    surface: str = Field(..., description="Token surface form")
    page: int = Field(..., description="Page in Finnegans Wake")
    line: int = Field(..., description="Line on the page")
    roots: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Etymological roots for this token",
    )
    semantic_fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Semantic fields activated by this token",
    )
    connections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Outgoing graph relationships from this token node",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "surface": "riverrun",
            "page": 3,
            "line": 1,
            "roots": [
                {
                    "language": "en",
                    "form": "river",
                    "gloss": "a natural watercourse",
                    "confidence": 1.0,
                }
            ],
            "semantic_fields": [
                {"name": "water", "vico_cycle": 4},
                {"name": "cyclic_return", "vico_cycle": 4},
            ],
            "connections": [
                {
                    "type": "ACTIVATES",
                    "target": {"label": "SemanticField", "name": "water"},
                    "properties": {},
                }
            ],
        }
    }}


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health-check response.

    Attributes
    ----------
    status:
        ``"ok"`` if all critical subsystems are healthy; ``"degraded"`` if
        some optional components are unavailable; ``"error"`` if a critical
        component has failed.
    model_loaded:
        True if the HuggingFace / TransformerLens model is loaded in memory.
    graph_connected:
        True if the Neo4j graph database connection is active.
    lenses_available:
        List of lens names that have been successfully initialised and are
        ready to use.
    """

    status: str = Field(..., description="'ok' | 'degraded' | 'error'")
    model_loaded: bool = Field(..., description="Whether the LLM model is loaded")
    graph_connected: bool = Field(..., description="Whether Neo4j is reachable")
    lenses_available: List[str] = Field(
        default_factory=list,
        description="Names of successfully initialised lenses",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "status": "ok",
            "model_loaded": True,
            "graph_connected": True,
            "lenses_available": ["viconian", "psychoanalytic", "mythological", "linguistic"],
        }
    }}
