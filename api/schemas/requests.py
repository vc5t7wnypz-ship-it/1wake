"""
Pydantic v2 request schemas for the WAKE interpretability API.

All models use strict field validation so that invalid inputs are caught at
the API boundary and return clear 422 errors rather than propagating into the
analysis pipeline.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Default set of lenses to run when the caller does not specify any.
_DEFAULT_LENSES: List[str] = [
    "viconian",
    "psychoanalytic",
    "mythological",
    "linguistic",
]


# ---------------------------------------------------------------------------
# PassageRequest
# ---------------------------------------------------------------------------

class PassageRequest(BaseModel):
    """Request body for the /api/wake/analyse endpoint.

    Attributes
    ----------
    passage:
        The raw Wake text segment to analyse (may be multi-sentence).
    page:
        Page number in Finnegans Wake (1-based, i.e. FW p.1 = page 1).
    line:
        Line number on the given page (1-based).
    lenses:
        Names of the lenses to run.  Defaults to the four standard lenses:
        viconian, psychoanalytic, mythological, linguistic.
    layers_to_capture:
        Optional explicit list of transformer layer indices to capture
        residual-stream activations from.  If None, the model's default
        layer set is used.
    run_anamnesis:
        Whether to run the anamnesis protocol (retrieve related passages,
        nodes, and generate a narrative) alongside the mechanical analysis.
        Defaults to True.
    """

    passage: str = Field(..., min_length=1, description="Raw Wake text segment")
    page: int = Field(..., ge=1, description="Page number in Finnegans Wake (1-based)")
    line: int = Field(..., ge=1, description="Line number on the page (1-based)")
    lenses: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_LENSES),
        description="Lens names to activate.  Defaults to the four standard lenses.",
    )
    layers_to_capture: Optional[List[int]] = Field(
        default=None,
        description=(
            "Specific transformer layer indices to capture activations from. "
            "If None, the model default layer set is used."
        ),
    )
    run_anamnesis: bool = Field(
        default=True,
        description="Whether to run the anamnesis retrieval protocol.",
    )

    @field_validator("lenses")
    @classmethod
    def lenses_must_not_be_empty(cls, v: List[str]) -> List[str]:
        """Require at least one lens."""
        if not v:
            raise ValueError("At least one lens must be specified.")
        return v

    @field_validator("layers_to_capture")
    @classmethod
    def layers_must_be_non_negative(
        cls, v: Optional[List[int]]
    ) -> Optional[List[int]]:
        """All layer indices must be ≥ 0."""
        if v is not None:
            for idx in v:
                if idx < 0:
                    raise ValueError(
                        f"Layer index {idx} is invalid; indices must be ≥ 0."
                    )
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "passage": "riverrun, past Eve and Adam's, from swerve of shore",
            "page": 3,
            "line": 1,
            "lenses": ["viconian", "psychoanalytic"],
            "layers_to_capture": [0, 6, 12, 18, 24],
            "run_anamnesis": True,
        }
    }}


# ---------------------------------------------------------------------------
# GraphNodeRequest
# ---------------------------------------------------------------------------

class GraphNodeRequest(BaseModel):
    """Request body (or query-parameter bundle) for a single node lookup.

    Attributes
    ----------
    surface:
        The surface form of the Wake token to look up (e.g. ``"riverrun"``).
    """

    surface: str = Field(..., min_length=1, description="Token surface form")

    model_config = {"json_schema_extra": {
        "example": {"surface": "riverrun"}
    }}


# ---------------------------------------------------------------------------
# GraphPathRequest
# ---------------------------------------------------------------------------

class GraphPathRequest(BaseModel):
    """Request for a shortest-path query between two graph nodes.

    Attributes
    ----------
    from_surface:
        Surface form of the source Wake token.
    to_surface:
        Surface form of the destination Wake token.
    max_hops:
        Maximum number of relationship hops allowed in the path.
        Must be between 1 and 10 inclusive.  Defaults to 4.
    """

    from_surface: str = Field(..., min_length=1, description="Source token surface")
    to_surface: str = Field(..., min_length=1, description="Destination token surface")
    max_hops: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of hops in the graph path (1–10).",
    )

    @model_validator(mode="after")
    def from_and_to_must_differ(self) -> "GraphPathRequest":
        """Source and destination must not be identical."""
        if self.from_surface == self.to_surface:
            raise ValueError(
                "from_surface and to_surface must be different nodes."
            )
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "from_surface": "riverrun",
            "to_surface": "Howth",
            "max_hops": 4,
        }
    }}


# ---------------------------------------------------------------------------
# LensComposeRequest
# ---------------------------------------------------------------------------

class LensComposeRequest(BaseModel):
    """Request for composing multiple lenses into a weighted blend.

    Attributes
    ----------
    lenses:
        Names of the lenses to compose.  At least two required.
    weights:
        Optional list of non-negative floats, one per lens.  If provided,
        must have the same length as *lenses* and must not all be zero.
        When None, equal weighting is assumed.
    """

    lenses: List[str] = Field(
        ...,
        min_length=2,
        description="Names of the lenses to compose (minimum 2).",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional non-negative weight per lens.  "
            "Must be the same length as *lenses* when provided."
        ),
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "LensComposeRequest":
        """Validate weights if provided."""
        if self.weights is not None:
            if len(self.weights) != len(self.lenses):
                raise ValueError(
                    f"weights has {len(self.weights)} entries but lenses has "
                    f"{len(self.lenses)} entries; they must match."
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("All weights must be non-negative.")
            if sum(self.weights) == 0:
                raise ValueError("At least one weight must be > 0.")
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "lenses": ["viconian", "psychoanalytic", "mythological"],
            "weights": [0.5, 0.3, 0.2],
        }
    }}
