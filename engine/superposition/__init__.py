"""engine.superposition — Superposition detection for WAKE."""

from engine.superposition.detector import (
    SuperpositionDetector,
    SuperpositionResult,
    compute_superposition_score,
)

__all__ = [
    "SuperpositionDetector",
    "SuperpositionResult",
    "compute_superposition_score",
]
