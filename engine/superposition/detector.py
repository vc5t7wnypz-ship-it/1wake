"""
WAKE Engine — SuperpositionDetector
=====================================
Detects and analyses semantic superposition in residual-stream activations:
the co-activation of multiple semantic fields within a single token embedding.

Spec reference: section 4.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SuperpositionResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SuperpositionResult:
    """Full superposition analysis for a single token / position.

    Attributes
    ----------
    token_surface:
        The raw surface string of the token being analysed.
    page:
        Source page number.
    line:
        Source line number.
    position:
        0-based position index in the sequence.
    residual:
        Residual-stream vector (shape ``(d_model,)``).
    active_fields:
        List of ``(field_name, activation_strength)`` pairs for all
        semantic fields with strength above the detection threshold,
        sorted descending by strength.
    dominant_field:
        The field with the highest activation strength, or ``None`` if no
        field clears the threshold.
    superposition_score:
        Scalar in [0.0, 1.0] summarising the degree of co-activation
        across semantic fields.  Higher = more superposed.
    probe_results:
        Raw probe outputs before thresholding: mapping from field name to
        raw activation value.
    cosine_similarities:
        Pairwise cosine similarities between the top-field probe directions,
        indicating whether the active fields are geometrically orthogonal
        (as predicted by the superposition hypothesis).
    classification:
        One of ``"superposed"`` | ``"collapsed"`` | ``"opaque"``.
    confidence:
        Overall confidence in the classification (0.0-1.0).
    notes:
        Optional free-text annotation.
    """

    token_surface: str
    page: int
    line: int
    position: int
    residual: np.ndarray
    active_fields: List[Tuple[str, float]] = field(default_factory=list)
    dominant_field: Optional[str] = None
    superposition_score: float = 0.0
    probe_results: Dict[str, float] = field(default_factory=dict)
    cosine_similarities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    classification: str = "opaque"
    confidence: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "token_surface": self.token_surface,
            "page": self.page,
            "line": self.line,
            "position": self.position,
            "residual": self.residual.tolist(),
            "active_fields": self.active_fields,
            "dominant_field": self.dominant_field,
            "superposition_score": self.superposition_score,
            "probe_results": self.probe_results,
            "cosine_similarities": {
                f"{k[0]}|{k[1]}": v for k, v in self.cosine_similarities.items()
            },
            "classification": self.classification,
            "confidence": self.confidence,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Standalone scoring function
# ---------------------------------------------------------------------------

def compute_superposition_score(
    probe_results: Dict[str, float],
    activation_threshold: float = 0.3,
) -> float:
    """Compute a superposition score from probe activation values.

    The score captures the *breadth* of co-activated fields:

    1. Keep only activations above *activation_threshold*.
    2. Normalise to a probability simplex via softmax.
    3. Compute Shannon entropy of the normalised distribution.
    4. Normalise entropy to [0, 1] by dividing by ``log(N_active)``.

    A score near 1.0 means many fields activate with similar strength
    (maximally superposed); near 0.0 means a single field dominates
    (collapsed).

    Parameters
    ----------
    probe_results:
        Mapping from field name to raw activation value (any real number).
    activation_threshold:
        Minimum activation required for a field to be considered active.

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not probe_results:
        return 0.0

    values = np.array(list(probe_results.values()), dtype=float)
    active_mask = values >= activation_threshold
    active_values = values[active_mask]

    if active_values.size == 0:
        return 0.0
    if active_values.size == 1:
        return 0.0

    # Softmax normalisation (numerically stable)
    shifted = active_values - active_values.max()
    exp_vals = np.exp(shifted)
    probs = exp_vals / exp_vals.sum()

    # Shannon entropy, normalised by max possible entropy log(N)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = math.log(float(active_values.size))

    if max_entropy <= 0.0:
        return 0.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SuperpositionDetector
# ---------------------------------------------------------------------------

class SuperpositionDetector:
    """Detect semantic superposition in residual-stream activations.

    Parameters
    ----------
    probe_directions:
        Dictionary mapping semantic field names to unit-norm probe direction
        vectors (numpy arrays of shape ``(d_model,)``).  These are typically
        the weight vectors of linear probes trained to detect each field.
    activation_threshold:
        Minimum cosine similarity (or linear probe activation) required for
        a field to be considered active (default 0.3).
    superposition_threshold:
        Minimum superposition score required to classify a token as
        ``"superposed"`` (default 0.4).
    collapse_threshold:
        Maximum superposition score below which a token is classified as
        ``"collapsed"`` (default 0.15).
    """

    def __init__(
        self,
        probe_directions: Optional[Dict[str, np.ndarray]] = None,
        activation_threshold: float = 0.3,
        superposition_threshold: float = 0.4,
        collapse_threshold: float = 0.15,
    ) -> None:
        self.probe_directions: Dict[str, np.ndarray] = probe_directions or {}
        self.activation_threshold = activation_threshold
        self.superposition_threshold = superposition_threshold
        self.collapse_threshold = collapse_threshold

        # Pre-normalise probe directions for efficient cosine similarity
        self._unit_probes: Dict[str, np.ndarray] = {}
        for name, direction in self.probe_directions.items():
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                self._unit_probes[name] = direction / norm
            else:
                self._unit_probes[name] = direction.copy()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        token_surface: str,
        residual: np.ndarray,
        page: int = 0,
        line: int = 0,
        position: int = 0,
        probe_results: Optional[Dict[str, float]] = None,
    ) -> SuperpositionResult:
        """Analyse superposition for a single token.

        Parameters
        ----------
        token_surface:
            Surface string of the token.
        residual:
            Residual-stream vector at the position of interest, shape
            ``(d_model,)``.
        page, line, position:
            Provenance metadata.
        probe_results:
            Pre-computed probe activations.  If *None* and
            :attr:`probe_directions` are set, activations are computed as
            cosine similarities between the residual and each probe vector.

        Returns
        -------
        :class:`SuperpositionResult`
        """
        # --- compute probe activations ---
        if probe_results is None:
            probe_results = self._compute_probe_activations(residual)

        # --- threshold and rank active fields ---
        active_fields: List[Tuple[str, float]] = [
            (name, float(val))
            for name, val in probe_results.items()
            if float(val) >= self.activation_threshold
        ]
        active_fields.sort(key=lambda x: x[1], reverse=True)

        dominant_field: Optional[str] = (
            active_fields[0][0] if active_fields else None
        )

        # --- superposition score ---
        score = compute_superposition_score(
            probe_results, self.activation_threshold
        )

        # --- pairwise cosine similarities between top probe directions ---
        cos_sims = self._pairwise_cosine_similarities(
            [name for name, _ in active_fields[:6]]
        )

        result = SuperpositionResult(
            token_surface=token_surface,
            page=page,
            line=line,
            position=position,
            residual=residual.copy(),
            active_fields=active_fields,
            dominant_field=dominant_field,
            superposition_score=score,
            probe_results=probe_results,
            cosine_similarities=cos_sims,
        )
        result.classification = self.classify(result)
        result.confidence = self._estimate_confidence(result)
        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def batch_analyze(
        self,
        tokens: List[Any],
        residuals: np.ndarray,
        probe_results_batch: Optional[List[Dict[str, float]]] = None,
    ) -> List[SuperpositionResult]:
        """Analyse superposition for a batch of tokens efficiently.

        Parameters
        ----------
        tokens:
            List of token objects.  Each should have ``.surface``, ``.page``,
            ``.line``, and ``.position`` attributes (compatible with
            :class:`engine.tokenizer.wake_tokenizer.WakeToken`).  Plain
            strings are also accepted.
        residuals:
            Residual-stream matrix of shape ``(n_tokens, d_model)``.
        probe_results_batch:
            Optional list of pre-computed probe-result dicts, one per token.
            If *None* and probe directions are set, cosine similarities are
            computed in a single batch matrix multiplication.

        Returns
        -------
        List of :class:`SuperpositionResult`, one per input token.
        """
        n = len(tokens)
        if residuals.shape[0] != n:
            raise ValueError(
                f"batch_analyze: expected {n} residual vectors, "
                f"got {residuals.shape[0]}"
            )

        # Batch-compute probe activations when not pre-supplied
        if probe_results_batch is None:
            probe_results_batch = self._compute_probe_activations_batch(residuals)

        results: List[SuperpositionResult] = []
        for i, (tok, residual) in enumerate(zip(tokens, residuals)):
            surface: str = getattr(tok, "surface", str(tok))
            page: int = getattr(tok, "page", 0)
            line: int = getattr(tok, "line", 0)
            position: int = getattr(tok, "position", i)
            pr: Optional[Dict[str, float]] = (
                probe_results_batch[i] if probe_results_batch else None
            )
            results.append(
                self.analyse(
                    token_surface=surface,
                    residual=residual,
                    page=page,
                    line=line,
                    position=position,
                    probe_results=pr,
                )
            )
        return results

    # Alias to match either British or American spelling
    batch_analyse = batch_analyze

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, result: SuperpositionResult) -> str:
        """Classify a :class:`SuperpositionResult`.

        Returns
        -------
        ``"superposed"``
            Multiple semantic fields co-activate above threshold AND
            the superposition score exceeds :attr:`superposition_threshold`.
        ``"collapsed"``
            One dominant field clearly dominates; superposition score is
            below :attr:`collapse_threshold`.
        ``"opaque"``
            Neither dominant field nor superposition pattern is clear:
            no field activates above threshold, or confidence is ambiguous.
        """
        has_dominant = result.dominant_field is not None
        n_active = len(result.active_fields)
        score = result.superposition_score

        if n_active == 0:
            return "opaque"

        if score >= self.superposition_threshold and n_active > 1:
            return "superposed"

        if score <= self.collapse_threshold and has_dominant:
            return "collapsed"

        if n_active == 1 and has_dominant:
            return "collapsed"

        # Low confidence across all fields — neither superposed nor cleanly collapsed
        return "opaque"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_probe_activations(
        self,
        residual: np.ndarray,
    ) -> Dict[str, float]:
        """Compute cosine similarity of *residual* against each probe direction."""
        if not self._unit_probes:
            return {}

        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < 1e-8:
            return {name: 0.0 for name in self._unit_probes}

        unit_residual = residual / residual_norm
        return {
            name: float(np.dot(unit_residual, probe))
            for name, probe in self._unit_probes.items()
        }

    def _compute_probe_activations_batch(
        self,
        residuals: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Batch-compute cosine similarities for many residual vectors.

        Parameters
        ----------
        residuals:
            Array of shape ``(n_tokens, d_model)``.

        Returns
        -------
        List of probe-result dicts, one per row.
        """
        if not self._unit_probes:
            return [{} for _ in range(residuals.shape[0])]

        # Stack probe directions: shape (n_fields, d_model)
        field_names = list(self._unit_probes.keys())
        probe_matrix = np.stack(
            [self._unit_probes[n] for n in field_names], axis=0
        )  # (n_fields, d_model)

        # Row-normalise residuals
        norms = np.linalg.norm(residuals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        unit_residuals = residuals / norms  # (n_tokens, d_model)

        # Batch dot product → (n_tokens, n_fields)
        activations = unit_residuals @ probe_matrix.T

        batch_results: List[Dict[str, float]] = []
        for row in activations:
            batch_results.append(
                {name: float(val) for name, val in zip(field_names, row)}
            )
        return batch_results

    def _pairwise_cosine_similarities(
        self,
        field_names: Sequence[str],
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise cosine similarities between named probe directions."""
        sims: Dict[Tuple[str, str], float] = {}
        valid = [n for n in field_names if n in self._unit_probes]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a = self._unit_probes[valid[i]]
                b = self._unit_probes[valid[j]]
                sims[(valid[i], valid[j])] = float(np.dot(a, b))
        return sims

    def _estimate_confidence(self, result: SuperpositionResult) -> float:
        """Heuristic confidence estimate for the classification.

        Confidence is higher when:
        - Active field activations are well above the threshold.
        - The classification is not ``"opaque"``.
        - Probe directions are available for most detected fields.
        """
        if result.classification == "opaque":
            return 0.3

        if not result.active_fields:
            return 0.2

        strengths = [s for _, s in result.active_fields]
        mean_strength = float(np.mean(strengths))

        probe_coverage = (
            len(self._unit_probes) / max(1, len(result.probe_results))
            if result.probe_results
            else 0.5
        )
        probe_coverage = min(probe_coverage, 1.0)

        base_conf = min(
            mean_strength / (self.activation_threshold + 1e-8), 1.0
        )
        return float(np.clip(base_conf * probe_coverage, 0.0, 1.0))
