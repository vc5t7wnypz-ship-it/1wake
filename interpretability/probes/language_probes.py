"""
Language field linear probes for WAKE interpretability.

Each probe is a logistic-regression classifier trained to detect whether a
given residual-stream activation expresses a particular semantic field
(as defined by the McHugh annotation schema and the WAKE knowledge graph).

Spec reference: section 4.2
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# The 19 semantic fields the probes classify against.
# These correspond to the SemanticField nodes in the WAKE Neo4j graph and are
# drawn from McHugh annotations, Joyce scholarship, and the WAKE ontology.
# ---------------------------------------------------------------------------
FIELDS: List[str] = [
    "water",               # rivers, seas, flow — ALP / Anna Livia
    "cyclic_return",       # Vichian ricorso; Eternal Return
    "divine_thunder",      # 100-letter thunderclap words; deity / fear
    "heroic_age",          # Vico's age of heroes; HCE as conquering overlord
    "human_age",           # Vico's democratic age; rational institutions
    "ricorso",             # dissolution / rebeginning; ALP's final monologue
    "vico_cyclic",         # generic Vichian cycle tagging
    "etymology_poetic",    # Vichian poetic etymology; root-sound meaning
    "body",                # HCE / corporeal; giant body as landscape
    "dream",               # Viconian pre-rational consciousness; Freudian id
    "exile",               # Joycean exile; diaspora; wandering
    "guilt_fall",          # HCE's sin; felix culpa; the Fall
    "letter",              # ALP's letter; writing; postal circuit
    "language_plurality",  # multilingual polyglot texture; Babel
    "sexual",              # libido; Freudian Eros; fertility
    "death_rebirth",       # wake as funeral; resurrection; Phoenix
    "family_drama",        # HCE/ALP/Shem/Shaun; Oedipal triangle
    "time",                # Bergsonian durée; cyclical vs. linear time
    "kabbalah",            # Kabbalistic sefiroth; mystical numerology
]

# Number of semantic fields
N_FIELDS: int = len(FIELDS)


# ---------------------------------------------------------------------------
# Dataclass for a single probe prediction
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result from running a probe on one activation vector / token position.

    This dataclass supports two usage modes:

    **Mode A — LanguageFieldProbe output** (fitted sklearn ensemble):
        ``field_scores``, ``top_field``, ``top_score``, ``active_fields``,
        ``superposition_score``, ``entropy``, ``raw_activation`` are populated.
        ``layer``, ``position``, ``token``, ``predictions``, ``confidence``
        are set to defaults.

    **Mode B — raw probe output** (e.g. a single logistic-regression pass or
    a test stub):
        ``layer``, ``position``, ``token``, ``predictions``, ``confidence``,
        ``top_field`` are the primary fields.
        ``field_scores`` mirrors ``predictions``; the remaining fields are
        derived automatically.

    Attributes
    ----------
    field_scores:
        Dict mapping each field name to its probability (0–1).
    top_field:
        Name of the field with the highest score.
    top_score:
        Score of the top field.
    active_fields:
        Fields whose score exceeds 0.5 (the default decision boundary).
    superposition_score:
        Measure of semantic superposition: 1 minus the normalised entropy of
        the score distribution.  A score near 1 means one field dominates
        (collapsed); a score near 0 means many fields are active (superposed).
    entropy:
        Shannon entropy of the field score distribution.  High entropy
        indicates genuine superposition.
    raw_activation:
        The input residual-stream vector (optional, can be None to save memory).
    layer:
        Transformer layer index (Mode B).
    position:
        Token position in the input sequence (Mode B).
    token:
        Surface string of the token at *position* (Mode B).
    predictions:
        Raw prediction dict mapping field name → probability (Mode B).
        Mirrors ``field_scores`` when provided.
    confidence:
        Confidence of the top prediction (Mode B).
    """

    # --- Mode A fields (LanguageFieldProbe) ---
    field_scores: Dict[str, float] = field(default_factory=dict)
    top_field: str = ""
    top_score: float = 0.0
    active_fields: List[str] = field(default_factory=list)
    superposition_score: float = 0.0
    entropy: float = 0.0
    raw_activation: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Mode B fields (direct probe / test stub) ---
    layer: int = 0
    position: int = 0
    token: str = ""
    predictions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Reconcile Mode A / Mode B fields after construction."""
        # If predictions provided but field_scores not, mirror them.
        if self.predictions and not self.field_scores:
            self.field_scores = dict(self.predictions)
        # If field_scores provided but predictions not, mirror them.
        if self.field_scores and not self.predictions:
            self.predictions = dict(self.field_scores)

        # Ensure top_field / top_score are set.
        if self.field_scores and not self.top_field:
            self.top_field = max(self.field_scores, key=self.field_scores.__getitem__)
            self.top_score = self.field_scores[self.top_field]

        if self.field_scores and self.top_score == 0.0:
            self.top_score = self.field_scores.get(self.top_field, 0.0)

        # Ensure confidence is set.
        if self.confidence == 0.0 and self.top_score > 0.0:
            self.confidence = self.top_score

        # Derive active_fields from field_scores if not set.
        if not self.active_fields and self.field_scores:
            self.active_fields = [
                f for f, p in self.field_scores.items() if p > 0.5
            ]

        # Derive entropy / superposition_score if not set.
        if self.entropy == 0.0 and self.field_scores:
            scores_arr = np.array(list(self.field_scores.values()), dtype=float)
            total = scores_arr.sum()
            if total > 0:
                p = scores_arr / total
            else:
                p = np.ones(len(scores_arr)) / max(len(scores_arr), 1)
            eps = 1e-12
            self.entropy = float(-np.sum(p * np.log(p + eps)))
            max_entropy = float(np.log(max(len(scores_arr), 2)))
            self.superposition_score = 1.0 - (self.entropy / (max_entropy + eps))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_superposed(self, threshold: float = 0.5) -> bool:
        """Return True if more than one field exceeds *threshold*."""
        return len(self.active_fields) > 1

    def dominant_field(self) -> str:
        """Alias for :attr:`top_field`."""
        return self.top_field

    def field_vector(self) -> np.ndarray:
        """Return scores as a numpy array aligned with :data:`FIELDS`."""
        return np.array([self.field_scores.get(f, 0.0) for f in FIELDS])


# ---------------------------------------------------------------------------
# LanguageFieldProbe
# ---------------------------------------------------------------------------

class LanguageFieldProbe:
    """Linear probe ensemble for semantic-field classification.

    One binary logistic-regression classifier is trained per field in
    :data:`FIELDS`.  The classifiers share a common :class:`StandardScaler`
    fitted on all training activations.

    Parameters
    ----------
    activation_dim:
        Expected dimensionality of each residual-stream vector.  Used for
        validation only; setting to ``None`` skips the check.
    C:
        Regularisation strength for all logistic-regression probes (default
        ``1.0``).  Lower values = stronger regularisation.
    max_iter:
        Maximum iterations for the logistic-regression solver (default
        ``1000``).
    """

    def __init__(
        self,
        activation_dim: Optional[int] = None,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        self.activation_dim = activation_dim
        self.C = C
        self.max_iter = max_iter

        # One classifier per field
        self._probes: Dict[str, LogisticRegression] = {
            f: LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            for f in FIELDS
        }
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        activations: np.ndarray,
        labels: Dict[str, np.ndarray],
    ) -> "LanguageFieldProbe":
        """Fit all probes.

        Parameters
        ----------
        activations:
            Shape ``[n_samples, d_model]``.  Each row is a residual-stream
            vector extracted at a particular layer and token position.
        labels:
            Dict mapping field name → binary label array of shape
            ``[n_samples]`` (1 = field active at this token, 0 = not active).
            Fields not present in *labels* are skipped.

        Returns
        -------
        LanguageFieldProbe
            ``self`` for method chaining.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"activations must be 2-D [n_samples, d_model], got shape {activations.shape}"
            )
        if self.activation_dim is not None and activations.shape[1] != self.activation_dim:
            raise ValueError(
                f"Expected activation dim {self.activation_dim}, got {activations.shape[1]}"
            )

        # Fit shared scaler
        scaled = self._scaler.fit_transform(activations)

        for f in FIELDS:
            if f not in labels:
                continue
            y = np.asarray(labels[f])
            if y.shape[0] != activations.shape[0]:
                raise ValueError(
                    f"Label array for field '{f}' has {y.shape[0]} samples, "
                    f"expected {activations.shape[0]}"
                )
            # Require at least one positive example
            if y.sum() == 0:
                continue
            self._probes[f].fit(scaled, y)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, activation: np.ndarray) -> ProbeResult:
        """Run all probes on a single activation vector.

        Parameters
        ----------
        activation:
            Shape ``[d_model]`` or ``[1, d_model]``.

        Returns
        -------
        ProbeResult
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        activation = np.asarray(activation, dtype=float)
        if activation.ndim == 1:
            activation = activation.reshape(1, -1)

        scaled = self._scaler.transform(activation)

        field_scores: Dict[str, float] = {}
        for f, probe in self._probes.items():
            # Some probes may not have been fitted (no positive labels)
            try:
                prob = probe.predict_proba(scaled)[0, 1]
            except Exception:
                prob = 0.0
            field_scores[f] = float(prob)

        return self._build_result(field_scores, activation[0])

    def predict_batch(self, activations: np.ndarray) -> List[ProbeResult]:
        """Vectorised prediction for a batch of activations.

        Parameters
        ----------
        activations:
            Shape ``[n_tokens, d_model]``.

        Returns
        -------
        List[ProbeResult]
            One :class:`ProbeResult` per row in *activations*.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_batch().")

        activations = np.asarray(activations, dtype=float)
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        scaled = self._scaler.transform(activations)
        n = scaled.shape[0]

        # Collect probabilities for all fields at once
        all_probs: Dict[str, np.ndarray] = {}
        for f, probe in self._probes.items():
            try:
                all_probs[f] = probe.predict_proba(scaled)[:, 1]
            except Exception:
                all_probs[f] = np.zeros(n)

        results: List[ProbeResult] = []
        for i in range(n):
            field_scores = {f: float(all_probs[f][i]) for f in FIELDS}
            results.append(self._build_result(field_scores, activations[i]))

        return results

    # ------------------------------------------------------------------
    # Probe directions
    # ------------------------------------------------------------------

    def get_probe_direction(self, field: str) -> np.ndarray:
        """Return the weight vector (``coef_``) of the probe for *field*.

        This is the "probe direction" used in superposition decomposition:
        projecting a residual-stream vector onto this direction measures how
        strongly the field is expressed.

        Parameters
        ----------
        field:
            One of the strings in :data:`FIELDS`.

        Returns
        -------
        np.ndarray
            Shape ``[d_model]``.  Raises :class:`ValueError` if *field* is
            unknown or its probe has not been fitted.
        """
        if field not in self._probes:
            raise ValueError(
                f"Unknown field '{field}'.  Valid fields: {FIELDS}"
            )
        probe = self._probes[field]
        if not hasattr(probe, "coef_"):
            raise ValueError(
                f"Probe for field '{field}' has not been fitted yet."
            )
        return probe.coef_[0]  # shape [d_model]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the fitted probes and scaler to *path*.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``probes/layer_12.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "probes": self._probes,
            "scaler": self._scaler,
            "fitted": self._fitted,
            "activation_dim": self.activation_dim,
            "C": self.C,
            "max_iter": self.max_iter,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "LanguageFieldProbe":
        """Load a previously saved probe ensemble from *path*.

        Parameters
        ----------
        path:
            Path to the pickle file written by :meth:`save`.

        Returns
        -------
        LanguageFieldProbe
            A fully fitted instance ready for :meth:`predict`.
        """
        path = Path(path)
        with open(path, "rb") as fh:
            payload: Dict[str, Any] = pickle.load(fh)

        instance = cls(
            activation_dim=payload.get("activation_dim"),
            C=payload.get("C", 1.0),
            max_iter=payload.get("max_iter", 1000),
        )
        instance._probes = payload["probes"]
        instance._scaler = payload["scaler"]
        instance._fitted = payload.get("fitted", True)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        field_scores: Dict[str, float],
        raw_activation: np.ndarray,
    ) -> ProbeResult:
        """Construct a :class:`ProbeResult` from raw scores."""
        scores_arr = np.array([field_scores.get(f, 0.0) for f in FIELDS], dtype=float)

        top_idx = int(np.argmax(scores_arr))
        top_field = FIELDS[top_idx]
        top_score = float(scores_arr[top_idx])

        active_fields = [FIELDS[i] for i, s in enumerate(scores_arr) if s > 0.5]

        # Shannon entropy of score distribution (treat scores as un-normalised probs)
        total = scores_arr.sum()
        if total > 0:
            p = scores_arr / total
        else:
            p = np.ones(N_FIELDS) / N_FIELDS
        eps = 1e-12
        entropy = float(-np.sum(p * np.log(p + eps)))

        # Superposition score: 1 – (entropy / max_entropy).
        # Near 1 = collapsed to single field; near 0 = maximally superposed.
        max_entropy = float(np.log(N_FIELDS))
        superposition_score = 1.0 - (entropy / (max_entropy + eps))

        return ProbeResult(
            field_scores=field_scores,
            top_field=top_field,
            top_score=top_score,
            active_fields=active_fields,
            superposition_score=superposition_score,
            entropy=entropy,
            raw_activation=raw_activation,
        )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"<LanguageFieldProbe fields={N_FIELDS} status={status}>"


# ---------------------------------------------------------------------------
# SuperpositionDetector
# ---------------------------------------------------------------------------

class SuperpositionDetector:
    """Detect and classify superposition states from probe results.

    A token's residual stream is said to be in *superposition* when multiple
    semantic fields are simultaneously active (reflected in high probe scores
    across several fields).  This class analyses a collection of
    :class:`ProbeResult` objects together with the raw residual stream and
    known probe directions to characterise the superposition geometry.

    Class variables
    ---------------
    OPAQUE_THRESHOLD:
        If ALL field scores are below this value the token is classified as
        "opaque": the model has no representational grip on the token at this
        layer.
    """

    OPAQUE_THRESHOLD: float = 0.2

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        probe_results: List[ProbeResult],
        residual_stream: np.ndarray,
        probe_directions: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Analyse superposition across a sequence of probe results.

        Parameters
        ----------
        probe_results:
            One :class:`ProbeResult` per token position.
        residual_stream:
            Shape ``[seq_len, d_model]``.  Raw residual-stream activations at
            the layer where the probes were applied.
        probe_directions:
            Dict mapping field name → weight vector (``[d_model]``).  Used to
            decompose each residual-stream vector into its field components.

        Returns
        -------
        Dict with keys:
            ``n_tokens``            — sequence length
            ``mean_entropy``        — mean probe entropy across tokens
            ``mean_superposition``  — mean superposition score across tokens
            ``superposed_positions``— list of token indices in superposition
            ``opaque_positions``    — list of token indices that are opaque
            ``collapsed_positions`` — list of token indices that are collapsed
            ``field_activations``   — Dict[field, List[float]] — per-token scores
            ``field_coactivation``  — [N_FIELDS × N_FIELDS] co-activation matrix
            ``dominant_fields``     — most common top field per position
            ``decomposition``       — per-token decomposition into probe directions
            ``token_states``        — list of "superposed"|"collapsed"|"opaque"
        """
        n_tokens = len(probe_results)

        # -- Aggregate entropy / superposition scores ----------------------
        entropies = np.array([r.entropy for r in probe_results])
        super_scores = np.array([r.superposition_score for r in probe_results])

        # -- Per-token state classification --------------------------------
        token_states: List[str] = []
        superposed_positions: List[int] = []
        opaque_positions: List[int] = []
        collapsed_positions: List[int] = []

        for i, pr in enumerate(probe_results):
            state = self._classify_single(pr)
            token_states.append(state)
            if state == "superposed":
                superposed_positions.append(i)
            elif state == "opaque":
                opaque_positions.append(i)
            else:
                collapsed_positions.append(i)

        # -- Field activation matrix [n_tokens × N_FIELDS] ----------------
        field_matrix = np.array(
            [[r.field_scores.get(f, 0.0) for f in FIELDS] for r in probe_results]
        )  # [n_tokens, N_FIELDS]

        field_activations: Dict[str, List[float]] = {
            f: field_matrix[:, i].tolist() for i, f in enumerate(FIELDS)
        }

        # -- Co-activation matrix: fields that are active at the same token
        # [N_FIELDS × N_FIELDS] entry (i,j) = fraction of tokens where both
        # fields i and j are simultaneously above 0.5
        binary = (field_matrix > 0.5).astype(float)
        coact = (binary.T @ binary) / (n_tokens + 1e-9)

        # -- Dominant field per position -----------------------------------
        dominant_fields: List[str] = [r.top_field for r in probe_results]

        # -- Residual-stream decomposition into probe directions -----------
        # Project each residual-stream vector onto each probe direction;
        # normalise by direction norm so the coefficient is meaningful.
        decomposition: List[Dict[str, float]] = []
        for i in range(n_tokens):
            vec = residual_stream[i] if i < residual_stream.shape[0] else np.zeros(1)
            token_decomp: Dict[str, float] = {}
            for f, direction in probe_directions.items():
                norm = float(np.linalg.norm(direction)) + 1e-12
                coeff = float(np.dot(vec, direction) / norm)
                token_decomp[f] = coeff
            decomposition.append(token_decomp)

        return {
            "n_tokens": n_tokens,
            "mean_entropy": float(entropies.mean()),
            "mean_superposition": float(super_scores.mean()),
            "superposed_positions": superposed_positions,
            "opaque_positions": opaque_positions,
            "collapsed_positions": collapsed_positions,
            "field_activations": field_activations,
            "field_coactivation": coact.tolist(),
            "dominant_fields": dominant_fields,
            "decomposition": decomposition,
            "token_states": token_states,
        }

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _classify_single(self, pr: ProbeResult) -> str:
        """Classify a single :class:`ProbeResult` as superposed/collapsed/opaque."""
        scores = np.array([pr.field_scores.get(f, 0.0) for f in FIELDS])

        if scores.max() < self.OPAQUE_THRESHOLD:
            return "opaque"

        n_active = int((scores > 0.5).sum())
        if n_active > 1:
            return "superposed"

        return "collapsed"

    def classify(self, analysis: Dict[str, Any]) -> str:
        """Summarise an analysis dict as a single string label.

        Parameters
        ----------
        analysis:
            Output of :meth:`analyze`.

        Returns
        -------
        str
            ``"superposed"`` | ``"collapsed"`` | ``"opaque"``

        Decision rule: the majority state wins; ties go to "superposed".
        """
        states: List[str] = analysis.get("token_states", [])
        if not states:
            return "opaque"

        counts: Dict[str, int] = {
            "superposed": states.count("superposed"),
            "collapsed": states.count("collapsed"),
            "opaque": states.count("opaque"),
        }
        max_count = max(counts.values())
        # Prefer superposed on tie (linguistically more interesting)
        for label in ("superposed", "collapsed", "opaque"):
            if counts[label] == max_count:
                return label
        return "opaque"

    def describe(self, analysis: Dict[str, Any]) -> str:
        """Return a human-readable description of the superposition state.

        Parameters
        ----------
        analysis:
            Output of :meth:`analyze`.

        Returns
        -------
        str
            A multi-sentence prose description suitable for display in a UI or
            notebook.
        """
        n_tokens: int = analysis.get("n_tokens", 0)
        mean_entropy: float = analysis.get("mean_entropy", 0.0)
        mean_super: float = analysis.get("mean_superposition", 0.0)
        superposed: List[int] = analysis.get("superposed_positions", [])
        opaque: List[int] = analysis.get("opaque_positions", [])
        collapsed: List[int] = analysis.get("collapsed_positions", [])
        dominant: List[str] = analysis.get("dominant_fields", [])

        label = self.classify(analysis)

        # Summary sentence
        frac_super = len(superposed) / max(n_tokens, 1)
        frac_opaque = len(opaque) / max(n_tokens, 1)
        frac_collapsed = len(collapsed) / max(n_tokens, 1)

        summary = (
            f"Sequence of {n_tokens} tokens: "
            f"{len(superposed)} superposed ({frac_super:.0%}), "
            f"{len(collapsed)} collapsed ({frac_collapsed:.0%}), "
            f"{len(opaque)} opaque ({frac_opaque:.0%})."
        )

        entropy_note = (
            f"Mean probe entropy is {mean_entropy:.3f} "
            f"(superposition score {mean_super:.3f})."
        )

        if label == "superposed":
            state_note = (
                "The residual stream is predominantly SUPERPOSED: the model "
                "holds multiple semantic fields simultaneously active, "
                "consistent with genuine polysemy or multilingual ambiguity."
            )
        elif label == "collapsed":
            from collections import Counter
            top_field = Counter(dominant).most_common(1)[0][0] if dominant else "unknown"
            state_note = (
                f"The residual stream is predominantly COLLAPSED onto the "
                f"'{top_field}' field: the model has selected a single "
                "interpretive frame, suppressing competing readings."
            )
        else:
            state_note = (
                "The residual stream is predominantly OPAQUE: probe scores "
                "are uniformly low, suggesting the model has not yet formed "
                "a stable semantic representation at this layer."
            )

        # Active fields across the sequence
        all_active: List[str] = []
        for pr_states in analysis.get("dominant_fields", []):
            all_active.append(pr_states)

        from collections import Counter
        field_counter: Counter = Counter(all_active)
        top_fields = field_counter.most_common(3)
        if top_fields:
            top_str = ", ".join(f"'{f}' ({c})" for f, c in top_fields)
            field_note = f"Most frequently dominant fields: {top_str}."
        else:
            field_note = ""

        parts = [summary, entropy_note, state_note]
        if field_note:
            parts.append(field_note)
        return "  ".join(parts)
