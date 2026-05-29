"""
WAKE Engine — AnamnesisProtocol
=================================
Anamnesis (Greek: ἀνάμνησις, "reminiscence") is the process of recovering
latent meaning that was always implicit in a token's residual stream but
not yet consciously articulated.

This module implements the AnamnesisProtocol: given a token, its activation
cache, and a set of active lenses, it:

1. Retrieves the residual stream at every layer for the token's position.
2. Probes all registered semantic fields.
3. Identifies the top-k lenses (by total field activation).
4. Finds novel cross-field graph paths that are NOT foregrounded by the
   dominant lenses, surfacing unexpected etymological or symbolic connections.
5. Generates a structured narrative (the "anamnestic trace").

Spec reference: section 6
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# AnamnesisResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnamnesisResult:
    """Full anamnesis trace for a single token position.

    Attributes
    ----------
    token_surface:
        The surface string of the token.
    page, line, position:
        Provenance metadata.
    layer_residuals:
        Dictionary mapping layer index to residual-stream numpy array
        (shape ``(d_model,)``).
    field_activations:
        Mapping from semantic field name to activation strength at the
        final layer (or a designated analysis layer).
    top_lenses:
        Ordered list of ``(lens_id, total_activation)`` pairs for the
        three highest-activating lenses.
    novel_connections:
        List of cross-field paths found by the graph analysis that are NOT
        foregrounded by the top-3 lenses.  Each path is a list of node ID
        strings representing a traversal in the knowledge graph.
    narrative:
        Human-readable anamnestic narrative string.
    confidence:
        Overall confidence in the analysis (0.0-1.0).
    metadata:
        Any additional key-value metadata.
    """

    token_surface: str
    page: int
    line: int
    position: int
    layer_residuals: Dict[int, np.ndarray] = field(default_factory=dict)
    field_activations: Dict[str, float] = field(default_factory=dict)
    top_lenses: List[Tuple[str, float]] = field(default_factory=list)
    novel_connections: List[List[str]] = field(default_factory=list)
    narrative: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "token_surface": self.token_surface,
            "page": self.page,
            "line": self.line,
            "position": self.position,
            "layer_residuals": {
                str(k): v.tolist() for k, v in self.layer_residuals.items()
            },
            "field_activations": self.field_activations,
            "top_lenses": self.top_lenses,
            "novel_connections": self.novel_connections,
            "narrative": self.narrative,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AnamnesisProtocol
# ---------------------------------------------------------------------------

class AnamnesisProtocol:
    """Recover latent meaning from a token's residual stream.

    Parameters
    ----------
    probe_directions:
        Dict mapping semantic field names to probe direction vectors
        (shape ``(d_model,)``).
    lens_registry:
        Dict mapping lens IDs to lens definition dicts.  Each definition
        should have at minimum:
            ``{"name": str, "foregrounds": [field_name, ...], "type": str}``
    graph:
        Optional graph backend (e.g. Neo4j driver wrapper).  When *None*
        novel connections are synthesised from the probe directions alone
        using cosine-orthogonality as a proxy for cross-field novelty.
    top_k_lenses:
        Number of dominant lenses to consider when searching for novel
        connections (default 3).
    activation_threshold:
        Minimum probe activation for a field to be considered active
        (default 0.2).
    """

    def __init__(
        self,
        probe_directions: Optional[Dict[str, np.ndarray]] = None,
        lens_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        graph: Optional[Any] = None,
        top_k_lenses: int = 3,
        activation_threshold: float = 0.2,
    ) -> None:
        self.probe_directions: Dict[str, np.ndarray] = probe_directions or {}
        self.lens_registry: Dict[str, Dict[str, Any]] = lens_registry or {}
        self.graph = graph
        self.top_k_lenses = top_k_lenses
        self.activation_threshold = activation_threshold

        # Pre-normalise probe directions
        self._unit_probes: Dict[str, np.ndarray] = {}
        for name, direction in self.probe_directions.items():
            norm = float(np.linalg.norm(direction))
            if norm > 1e-8:
                self._unit_probes[name] = direction / norm
            else:
                self._unit_probes[name] = direction.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        token_surface: str,
        layer_residuals: Dict[int, np.ndarray],
        page: int = 0,
        line: int = 0,
        position: int = 0,
        analysis_layer: Optional[int] = None,
    ) -> AnamnesisResult:
        """Run the full anamnesis protocol for a single token.

        Parameters
        ----------
        token_surface:
            Raw surface text of the token.
        layer_residuals:
            Dict mapping layer index → residual-stream vector (d_model,).
        page, line, position:
            Provenance metadata.
        analysis_layer:
            Which layer's residual to use for field probing.  If *None*,
            the highest layer key is used (i.e. the final residual stream).

        Returns
        -------
        :class:`AnamnesisResult`
        """
        # --- select analysis residual ---
        if not layer_residuals:
            # No activations available; return minimal result
            return AnamnesisResult(
                token_surface=token_surface,
                page=page,
                line=line,
                position=position,
            )

        layer = (
            analysis_layer
            if analysis_layer is not None
            else max(layer_residuals.keys())
        )
        if layer not in layer_residuals:
            layer = max(layer_residuals.keys())
        residual = layer_residuals[layer]

        # --- probe all semantic fields ---
        field_activations = self._probe_fields(residual)

        # --- identify top-k lenses ---
        top_lenses = self._rank_lenses(field_activations)

        # --- find novel cross-field connections ---
        novel_connections = self._find_novel_connections(
            field_activations=field_activations,
            top_lenses=top_lenses,
        )

        # --- generate narrative ---
        narrative = self._generate_narrative(
            token_surface=token_surface,
            field_activations=field_activations,
            top_lenses=top_lenses,
            novel_connections=novel_connections,
        )

        # --- confidence estimate ---
        confidence = self._estimate_confidence(
            field_activations=field_activations,
            top_lenses=top_lenses,
            novel_connections=novel_connections,
        )

        return AnamnesisResult(
            token_surface=token_surface,
            page=page,
            line=line,
            position=position,
            layer_residuals=layer_residuals,
            field_activations=field_activations,
            top_lenses=top_lenses,
            novel_connections=novel_connections,
            narrative=narrative,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def save_result(result: AnamnesisResult, path: Path) -> None:
        """Save an :class:`AnamnesisResult` to a JSON file.

        Numpy arrays are converted to lists via ``.tolist()`` so that the
        output is natively JSON-serialisable.

        Parameters
        ----------
        result:
            The result to serialise.
        path:
            Destination file path.  Parent directories must already exist.
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)

    @staticmethod
    def load_result(path: Path) -> AnamnesisResult:
        """Load an :class:`AnamnesisResult` from a JSON file.

        Parameters
        ----------
        path:
            Source file path produced by :meth:`save_result`.

        Returns
        -------
        :class:`AnamnesisResult` with numpy arrays reconstructed from lists.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        # Reconstruct numpy arrays from serialised lists
        layer_residuals: Dict[int, np.ndarray] = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in data.get("layer_residuals", {}).items()
        }

        # Reconstruct top_lenses list of tuples (json stores them as lists)
        top_lenses: List[Tuple[str, float]] = [
            (item[0], float(item[1]))
            for item in data.get("top_lenses", [])
        ]

        return AnamnesisResult(
            token_surface=data.get("token_surface", ""),
            page=int(data.get("page", 0)),
            line=int(data.get("line", 0)),
            position=int(data.get("position", 0)),
            layer_residuals=layer_residuals,
            field_activations=dict(data.get("field_activations", {})),
            top_lenses=top_lenses,
            novel_connections=data.get("novel_connections", []),
            narrative=data.get("narrative", ""),
            confidence=float(data.get("confidence", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _probe_fields(self, residual: np.ndarray) -> Dict[str, float]:
        """Compute cosine similarity activations for all probe directions."""
        if not self._unit_probes:
            return {}

        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < 1e-8:
            return {name: 0.0 for name in self._unit_probes}

        unit_res = residual / residual_norm
        activations: Dict[str, float] = {}
        for name, probe in self._unit_probes.items():
            activations[name] = float(np.dot(unit_res, probe))
        return activations

    def _rank_lenses(
        self,
        field_activations: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """Rank lenses by their total activation over their foregrounded fields.

        For each lens in :attr:`lens_registry`, sum the field activations
        for all fields in ``lens["foregrounds"]``.  Return the top-k
        ``(lens_id, total_activation)`` pairs sorted descending.

        Parameters
        ----------
        field_activations:
            Mapping from field name to activation strength.

        Returns
        -------
        List of ``(lens_id, score)`` tuples, length ≤ :attr:`top_k_lenses`.
        """
        lens_scores: List[Tuple[str, float]] = []
        for lens_id, lens_def in self.lens_registry.items():
            foregrounds: List[str] = lens_def.get("foregrounds", [])
            total = sum(
                field_activations.get(f, 0.0)
                for f in foregrounds
                if field_activations.get(f, 0.0) >= self.activation_threshold
            )
            lens_scores.append((lens_id, total))

        lens_scores.sort(key=lambda x: x[1], reverse=True)
        return lens_scores[: self.top_k_lenses]

    def _find_novel_connections(
        self,
        field_activations: Dict[str, float],
        top_lenses: List[Tuple[str, float]],
    ) -> List[List[str]]:
        """Find graph paths that cross fields NOT in the top-3 lens definitions.

        Strategy
        --------
        1. Collect all field names foregrounded by the top-k lenses.
        2. Identify fields that are *active* (above threshold) but NOT in
           the foregrounded set — these are the "unexpected" activations.
        3. For each pair of unexpected fields that both activate, attempt to
           find a path connecting them.  If a graph backend is available, a
           real Cypher query is issued.  Otherwise, we synthesise virtual paths
           from the probe-direction geometry: fields whose probe vectors are
           nearly orthogonal but both active represent genuinely distinct
           semantic dimensions resonating simultaneously.

        Returns
        -------
        List of paths.  Each path is a list of string node IDs (or
        human-readable labels).
        """
        # --- collect foregrounded fields from top lenses ---
        foregrounded: set[str] = set()
        for lens_id, _ in top_lenses:
            if lens_id in self.lens_registry:
                lens_def = self.lens_registry[lens_id]
                foregrounded.update(lens_def.get("foregrounds", []))

        # --- active but unexpected fields ---
        active_unexpected: List[str] = [
            fname
            for fname, strength in field_activations.items()
            if strength >= self.activation_threshold and fname not in foregrounded
        ]

        if len(active_unexpected) < 2:
            return []

        novel_paths: List[List[str]] = []

        if self.graph is not None:
            # Real graph traversal: find paths between pairs of unexpected nodes
            for i in range(len(active_unexpected)):
                for j in range(i + 1, len(active_unexpected)):
                    field_a = active_unexpected[i]
                    field_b = active_unexpected[j]
                    path = self._graph_path(field_a, field_b)
                    if path:
                        novel_paths.append(path)
        else:
            # Geometry-based surrogate: fields with low cosine similarity
            # between their probe directions represent orthogonal semantic
            # dimensions — the most surprising co-activations.
            for i in range(len(active_unexpected)):
                for j in range(i + 1, len(active_unexpected)):
                    field_a = active_unexpected[i]
                    field_b = active_unexpected[j]
                    sim = self._probe_cosine(field_a, field_b)
                    # Orthogonal fields (|cos| < 0.3) that both activate are novel
                    if abs(sim) < 0.3:
                        path = [
                            f"field:{field_a}",
                            f"semantic-bridge:{field_a}↔{field_b}",
                            f"field:{field_b}",
                        ]
                        novel_paths.append(path)

        return novel_paths

    def _generate_narrative(
        self,
        token_surface: str,
        field_activations: Dict[str, float],
        top_lenses: List[Tuple[str, float]],
        novel_connections: List[List[str]],
    ) -> str:
        """Generate a structured narrative describing the anamnetic trace.

        The narrative is formatted in three sections:
        - **Token**: what the token is, which semantic fields it activates.
        - **Dominant lenses**: the interpretive frameworks that foreground
          the token's primary meanings.
        - **Novel connections**: cross-field resonances not covered by the
          dominant lenses.

        Parameters
        ----------
        token_surface:
            The token being analysed.
        field_activations:
            Full field-activation mapping.
        top_lenses:
            Ranked list of ``(lens_id, score)`` pairs.
        novel_connections:
            List of cross-field paths.

        Returns
        -------
        Multi-paragraph narrative string.
        """
        # --- active fields section ---
        active = sorted(
            [
                (name, strength)
                for name, strength in field_activations.items()
                if strength >= self.activation_threshold
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        if active:
            field_list = ", ".join(
                f"{name} ({strength:.2f})" for name, strength in active[:6]
            )
        else:
            field_list = "(none above threshold)"

        narrative_parts: List[str] = []

        # Section 1: Token
        narrative_parts.append(
            f"Token: «{token_surface}»\n"
            f"Active semantic fields: {field_list}"
        )

        # Section 2: Dominant lenses
        if top_lenses:
            lens_descriptions: List[str] = []
            for lens_id, score in top_lenses:
                lens_def = self.lens_registry.get(lens_id, {})
                lens_name = lens_def.get("name", lens_id)
                lens_type = lens_def.get("type", "unknown")
                foregrounds = lens_def.get("foregrounds", [])
                lens_descriptions.append(
                    f"  • {lens_name} (type={lens_type}, score={score:.3f},"
                    f" foregrounds={foregrounds})"
                )
            narrative_parts.append(
                "Dominant lenses:\n" + "\n".join(lens_descriptions)
            )
        else:
            narrative_parts.append("Dominant lenses: none registered")

        # Section 3: Novel connections
        if novel_connections:
            path_strs: List[str] = []
            for path in novel_connections[:5]:
                path_strs.append("  • " + " → ".join(path))
            narrative_parts.append(
                "Novel cross-field connections (not foregrounded by dominant lenses):\n"
                + "\n".join(path_strs)
            )
        else:
            narrative_parts.append(
                "Novel cross-field connections: none detected"
            )

        return "\n\n".join(narrative_parts)

    def _estimate_confidence(
        self,
        field_activations: Dict[str, float],
        top_lenses: List[Tuple[str, float]],
        novel_connections: List[List[str]],
    ) -> float:
        """Heuristic confidence estimate for the full anamnesis result.

        Higher confidence when:
        - Probe directions are registered (we can actually probe).
        - Several fields activate above threshold.
        - The lens registry is populated.
        """
        if not self._unit_probes:
            return 0.25  # No probe directions: low confidence

        n_active = sum(
            1 for s in field_activations.values() if s >= self.activation_threshold
        )
        active_ratio = min(n_active / max(1, len(self._unit_probes)), 1.0)

        lens_bonus = 0.2 if top_lenses and top_lenses[0][1] > 0.0 else 0.0
        novel_bonus = 0.1 if novel_connections else 0.0

        return float(
            np.clip(0.4 + 0.3 * active_ratio + lens_bonus + novel_bonus, 0.0, 1.0)
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _graph_path(
        self,
        field_a: str,
        field_b: str,
        max_hops: int = 4,
    ) -> Optional[List[str]]:
        """Query the graph for a shortest path between two semantic fields.

        Returns the path as a list of node labels, or ``None`` if no path
        is found within *max_hops*.
        """
        if self.graph is None:
            return None
        try:
            query = (
                "MATCH p = shortestPath("
                "  (a:SemanticField {name: $field_a})"
                "  -[*..{max_hops}]-"
                "  (b:SemanticField {name: $field_b})"
                ") RETURN [n IN nodes(p) | coalesce(n.name, n.token_id)] AS path "
                "LIMIT 1"
            ).replace("{max_hops}", str(max_hops))
            results = list(self.graph.run(query, field_a=field_a, field_b=field_b))
            if results:
                return list(results[0]["path"])
        except Exception:
            pass
        return None

    def _probe_cosine(self, field_a: str, field_b: str) -> float:
        """Return cosine similarity between two probe directions.

        Returns 0.0 when either field is not in the probe registry.
        """
        a = self._unit_probes.get(field_a)
        b = self._unit_probes.get(field_b)
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))
