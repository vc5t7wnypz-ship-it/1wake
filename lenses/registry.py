"""
Lens registry for the WAKE interpretability engine.

Provides a central catalogue of all available lenses, a lookup helper, and
the :class:`ComposedLens` utility for blending multiple lenses.

Public symbols
──────────────
ALL_LENSES       — dict[str, type[Lens]]: slug → class
get_lens()       — factory: name + optional graph_client → Lens
compose_lenses() — convenience constructor for ComposedLens
list_lenses()    — sorted list of registered names
ComposedLens     — weighted ensemble of Lens instances
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type

from .base import Lens, LensConfig
from .brunian import BrunianLens
from .freudian import FreudianLens
from .irish_mythology import IrishMythologyLens
from .kabbalistic import KabbalisticLens
from .norse import NorseLens
from .viconian import ViconianLens

# ---------------------------------------------------------------------------
# Registry — maps slug names to lens classes
# ---------------------------------------------------------------------------

ALL_LENSES: Dict[str, Type[Lens]] = {
    "viconian": ViconianLens,
    "kabbalistic": KabbalisticLens,
    "freudian": FreudianLens,
    "irish_mythology": IrishMythologyLens,
    "norse": NorseLens,
    "brunian": BrunianLens,
}


def get_lens(name: str, graph_client: Any = None) -> Lens:
    """Instantiate a lens by its slug name.

    Parameters
    ----------
    name:
        The slug name of the lens (e.g. ``"viconian"``).
    graph_client:
        Optional Neo4j driver to inject.

    Returns
    -------
    Lens
        An instantiated, ready-to-use lens.

    Raises
    ------
    KeyError
        When *name* is not registered.
    """
    try:
        cls = ALL_LENSES[name]
    except KeyError:
        available = ", ".join(sorted(ALL_LENSES))
        raise KeyError(
            f"Unknown lens {name!r}.  Available lenses: {available}"
        ) from None
    return cls(graph_client=graph_client)


def compose_lenses(
    names: List[str],
    weights: Optional[List[float]] = None,
    graph_client: Any = None,
) -> "ComposedLens":
    """Build a :class:`ComposedLens` from a list of lens names.

    Parameters
    ----------
    names:
        Ordered list of lens slug identifiers.
    weights:
        Optional parallel list of non-negative weights.  Defaults to uniform
        (1/N each).  Will be normalised to sum to 1 internally.
    graph_client:
        Optional graph client shared across all instantiated lenses.

    Returns
    -------
    ComposedLens
        Ready-to-use composed lens ensemble.

    Raises
    ------
    KeyError
        If any name in *names* is not found in :data:`ALL_LENSES`.
    ValueError
        If *names* is empty, or if *weights* length mismatches *names*.
    """
    if not names:
        raise ValueError("compose_lenses() requires at least one lens name.")
    lenses = [get_lens(n, graph_client=graph_client) for n in names]
    return ComposedLens(lenses=lenses, weights=weights)


def list_lenses() -> List[str]:
    """Return a sorted list of all registered lens names."""
    return sorted(ALL_LENSES.keys())


# ---------------------------------------------------------------------------
# ComposedLens — blend multiple lenses
# ---------------------------------------------------------------------------

class ComposedLens:
    """A weighted blend of two or more :class:`Lens` instances.

    Running multiple lenses on the same passage is the primary mechanism for
    detecting *superposition* (multiple semantic frames simultaneously active
    in the residual stream) versus *parse collapse* (one frame dominates and
    suppresses the rest).

    The composed lens combines the foregrounded fields, probe targets, and
    system prompts of its component lenses.  When building context for a
    passage each component lens is consulted in order and their contexts are
    concatenated, weighted by the lens weights.

    Parameters
    ----------
    lenses:
        List of :class:`Lens` instances to compose.
    weights:
        Optional list of non-negative floats, one per lens.  Will be
        normalised to sum to 1.0.  When *None* each lens is given equal
        weight.
    graph_client:
        Optional graph client (overrides any client already attached to
        the component lenses).
    """

    def __init__(
        self,
        lenses: List[Lens],
        weights: Optional[List[float]] = None,
        graph_client: Any = None,
    ) -> None:
        if not lenses:
            raise ValueError("ComposedLens requires at least one lens.")
        self.lenses: List[Lens] = list(lenses)

        if weights is None:
            n = len(self.lenses)
            self.weights: List[float] = [1.0 / n] * n
        else:
            if len(weights) != len(self.lenses):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of lenses ({len(self.lenses)})."
                )
            if any(w < 0 for w in weights):
                raise ValueError("All weights must be non-negative.")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Weights must sum to a positive number.")
            self.weights = [w / total for w in weights]

        if graph_client is not None:
            for lens in self.lenses:
                lens.graph = graph_client

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Concatenated slug names joined by ``+``."""
        return "+".join(lens.config.name for lens in self.lenses)

    @property
    def foregrounded_fields(self) -> List[str]:
        """Union of all component foregrounded fields (de-duplicated, insertion-ordered)."""
        seen: Dict[str, None] = {}
        for lens in self.lenses:
            for f in lens.config.foregrounded_fields:
                seen[f] = None
        return list(seen.keys())

    @property
    def probe_targets(self) -> List[str]:
        """Union of all component probe targets (de-duplicated, insertion-ordered)."""
        seen: Dict[str, None] = {}
        for lens in self.lenses:
            for t in lens.config.probe_targets:
                seen[t] = None
        return list(seen.keys())

    @property
    def attention_prior_union(self) -> Dict[str, float]:
        """Weighted-average attention priors across all component lenses.

        For each attention-head role present in any component lens, the
        composed prior is the weighted average of per-lens values (treating
        absent roles as 0).
        """
        all_roles: set = set()
        for lens in self.lenses:
            all_roles.update(lens.config.attention_priors.keys())

        combined: Dict[str, float] = {}
        for role in sorted(all_roles):
            combined[role] = sum(
                w * lens.config.attention_priors.get(role, 0.0)
                for lens, w in zip(self.lenses, self.weights)
            )
        return combined

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    def get_context_for_passage(self, passage: str, wake_tokens: list) -> str:
        """Build a composed context string from all component lenses.

        Each component lens contributes a block labelled with its weight.
        The blocks are concatenated with a separator line.

        Parameters
        ----------
        passage:
            Raw Wake passage text.
        wake_tokens:
            List of :class:`WakeToken` objects for *passage*.

        Returns
        -------
        str
            A combined context string with all lens perspectives.
        """
        parts: List[str] = []
        for lens, weight in zip(self.lenses, self.weights):
            ctx = lens.get_context_for_passage(passage, wake_tokens)
            parts.append(f"[Weight: {weight:.3f}]\n{ctx}")
        separator = "\n" + "─" * 60 + "\n"
        return separator.join(parts)

    def get_all_contexts(
        self, passage: str, wake_tokens: list
    ) -> List[Dict[str, Any]]:
        """Run every component lens on *passage* and return structured results.

        Returns
        -------
        list[dict]
            One dict per lens with keys ``lens`` (name), ``weight`` (float),
            and ``context`` (str).
        """
        results: List[Dict[str, Any]] = []
        for lens, weight in zip(self.lenses, self.weights):
            ctx = lens.get_context_for_passage(passage, wake_tokens)
            results.append(
                {"lens": lens.config.name, "weight": weight, "context": ctx}
            )
        return results

    def get_merged_context(self, passage: str, wake_tokens: list) -> str:
        """Return a merged context string, lenses sorted descending by weight.

        Lenses with equal weights are sorted alphabetically by name for
        determinism.  A superposition-analysis header is prepended.

        Returns
        -------
        str
            All lens context blocks concatenated with ``═══`` dividers.
        """
        contexts = self.get_all_contexts(passage, wake_tokens)
        contexts.sort(key=lambda d: (-d["weight"], d["lens"]))
        divider = "\n" + "═" * 72 + "\n"
        header = (
            f"[COMPOSED LENS: {', '.join(d['lens'] for d in contexts)}]\n"
            f"SUPERPOSITION ANALYSIS — {len(self.lenses)} frames active\n"
            f"Passage: {passage}\n"
        )
        body = divider.join(
            f"[weight={d['weight']:.3f}]\n{d['context']}" for d in contexts
        )
        return f"{header}{divider}{body}"

    # ------------------------------------------------------------------
    # Superposition scoring
    # ------------------------------------------------------------------

    def superposition_score(self, passage: str, wake_tokens: list) -> float:
        """Heuristic superposition score in [0, 1] for *passage*.

        Measures how evenly the lens activations are distributed: a score
        near 1.0 indicates all lenses activate equally (genuine superposition);
        a score near 0.0 indicates one lens dominates (parse collapse).

        This is a *structural* heuristic based on the count of activated graph
        nodes per lens, not a probe-based score.  The authoritative
        superposition measurement is performed by the probe pipeline; this
        method is useful for fast exploratory ranking.

        Returns
        -------
        float
            Normalised Shannon entropy of the lens-node-count distribution,
            in [0, 1].  Returns 1.0 when no graph data is available (offline
            mode), treating the absence of graph evidence as maximal ambiguity.
        """
        if len(self.lenses) == 1:
            return 1.0

        node_counts: List[float] = []
        for lens in self.lenses:
            nodes = lens._traverse_graph(wake_tokens)
            node_counts.append(float(len(nodes)) if nodes else 0.0)

        total = sum(node_counts)
        if total == 0:
            return 1.0

        probs = [c / total for c in node_counts]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        max_entropy = math.log(len(self.lenses))
        return entropy / max_entropy if max_entropy > 0.0 else 1.0

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.lenses)

    def __iter__(self):
        return iter(self.lenses)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{lens.config.name}:{w:.2f}"
            for lens, w in zip(self.lenses, self.weights)
        )
        return f"<ComposedLens [{parts}]>"
