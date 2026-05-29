"""
Base classes for the WAKE lens system.

A Lens is a named query state that defines how to read a passage: which
semantic fields to foreground, how to traverse the knowledge graph, where to
place attention priors, and which linear probes to fire.  Multiple lenses
running on the same passage reveal whether the LLM is holding genuine
superposition (multiple semantic frames active simultaneously) or has
collapsed to a single parse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LensConfig:
    """Declarative configuration bundle for a single lens.

    Attributes
    ----------
    name:
        Short, slug-style identifier, e.g. ``"viconian"``.
    system_prompt:
        The full system prompt that primes the model to read through this
        interpretive frame.
    foregrounded_fields:
        Semantic-field names (matching ``SemanticField.name`` nodes in the
        knowledge graph) that this lens treats as salient.
    graph_traversal:
        Cypher-query parameters that control how the graph is walked when
        building context.  Keys vary per lens but must include at least
        ``source_entity`` (optional) and ``language_filter`` (optional list
        of BCP-47/ISO 639-3 codes).
    attention_priors:
        Mapping from attention-head role name to a float in [0, 1] that
        biases which heads the lens considers diagnostic.
    probe_targets:
        Names of linear-probe classifiers (registered in the probe registry)
        that should be evaluated when this lens is active.
    """

    name: str
    system_prompt: str
    foregrounded_fields: list[str] = field(default_factory=list)
    graph_traversal: dict[str, Any] = field(default_factory=dict)
    attention_priors: dict[str, float] = field(default_factory=dict)
    probe_targets: list[str] = field(default_factory=list)


class Lens(ABC):
    """Abstract base class for all WAKE lenses.

    Subclasses must implement :meth:`_build_config`.  Everything else is
    provided here.

    Parameters
    ----------
    graph_client:
        An optional Neo4j ``GraphDatabase`` driver or a compatible wrapper
        that exposes a ``run(query, **params)`` method.  When ``None`` the
        lens works in offline mode: graph nodes are skipped and only the
        system prompt is injected.
    """

    def __init__(self, graph_client: Any = None) -> None:
        self.graph: Any = graph_client
        self.config: LensConfig = self._build_config()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_config(self) -> LensConfig:
        """Return the :class:`LensConfig` that characterises this lens."""
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context_for_passage(self, passage: str, wake_tokens: list) -> str:
        """Build a lens-conditioned context string for *passage*.

        Parameters
        ----------
        passage:
            The raw Wake text segment being analysed.
        wake_tokens:
            A list of token objects, each expected to carry a ``surface``
            attribute (the raw string form of the token).

        Returns
        -------
        str
            A formatted context block ready for injection into a model
            prompt or interpretability pipeline.
        """
        relevant_nodes = self._traverse_graph(wake_tokens)
        return self._format_context(passage, relevant_nodes)

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def _traverse_graph(self, wake_tokens: list) -> list:
        """Query the knowledge graph for nodes activated by *wake_tokens*.

        Returns an empty list when no graph client is attached (offline mode).
        """
        if self.graph is None:
            return []

        token_surfaces: list[str] = [t.surface for t in wake_tokens]

        # Base Cypher: follow WakeToken → SemanticField edges then collect
        # one hop of additional context around each matched token.
        query = """
            MATCH (t:WakeToken)-[:ACTIVATES]->(sf:SemanticField)
            WHERE t.surface IN $surfaces
            AND sf.name IN $foregrounded
            RETURN t, sf, [(t)-[r]->(n) | {rel: type(r), node: n}] AS context
            LIMIT 50
        """

        # Extend the query if this lens declares a source-entity filter.
        source_entity: str | None = self.config.graph_traversal.get("source_entity")
        if source_entity:
            query = """
                MATCH (t:WakeToken)-[:ACTIVATES]->(sf:SemanticField)
                WHERE t.surface IN $surfaces
                AND sf.name IN $foregrounded
                WITH t, sf
                OPTIONAL MATCH (src:SourceEntity {name: $source_entity})-[:ENCODES]->(sf)
                RETURN t, sf, src,
                       [(t)-[r]->(n) | {rel: type(r), node: n}] AS context
                LIMIT 50
            """
            return self.graph.run(
                query,
                surfaces=token_surfaces,
                foregrounded=self.config.foregrounded_fields,
                source_entity=source_entity,
            )

        return self.graph.run(
            query,
            surfaces=token_surfaces,
            foregrounded=self.config.foregrounded_fields,
        )

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _format_context(self, passage: str, nodes: list) -> str:
        """Render *nodes* and *passage* into a lens-annotated context string."""
        header = f"[{self.config.name.upper()} LENS]\n{self.config.system_prompt}"

        if not nodes:
            return f"{header}\n\nPassage: {passage}"

        node_lines: list[str] = []
        for n in nodes[:20]:
            if hasattr(n, "get"):
                t_node = n.get("t") or {}
                sf_node = n.get("sf") or {}
                surface: str = t_node.get("surface", "") if hasattr(t_node, "get") else ""
                field_name: str = sf_node.get("name", "") if hasattr(sf_node, "get") else ""
                if surface and field_name:
                    node_lines.append(f"  {surface} → {field_name}")

        node_section = "\n".join(node_lines)
        return (
            f"{header}\n\n"
            f"Activated graph nodes:\n{node_section}\n\n"
            f"Passage: {passage}"
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Lens name={self.config.name!r}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lens):
            return NotImplemented
        return self.config.name == other.config.name

    def __hash__(self) -> int:
        return hash(self.config.name)
