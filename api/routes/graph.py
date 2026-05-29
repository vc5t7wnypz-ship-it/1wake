"""
Graph query routes for the WAKE interpretability API.

Endpoints
---------
GET /api/wake/graph/node/{surface}            — single Wake-token node
GET /api/wake/graph/path                      — shortest path between two nodes
GET /api/wake/graph/neighborhood/{surface}    — N-hop neighbourhood subgraph
GET /api/wake/graph/semantic-fields           — all semantic fields with token counts
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..schemas.requests import GraphPathRequest
from ..schemas.responses import GraphNodeResponse
from ..state import AppState, app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wake/graph", tags=["graph"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_state() -> AppState:
    """Return the shared AppState singleton."""
    return app_state


def _require_graph(state: AppState) -> Any:
    """Raise 503 if the Neo4j graph is not connected."""
    if state.graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Neo4j graph is not connected.  "
                "Set NEO4J_URI and ensure the database is running."
            ),
        )
    return state.graph


# ---------------------------------------------------------------------------
# GET /api/wake/graph/node/{surface}
# ---------------------------------------------------------------------------

@router.get(
    "/node/{surface}",
    response_model=GraphNodeResponse,
    summary="Fetch a single Wake-token graph node",
    description=(
        "Returns the WakeToken node for *surface*, together with its "
        "etymological roots, semantic fields, and outgoing relationships."
    ),
)
async def get_node(
    surface: str,
    state: AppState = Depends(get_state),
) -> GraphNodeResponse:
    """Fetch a WakeToken node by its surface form."""
    driver = _require_graph(state)

    query = """
        MATCH (t:WakeToken {surface: $surface})
        OPTIONAL MATCH (t)-[hr:HAS_ROOT]->(r:EtymRoot)
        OPTIONAL MATCH (r)-[:IN_LANGUAGE]->(lang:Language)
        OPTIONAL MATCH (t)-[:ACTIVATES]->(sf:SemanticField)
        OPTIONAL MATCH (t)-[:BELONGS_TO_CYCLE]->(vc:VicoCycle)
        RETURN
            t.surface     AS surface,
            t.page        AS page,
            t.line        AS line,
            t.notes       AS notes,
            collect(DISTINCT {
                language:   r.language,
                form:       r.form,
                gloss:      r.gloss,
                confidence: hr.confidence
            }) AS roots,
            collect(DISTINCT {
                name:       sf.name,
                vico_cycle: vc.cycle
            }) AS semantic_fields
        LIMIT 1
    """

    try:
        with driver.session() as session:
            record = session.run(query, surface=surface).single()
    except Exception as exc:
        logger.error("Graph query failed for node '%s': %s", surface, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph query error: {exc}",
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No WakeToken node found for surface '{surface}'.",
        )

    # Fetch connections separately (outgoing edges)
    conn_query = """
        MATCH (t:WakeToken {surface: $surface})-[r]->(n)
        RETURN type(r) AS rel_type, labels(n) AS target_labels,
               properties(n) AS target_props
        LIMIT 50
    """
    connections: List[Dict[str, Any]] = []
    try:
        with driver.session() as session:
            for row in session.run(conn_query, surface=surface):
                connections.append({
                    "type": row["rel_type"],
                    "target": {
                        "labels": row["target_labels"],
                        **dict(row["target_props"]),
                    },
                    "properties": {},
                })
    except Exception as exc:
        logger.warning("Could not fetch connections for '%s': %s", surface, exc)

    # Filter out null entries in roots / semantic_fields
    roots: List[Dict[str, Any]] = [
        r for r in (record["roots"] or [])
        if r.get("form") is not None
    ]
    semantic_fields: List[Dict[str, Any]] = [
        sf for sf in (record["semantic_fields"] or [])
        if sf.get("name") is not None
    ]

    return GraphNodeResponse(
        surface=record["surface"] or surface,
        page=int(record["page"] or 0),
        line=int(record["line"] or 0),
        roots=roots,
        semantic_fields=semantic_fields,
        connections=connections,
    )


# ---------------------------------------------------------------------------
# GET /api/wake/graph/path
# ---------------------------------------------------------------------------

@router.get(
    "/path",
    summary="Find shortest path between two Wake-token nodes",
    description=(
        "Returns the shortest path in the WAKE knowledge graph between "
        "*from_surface* and *to_surface*, up to *max_hops* hops."
    ),
)
async def get_path(
    from_surface: str = Query(..., description="Source token surface form"),
    to_surface: str = Query(..., description="Destination token surface form"),
    max_hops: int = Query(default=4, ge=1, le=10, description="Maximum hops"),
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """Find the shortest path between two Wake-token nodes."""
    if from_surface == to_surface:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="from_surface and to_surface must be different.",
        )

    driver = _require_graph(state)

    # Use Cypher variable-length path with shortestPath
    query = f"""
        MATCH
            (src:WakeToken {{surface: $from_surface}}),
            (dst:WakeToken {{surface: $to_surface}}),
            path = shortestPath((src)-[*1..{max_hops}]-(dst))
        RETURN path, length(path) AS hops
        LIMIT 1
    """

    try:
        with driver.session() as session:
            record = session.run(
                query,
                from_surface=from_surface,
                to_surface=to_surface,
            ).single()
    except Exception as exc:
        logger.error("Path query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph path query error: {exc}",
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No path found between '{from_surface}' and '{to_surface}' "
                f"within {max_hops} hops."
            ),
        )

    path = record["path"]
    hops = int(record["hops"])

    # Serialise path nodes and relationships
    path_nodes: List[Dict[str, Any]] = []
    path_rels: List[Dict[str, Any]] = []

    try:
        for node in path.nodes:
            path_nodes.append(
                {"labels": list(node.labels), **dict(node.items())}
            )
        for rel in path.relationships:
            path_rels.append(
                {
                    "type": rel.type,
                    "start": dict(rel.start_node.items()),
                    "end": dict(rel.end_node.items()),
                    **dict(rel.items()),
                }
            )
    except Exception as exc:
        logger.warning("Error serialising path: %s", exc)

    return {
        "from_surface": from_surface,
        "to_surface": to_surface,
        "hops": hops,
        "nodes": path_nodes,
        "relationships": path_rels,
    }


# ---------------------------------------------------------------------------
# GET /api/wake/graph/neighborhood/{surface}
# ---------------------------------------------------------------------------

@router.get(
    "/neighborhood/{surface}",
    summary="Return the N-hop neighbourhood of a Wake-token node",
    description=(
        "Returns all nodes reachable from *surface* within *hops* "
        "relationship traversals.  Default 2 hops."
    ),
)
async def get_neighborhood(
    surface: str,
    hops: int = Query(default=2, ge=1, le=5, description="Number of hops (1–5)"),
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """Return the N-hop neighbourhood subgraph around *surface*."""
    driver = _require_graph(state)

    query = f"""
        MATCH (root:WakeToken {{surface: $surface}})
        CALL apoc.path.subgraphAll(root, {{maxLevel: {hops}}})
        YIELD nodes, relationships
        RETURN nodes, relationships
    """

    # Fallback if APOC is not available
    query_no_apoc = f"""
        MATCH (root:WakeToken {{surface: $surface}})-[*0..{hops}]-(n)
        WITH DISTINCT n
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m IN collect(n)
        RETURN collect(DISTINCT n) AS nodes, collect(DISTINCT r) AS relationships
    """

    nodes_data: List[Dict[str, Any]] = []
    rels_data: List[Dict[str, Any]] = []
    root_found = False

    try:
        with driver.session() as session:
            try:
                record = session.run(query, surface=surface).single()
            except Exception:
                record = None

            if record is None:
                record = session.run(query_no_apoc, surface=surface).single()

        if record is not None:
            root_found = True
            for node in (record["nodes"] or []):
                nodes_data.append(
                    {"labels": list(node.labels), **dict(node.items())}
                )
            for rel in (record["relationships"] or []):
                try:
                    rels_data.append(
                        {
                            "type": rel.type,
                            "start": dict(rel.start_node.items()),
                            "end": dict(rel.end_node.items()),
                        }
                    )
                except Exception:
                    pass

    except Exception as exc:
        logger.error("Neighbourhood query failed for '%s': %s", surface, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph neighbourhood query error: {exc}",
        ) from exc

    if not root_found or not nodes_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No node found for surface '{surface}'.",
        )

    return {
        "surface": surface,
        "hops": hops,
        "node_count": len(nodes_data),
        "relationship_count": len(rels_data),
        "nodes": nodes_data,
        "relationships": rels_data,
    }


# ---------------------------------------------------------------------------
# GET /api/wake/graph/semantic-fields
# ---------------------------------------------------------------------------

@router.get(
    "/semantic-fields",
    summary="List all semantic fields with token counts",
    description=(
        "Returns every SemanticField node in the WAKE graph together with "
        "the number of WakeToken nodes that activate it."
    ),
)
async def list_semantic_fields(
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """List all SemanticField nodes with their token-count statistics."""
    driver = _require_graph(state)

    query = """
        MATCH (sf:SemanticField)
        OPTIONAL MATCH (t:WakeToken)-[:ACTIVATES]->(sf)
        RETURN sf.name AS name, count(t) AS token_count
        ORDER BY token_count DESC
    """

    try:
        with driver.session() as session:
            records = session.run(query).data()
    except Exception as exc:
        logger.error("semantic-fields query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph query error: {exc}",
        ) from exc

    fields: List[Dict[str, Any]] = [
        {"name": r["name"], "token_count": int(r["token_count"] or 0)}
        for r in records
        if r.get("name")
    ]

    return {
        "semantic_fields": fields,
        "total": len(fields),
    }
