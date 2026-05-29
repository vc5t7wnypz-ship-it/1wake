// ============================================================
// WAKE — Graph Path Queries
// ============================================================
// Path queries over the WAKE token graph, using PRECEDES,
// PRECEDES_LINE, HAS_ROOT, ACTIVATES, BELONGS_TO_CYCLE,
// RESONATES_WITH, and ECHOES relationships.
//
// All queries assume the schema in graph/schema/wake_schema.cypher.
// Parameters passed via $param syntax.
//
// NOTE: Shortest-path queries use Neo4j's built-in shortestPath()
//       and allShortestPaths() procedures.  For large subgraphs,
//       consider using the GDS library: gds.shortestPath.dijkstra.
// ============================================================


// ------------------------------------------------------------
// Q1: Shortest path between two tokens (any relationship type).
//
// Parameters:
//   $surface_a  — surface form of the start token (e.g. "riverrun")
//   $surface_b  — surface form of the end token (e.g. "Howth")
//
// Returns: path length, sequence of node labels + key properties,
//          and relationship types along the path.
// ------------------------------------------------------------
// Q1: Shortest path between two tokens
MATCH (a:WakeToken {surface: $surface_a}),
      (b:WakeToken {surface: $surface_b}),
      p = shortestPath((a)-[*..10]-(b))
RETURN
    length(p)                                           AS path_length,
    [n IN nodes(p)    | coalesce(n.surface, n.name, n.form, n.sefirah, n.cycle)] AS node_labels,
    [r IN relationships(p) | type(r)]                  AS rel_types,
    [n IN nodes(p)    | labels(n)[0]]                  AS node_types
ORDER BY path_length ASC
LIMIT 5;


// ------------------------------------------------------------
// Q2: All paths ≤ 4 hops between two tokens with relationship
//     type sequences.  Useful for enumerating all narrative
//     routes connecting two surface forms.
//
// Parameters:
//   $surface_a    — start token surface form
//   $surface_b    — end token surface form
//   $max_hops     — maximum path length (recommend ≤ 4 for performance)
//
// Returns: path length, ordered node labels, ordered rel-type sequence,
//          and a human-readable path description string.
// ------------------------------------------------------------
// Q2: All paths up to $max_hops between two tokens
MATCH (a:WakeToken {surface: $surface_a}),
      (b:WakeToken {surface: $surface_b}),
      p = allShortestPaths((a)-[*..4]-(b))
WITH p,
     length(p)                                               AS hops,
     [n IN nodes(p) | coalesce(n.surface, n.name, n.form, n.sefirah, toString(n.cycle))]
         AS node_labels,
     [r IN relationships(p) | type(r)]                       AS rel_seq
RETURN
    hops,
    node_labels,
    rel_seq,
    reduce(s = '', i IN range(0, size(node_labels)-1) |
        s + node_labels[i] +
        CASE WHEN i < size(rel_seq) THEN ' -[' + rel_seq[i] + ']-> ' ELSE '' END
    ) AS path_description
ORDER BY hops ASC, path_description ASC
LIMIT 50;


// ------------------------------------------------------------
// Q3: Nodes reachable from a token within 2 hops,
//     grouped by node type and relationship type.
//
// Parameters:
//   $surface     — starting token surface form
//
// Returns: neighbour node type, relationship type used,
//          count of such neighbours, and example node identifiers.
// ------------------------------------------------------------
// Q3: 2-hop neighbourhood, grouped by type
MATCH (t:WakeToken {surface: $surface})-[r*1..2]-(neighbour)
WHERE neighbour <> t
WITH
    labels(neighbour)[0]                              AS neighbour_type,
    type(last(r))                                     AS last_rel_type,
    count(DISTINCT neighbour)                          AS count,
    collect(DISTINCT
        coalesce(neighbour.surface, neighbour.name,
                 neighbour.form, neighbour.sefirah,
                 toString(neighbour.cycle)))[..6]     AS examples
RETURN
    neighbour_type,
    last_rel_type,
    count,
    examples
ORDER BY count DESC, neighbour_type ASC;


// ------------------------------------------------------------
// Q4: Bridge nodes — nodes that lie on many shortest paths
//     between token pairs in a given page range.
//     High betweenness centrality indicates a node that
//     mediates many thematic connections (structural bridges).
//
// NOTE: For production use, replace this approximation with
//       the GDS betweenness centrality procedure:
//         CALL gds.betweenness.stream('wakeGraph') YIELD nodeId, score
//       The query below computes a manual approximation over a
//       sample of token pairs on the given page.
//
// Parameters:
//   $page         — page number to sample token pairs from
//   $sample_size  — number of token pairs to sample (e.g. 20)
//
// Returns: bridge node identifier, node type, betweenness-proxy score,
//          and list of paths it appears on.
// ------------------------------------------------------------
// Q4: Bridge nodes (betweenness approximation) for a page sample
MATCH (t:WakeToken)
WHERE t.page = $page
WITH collect(t) AS page_tokens
UNWIND range(0, toInteger($sample_size) - 1) AS idx
WITH page_tokens[idx % size(page_tokens)] AS token_a,
     page_tokens[(idx + toInteger(size(page_tokens) / 2)) % size(page_tokens)] AS token_b
WHERE token_a.token_id <> token_b.token_id
MATCH p = shortestPath((token_a)-[*..8]-(token_b))
UNWIND nodes(p)[1..-1] AS bridge_candidate   // exclude endpoints
WITH
    bridge_candidate,
    count(p) AS path_count
WHERE path_count > 1
RETURN
    coalesce(
        bridge_candidate.surface,
        bridge_candidate.name,
        bridge_candidate.form,
        bridge_candidate.sefirah,
        toString(bridge_candidate.cycle)
    )                           AS bridge_node_id,
    labels(bridge_candidate)[0] AS node_type,
    path_count                  AS betweenness_proxy
ORDER BY betweenness_proxy DESC
LIMIT 20;
