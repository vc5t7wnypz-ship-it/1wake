// ============================================================
// WAKE — Semantic Field Queries
// ============================================================
// All queries assume the schema in graph/schema/wake_schema.cypher.
// Parameters passed via $param syntax.
// ============================================================


// ------------------------------------------------------------
// Q1: All tokens activating a given semantic field,
//     ordered by page/line.
//
// Parameters:
//   $field_name  — semantic field name (e.g. "water", "cycle", "ALP")
//
// Returns: token surface, page, line, position, and activation strength
//          (if stored), plus optional Vico cycle and sefirah.
// ------------------------------------------------------------
// Q1: Tokens activating a semantic field
MATCH (t:WakeToken)-[rel:ACTIVATES]->(s:SemanticField {name: $field_name})
OPTIONAL MATCH (t)-[:BELONGS_TO_CYCLE]->(v:VicoCycle)
OPTIONAL MATCH (t)-[:RESONATES_WITH]->(k:KabbalahNode)
RETURN
    t.surface           AS surface,
    t.page              AS page,
    t.line              AS line,
    t.position          AS position,
    coalesce(rel.strength, 1.0) AS activation_strength,
    v.cycle             AS vico_cycle,
    v.name              AS vico_name,
    k.sefirah           AS sefirah
ORDER BY t.page ASC, t.line ASC, t.position ASC;


// ------------------------------------------------------------
// Q2: Semantic field co-activation network.
//     Which pairs of semantic fields appear together on the
//     same WakeToken?  Returns edge weights for network analysis.
//
// Parameters: (none — full network)
//   Optionally filter by minimum co-occurrence count with $min_cooc.
//
// Returns: field_a, field_b (alphabetical), co-occurrence count,
//          and token examples.
// ------------------------------------------------------------
// Q2: Semantic field co-activation pairs
MATCH (t:WakeToken)-[:ACTIVATES]->(s1:SemanticField)
MATCH (t)-[:ACTIVATES]->(s2:SemanticField)
WHERE s1.name < s2.name   // alphabetical dedup — each pair once
WITH s1.name AS field_a, s2.name AS field_b,
     count(t) AS cooc_count,
     collect(t.surface)[..5] AS example_tokens
WHERE cooc_count >= $min_cooc
RETURN
    field_a,
    field_b,
    cooc_count,
    example_tokens
ORDER BY cooc_count DESC;


// ------------------------------------------------------------
// Q3: Lens-foregrounded semantic fields with their token counts.
//     Shows which fields are emphasised by each interpretive lens
//     and how many tokens activate each foregrounded field.
//
// Parameters: (none — all lenses)
//
// Returns: lens name, lens type, foregrounded field name,
//          token count for that field, and description of the field.
// ------------------------------------------------------------
// Q3: Lens-foregrounded fields with token counts
MATCH (l:Lens)-[:FOREGROUNDS]->(s:SemanticField)
OPTIONAL MATCH (t:WakeToken)-[:ACTIVATES]->(s)
WITH l, s, count(t) AS token_count
RETURN
    l.name              AS lens_name,
    l.type              AS lens_type,
    s.name              AS field_name,
    coalesce(s.description, "") AS field_description,
    token_count
ORDER BY l.name ASC, token_count DESC;


// ------------------------------------------------------------
// Q4: Top semantic fields by activation strength for a passage range.
//     Summarise the dominant themes within a page range.
//
// Parameters:
//   $page_start  — first page of the passage (inclusive)
//   $page_end    — last page of the passage (inclusive)
//   $limit       — number of top fields to return (e.g. 20)
//
// Returns: field name, total activating tokens, sum/mean activation
//          strength, example token surfaces.
// ------------------------------------------------------------
// Q4: Top semantic fields in a passage range
MATCH (t:WakeToken)-[rel:ACTIVATES]->(s:SemanticField)
WHERE t.page >= $page_start AND t.page <= $page_end
WITH
    s.name                          AS field_name,
    count(t)                        AS token_count,
    sum(coalesce(rel.strength, 1.0))  AS total_strength,
    avg(coalesce(rel.strength, 1.0))  AS mean_strength,
    collect(DISTINCT t.surface)[..8]  AS sample_tokens
RETURN
    field_name,
    token_count,
    total_strength,
    mean_strength,
    sample_tokens
ORDER BY total_strength DESC, token_count DESC
LIMIT $limit;
