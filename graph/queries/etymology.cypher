// ============================================================
// WAKE — Etymology Queries
// ============================================================
// All queries assume the schema in graph/schema/wake_schema.cypher.
// Parameters are passed via the Neo4j driver or cypher-shell using
// the $param syntax.
// ============================================================


// ------------------------------------------------------------
// Q1: Get all etymological roots for a surface token
//     with confidence scores, sorted strongest-first.
//
// Parameters:
//   $surface  — the surface form to look up (e.g. "riverrun")
//
// Returns: root form, gloss, language code, confidence, token page/line
// ------------------------------------------------------------
// Q1: All roots for a surface token
MATCH (t:WakeToken {surface: $surface})-[rel:HAS_ROOT]->(r:EtymRoot)
OPTIONAL MATCH (r)-[:IN_LANGUAGE]->(lang:Language)
RETURN
    t.surface           AS surface,
    t.page              AS page,
    t.line              AS line,
    r.language          AS language_code,
    coalesce(lang.name, r.language) AS language_name,
    r.form              AS root_form,
    r.gloss             AS gloss,
    rel.confidence      AS confidence
ORDER BY rel.confidence DESC, r.language ASC;


// ------------------------------------------------------------
// Q2: Find all tokens sharing a root with a given token
//     (cross-language cognate search).
//
// Parameters:
//   $surface  — the anchor surface form
//
// Returns: cognate token surface, page, line, shared root form,
//          language, and the confidence on both ends.
// ------------------------------------------------------------
// Q2: Cross-language cognates sharing a root
MATCH (anchor:WakeToken {surface: $surface})-[rel_a:HAS_ROOT]->(r:EtymRoot)
MATCH (cognate:WakeToken)-[rel_c:HAS_ROOT]->(r)
WHERE cognate.token_id <> anchor.token_id
RETURN
    anchor.surface                      AS anchor_surface,
    cognate.surface                     AS cognate_surface,
    cognate.page                        AS cognate_page,
    cognate.line                        AS cognate_line,
    r.language                          AS shared_language,
    r.form                              AS shared_root_form,
    r.gloss                             AS shared_gloss,
    rel_a.confidence                    AS anchor_confidence,
    rel_c.confidence                    AS cognate_confidence,
    rel_a.confidence * rel_c.confidence AS joint_confidence
ORDER BY joint_confidence DESC, cognate.page ASC, cognate.line ASC;


// ------------------------------------------------------------
// Q3: Trace full etymology chain from surface token → roots
//     → tokens sharing those roots → their additional roots.
//     (3-hop exploration: token → root → cognate_token → root)
//
// Parameters:
//   $surface    — starting surface form
//   $min_conf   — minimum confidence threshold on each HAS_ROOT edge
//                 (e.g. 0.7 to filter speculative connections)
//
// Returns: path summary with each hop labelled.
// ------------------------------------------------------------
// Q3: 3-hop etymology chain
MATCH (t1:WakeToken {surface: $surface})-[r1:HAS_ROOT]->(root1:EtymRoot)
WHERE r1.confidence >= $min_conf
MATCH (t2:WakeToken)-[r2:HAS_ROOT]->(root1)
WHERE t2.token_id <> t1.token_id
  AND r2.confidence >= $min_conf
MATCH (t2)-[r3:HAS_ROOT]->(root2:EtymRoot)
WHERE r3.confidence >= $min_conf
RETURN
    t1.surface          AS start_surface,
    t1.page             AS start_page,
    root1.language      AS hop1_language,
    root1.form          AS hop1_root,
    r1.confidence       AS hop1_conf,
    t2.surface          AS midpoint_surface,
    t2.page             AS midpoint_page,
    root2.language      AS hop2_language,
    root2.form          AS hop2_root,
    r3.confidence       AS hop2_conf
ORDER BY hop1_conf DESC, hop2_conf DESC, t2.page ASC
LIMIT 100;


// ------------------------------------------------------------
// Q4: Tokens with the most diverse language attributions
//     (superposition candidates — highest linguistic entropy).
//
// Parameters:
//   $min_languages  — minimum distinct language count (e.g. 3)
//   $limit          — how many results to return (e.g. 25)
//
// Returns: surface form, page, line, distinct language count,
//          list of language codes, and mean confidence across roots.
// ------------------------------------------------------------
// Q4: Superposition candidates — most linguistically diverse tokens
MATCH (t:WakeToken)-[rel:HAS_ROOT]->(r:EtymRoot)
WITH
    t,
    count(DISTINCT r.language)  AS lang_count,
    collect(DISTINCT r.language) AS languages,
    avg(rel.confidence)          AS mean_confidence
WHERE lang_count >= $min_languages
RETURN
    t.surface        AS surface,
    t.page           AS page,
    t.line           AS line,
    lang_count       AS distinct_language_count,
    languages        AS language_codes,
    mean_confidence  AS mean_root_confidence
ORDER BY lang_count DESC, mean_confidence DESC
LIMIT $limit;
