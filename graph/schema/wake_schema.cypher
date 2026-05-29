// ============================================================
// WAKE Interpretability Project — Neo4j Schema
// ============================================================
// Run once against a fresh Neo4j 5.x instance.
// Order matters: constraints before indexes.
// ============================================================

// ------------------------------------------------------------
// CONSTRAINTS
// ------------------------------------------------------------

// WakeToken: unique identifier is page + line + position + surface form
CREATE CONSTRAINT wake_token_id IF NOT EXISTS
  FOR (t:WakeToken) REQUIRE t.token_id IS UNIQUE;

// EtymRoot: unique per (language, form) pair
CREATE CONSTRAINT etym_root_id IF NOT EXISTS
  FOR (r:EtymRoot) REQUIRE r.root_id IS UNIQUE;

// SemanticField: unique field name
CREATE CONSTRAINT semantic_field_name IF NOT EXISTS
  FOR (s:SemanticField) REQUIRE s.name IS UNIQUE;

// VicoCycle: unique cycle number (1–4)
CREATE CONSTRAINT vico_cycle_number IF NOT EXISTS
  FOR (v:VicoCycle) REQUIRE v.cycle IS UNIQUE;

// KabbalahNode: unique sefirah name
CREATE CONSTRAINT kabbalah_node_name IF NOT EXISTS
  FOR (k:KabbalahNode) REQUIRE k.sefirah IS UNIQUE;

// Lens: unique lens identifier
CREATE CONSTRAINT lens_id IF NOT EXISTS
  FOR (l:Lens) REQUIRE l.lens_id IS UNIQUE;

// Language: unique ISO 639-1 code
CREATE CONSTRAINT language_code IF NOT EXISTS
  FOR (lang:Language) REQUIRE lang.code IS UNIQUE;

// ------------------------------------------------------------
// STANDARD INDEXES
// ------------------------------------------------------------

// WakeToken lookup by page and line (most common query pattern)
CREATE INDEX wake_token_page IF NOT EXISTS
  FOR (t:WakeToken) ON (t.page);

CREATE INDEX wake_token_page_line IF NOT EXISTS
  FOR (t:WakeToken) ON (t.page, t.line);

// WakeToken lookup by surface form (cross-book collation)
CREATE INDEX wake_token_surface IF NOT EXISTS
  FOR (t:WakeToken) ON (t.surface);

// EtymRoot lookup by language
CREATE INDEX etym_root_language IF NOT EXISTS
  FOR (r:EtymRoot) ON (r.language);

// EtymRoot lookup by form within a language
CREATE INDEX etym_root_form IF NOT EXISTS
  FOR (r:EtymRoot) ON (r.form);

// SemanticField index for fast name lookup
CREATE INDEX semantic_field_name_idx IF NOT EXISTS
  FOR (s:SemanticField) ON (s.name);

// Lens index for lookup by name and type
CREATE INDEX lens_name IF NOT EXISTS
  FOR (l:Lens) ON (l.name);

CREATE INDEX lens_type IF NOT EXISTS
  FOR (l:Lens) ON (l.type);

// ------------------------------------------------------------
// FULL-TEXT INDEXES
// ------------------------------------------------------------

// Full-text search over token surface forms and gloss notes
CREATE FULLTEXT INDEX wake_token_fulltext IF NOT EXISTS
  FOR (t:WakeToken) ON EACH [t.surface, t.notes];

// Full-text search over EtymRoot glosses
CREATE FULLTEXT INDEX etym_root_fulltext IF NOT EXISTS
  FOR (r:EtymRoot) ON EACH [r.form, r.gloss];

// ------------------------------------------------------------
// VECTOR INDEX (for anamnesis / embedding-based retrieval)
// ------------------------------------------------------------

// Residual-stream embeddings stored on WakeToken nodes.
// Dimensions = 4096 to match Llama-3 / Mistral hidden size.
// Adjust `vector.dimensions` when using smaller models.
CREATE VECTOR INDEX node_embeddings IF NOT EXISTS
  FOR (n:WakeToken) ON (n.embedding)
  OPTIONS {
    indexConfig: {
      `vector.dimensions`: 4096,
      `vector.similarity_function`: 'cosine'
    }
  };

// ------------------------------------------------------------
// NODE PROPERTY DESCRIPTIONS
// (Comments only — Neo4j does not enforce property schemas)
// ------------------------------------------------------------

// WakeToken properties:
//   token_id        string  — "{page}_{line}_{position}" e.g. "3_1_0"
//   surface         string  — exact text as it appears in the Wake
//   page            integer — Faber/Viking 1939 page number (3–628)
//   line            integer — 1-based line number within the page (1–25)
//   position        integer — 0-based token index within the line
//   book            integer — book number (1–4)
//   chapter         integer — chapter number within the book
//   notes           string  — free-text scholarly annotation
//   embedding       float[] — residual-stream vector (dimension = model hidden size)

// EtymRoot properties:
//   root_id         string  — "{language}_{form}" e.g. "la_rivus"
//   language        string  — ISO 639-1 / extended code (e.g. "non", "grc")
//   form            string  — root form in the source language
//   gloss           string  — English gloss

// Relationship: (WakeToken)-[:HAS_ROOT {confidence: float}]->(EtymRoot)
//   confidence      float   — annotator confidence 0.0–1.0

// SemanticField properties:
//   name            string  — canonical field name, lowercase hyphenated
//   description     string  — human-readable description
//   source          string  — which scholarly source defined this field

// Relationship: (WakeToken)-[:ACTIVATES {strength: float}]->(SemanticField)
//   strength        float   — activation intensity 0.0–1.0

// VicoCycle properties:
//   cycle           integer — 1=Theocratic, 2=Aristocratic, 3=Democratic, 4=Ricorso
//   name            string  — human-readable name
//   description     string  — description of this Viconian age

// Relationship: (WakeToken)-[:BELONGS_TO_CYCLE]->(VicoCycle)

// KabbalahNode properties:
//   sefirah         string  — sefirah name (e.g. "Malkuth", "Kether")
//   number          integer — position on the Tree of Life (1=Kether … 10=Malkuth)
//   description     string  — kabbalistic description

// Relationship: (WakeToken)-[:RESONATES_WITH {strength: float}]->(KabbalahNode)
//   strength        float   — resonance intensity 0.0–1.0

// Lens properties:
//   lens_id         string  — unique identifier slug
//   name            string  — display name
//   type            string  — "vico" | "kabbalah" | "etymological" | "geometric" | "custom"
//   description     string  — what interpretive framework this lens applies
//   foregrounds     string[] — list of SemanticField names foregrounded by this lens

// Relationship: (Lens)-[:FOREGROUNDS]->(SemanticField)
// Relationship: (Lens)-[:USES_PROBE]->(ProbeDefinition)

// Language properties:
//   code            string  — ISO 639-1 or extended code
//   name            string  — full language name in English
//   family          string  — language family (e.g. "Germanic", "Celtic", "Semitic")

// Token-sequence relationships:
// (WakeToken)-[:PRECEDES]->(WakeToken)        — adjacent tokens, same line
// (WakeToken)-[:PRECEDES_LINE]->(WakeToken)   — last token of line N → first token of line N+1
// (WakeToken)-[:ECHOES]->(WakeToken)          — long-range motif echo (from Hart / Tindall indices)
