# WAKE — Architecture & Code Walkthrough

This document explains what each component does, why it exists, and how the parts connect. Read it alongside the source to understand the full system.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [Data Flow](#2-data-flow)
3. [Corpus Layer](#3-corpus-layer)
4. [Knowledge Graph](#4-knowledge-graph)
5. [Lens System](#5-lens-system)
6. [Engine](#6-engine)
7. [Interpretability Layer](#7-interpretability-layer)
8. [API](#8-api)
9. [UI](#9-ui)
10. [Experiments](#10-experiments)
11. [Tests](#11-tests)
12. [Why Each Design Decision Was Made](#12-why-each-design-decision-was-made)

---

## 1. The Core Idea

The Wake is written so that every word can be parsed simultaneously as two or more languages carrying incompatible meanings. Standard NLP handles this by collapsing to a single parse. This project asks: does a large language model also collapse, or does it hold multiple parses at once in the geometry of its internal activations?

There are two failure modes that are indistinguishable in the model's output probabilities:

- **Parse collapse** — the model chooses one reading (e.g. English "river") and ignores the others (Latin "rivo", Norse "runa"). The residual stream lies near one interpretive cluster.
- **Superposition** — the model holds all readings active simultaneously. The residual stream lies between interpretive clusters, equidistant from several.

The project distinguishes these using *residual stream geometry*: projecting the model's internal vectors into a low-dimensional space where each interpretive framework ("lens") occupies a region, then asking whether a given token's vector falls inside one region or between several.

The secondary question — the "anamnesis protocol" — asks it in reverse: given the model's activation state for a passage, what frameworks does that state implicitly reach for? The passage becomes the query; the knowledge graph becomes the retrieval target.

---

## 2. Data Flow

```
Raw FW text (corpus/raw/)
        │
        ▼
WakeTokenizer               ← morpheme_db/ (81 multilingual morphemes)
        │  produces WakeToken list (surface, morpheme_candidates, language_candidates)
        ▼
Neo4j Knowledge Graph       ← McHugh annotations JSON
        │  WakeToken, Root, SemanticField, MythFigure, KabbalahNode nodes
        │  DERIVES_FROM, ACTIVATES, CO_ACTIVATES_WITH, PRECEDES edges
        ▼
Lens system                 ← lenses/ (6 frameworks)
        │  each lens: system_prompt + graph traversal + foregrounded fields
        ▼
MultiPassRunner             ← engine/multipass/runner.py
        │  runs the same passage N times, once per lens
        │  captures residual streams, attention patterns, entropy
        ▼
LanguageFieldProbe          ← interpretability/probes/language_probes.py
        │  19-class logistic regression, one per semantic field
        │  predicts field probabilities from residual stream vectors
        ▼
SuperpositionDetector       ← engine/superposition/detector.py
        │  classifies each token as superposed / collapsed / opaque
        ▼
ResidualStreamAnalyzer      ← interpretability/residual/stream_geometry.py
        │  PCA/UMAP of residual streams across lenses
        │  produces 2D geometry for SuperpositionMap
        ▼
AnamnesisProtocol           ← engine/anamnesis/protocol.py
        │  runs passage WITHOUT lens context
        │  fingerprint → lens similarity ranking → novel connection detection
        ▼
FastAPI (/api/wake/)        ← api/
        │
        ▼
React UI                    ← ui/src/
        (EntropyWaveform, LensComparison, SuperpositionMap,
         GraphExplorer, AnamnesisPanel)
```

---

## 3. Corpus Layer

```
corpus/
├── raw/                    FW text, one file per chapter, {page}.{line}: format
├── mcHugh/                 Roland McHugh's Annotations as structured JSON
│   └── sample_annotations.json
├── notebooks/              Joyce's working notebooks (digitized)
└── sources/                Vico, Bruno, Campbell, Hart, Tindall, Glasheen
```

**`corpus/raw/`** — The source text in a machine-readable format. Each line is prefixed `3.1: riverrun, past Eve and Adam's...` so any downstream tool can recover page and line numbers without separate index files.

**`corpus/mcHugh/sample_annotations.json`** — The McHugh annotation format used throughout the pipeline. Each entry covers one surface token and may contain multiple etymological roots (with language, meaning, confidence), Viconian cycle attributions, and cross-references to source figures. This JSON becomes the primary input to the Neo4j ingestion pipeline.

The format was chosen because McHugh's annotations are the most comprehensive scholarly account of the Wake's word-by-word sources. Structuring them as JSON makes them queryable and allows the graph ingestion pipeline to operate without natural-language parsing.

---

## 4. Knowledge Graph

```
graph/
├── schema/wake_schema.cypher       Node/edge type definitions + indexes
├── ingest/
│   ├── mcHugh_ingest.py            McHugh JSON → Neo4j
│   └── raw_text_ingest.py          Raw text → WakeToken nodes + PRECEDES edges
├── queries/
│   ├── etymology.cypher            Etymology chain queries
│   ├── semantic_fields.cypher      Field co-activation queries
│   └── paths.cypher                Shortest-path + neighborhood queries
└── probes/
    └── graph_probe_definitions.py  GraphProbeDefinition objects
```

### Node types

| Node | Purpose |
|---|---|
| `WakeToken` | A surface token as it appears in the text (page, line, surface, entropy) |
| `Root` | An etymological root in any language (form, language ISO code, meaning, period) |
| `SemanticField` | A named cluster of meaning (e.g. `cyclic_return`, `divine_thunder`) |
| `MythFigure` | A mythological figure from any tradition |
| `KabbalahNode` | A sefirah or Kabbalistic concept |
| `SourceEntity` | A person, text, or concept from Joyce's documented sources |
| `Lens` | A registered lens framework (linked to SemanticFields it foregrounds) |

### Edge types

| Edge | Meaning |
|---|---|
| `DERIVES_FROM` | Token → Root; the etymological derivation, with confidence and mechanism |
| `COGNATE_WITH` | Root → Root; cross-language cognate, false friend, or common ancestor |
| `ACTIVATES` | Token → SemanticField; the field this token lights up |
| `PRECEDES` | Token → Token; sequential adjacency in the text |
| `CO_ACTIVATES_WITH` | Token → Token; both activate under a specific lens |
| `RESONATES_WITH` | Token → KabbalahNode; Kabbalistic resonance |
| `FOREGROUNDS` | Lens → SemanticField; which fields the lens attends to |

### Why Neo4j

The knowledge structure is inherently a graph: words derive from roots, roots are cognate with other roots across languages, tokens co-activate under lenses, and mythological figures are connected through shared attributes. A relational database would require deeply nested joins to answer "find all tokens that share a root with 'riverrun' and also activate the `cyclic_return` field." In Cypher this is one query. The vector index (Neo4j 5.x) enables the anamnesis protocol to query the graph using embedding similarity rather than keyword matching.

### `graph/ingest/mcHugh_ingest.py`

Reads the structured McHugh JSON and writes to Neo4j using MERGE (idempotent — safe to rerun). Handles two annotation types: etymological roots (creates `Root` nodes and `DERIVES_FROM` edges) and Viconian source attributions (creates `SemanticField` nodes and `ACTIVATES` edges). Runs in batches; logs progress.

### `graph/ingest/raw_text_ingest.py`

Parses the `{page}.{line}: text` format, splits surface tokens with a regex that strips punctuation attachments (so "riverrun," becomes "riverrun"), and writes `WakeToken` nodes in batches of 500. Then writes `PRECEDES` edges to capture sequential adjacency, which is used for co-activation analysis and path-finding.

### `graph/probes/graph_probe_definitions.py`

`GraphProbeDefinition` objects that define structured database queries for five structural properties:
- `etymology_depth_probe` — how many hops from surface to deepest root
- `multilingual_breadth_probe` — how many distinct languages are attributed
- `semantic_field_density_probe` — how many SemanticFields activate at this token
- `vico_cycle_probe` — which Vico cycle (1–4, ricorso) the token belongs to
- `kabbalah_resonance_probe` — which sefirah this token resonates with

These are not ML probes — they are graph-based structural measurements that supplement the linear probes in the interpretability layer.

---

## 5. Lens System

```
lenses/
├── base.py             LensConfig dataclass + Lens ABC
├── viconian.py
├── kabbalistic.py
├── freudian.py
├── irish_mythology.py
├── norse.py
├── brunian.py
└── registry.py         ALL_LENSES dict, ComposedLens, factory functions
```

### `lenses/base.py`

`LensConfig` is a dataclass holding everything needed to run one interpretive framework:

```python
@dataclass
class LensConfig:
    name: str                          # "viconian"
    system_prompt: str                 # full prompt priming the model
    foregrounded_fields: List[str]     # which SemanticFields to weight up
    graph_traversal: Dict[str, Any]    # which node/edge types to traverse
    attention_priors: Dict[str, float] # expected attention head activity
    probe_targets: List[str]           # which linear probes to activate
```

`Lens` is an abstract base class. `_build_config()` is the only method subclasses must implement. The base class provides `get_context_for_passage()`, which:
1. Queries the knowledge graph for nodes matching this lens's foregrounded fields
2. Formats those nodes into a context block prepended to the raw passage

This means the model receives different context depending on which lens is active — Viconian context foregrounds cycle and thunder-word nodes, Kabbalistic context foregrounds sefirot and divine-name nodes, and so on.

### The six lenses

| Lens | Framework | Key foregrounded fields |
|---|---|---|
| `ViconianLens` | Vico's four-age historical cycle (theocratic → heroic → human → ricorso) | `cyclic_return`, `divine_thunder`, `ricorso`, `etymology_poetic` |
| `KabbalisticLens` | Lurianic Kabbalah: tzimtzum, sefirot, PaRDeS hermeneutics, tikkun olam | `divine_name`, `exile_return`, `hidden_revelation`, `kabbalistic_divine` |
| `FreudianLens` | Dreamwork (condensation/Verdichtung, displacement/Verschiebung), repetition compulsion, the uncanny | `dreamwork_condensation`, `unconscious_return`, `father_complex`, `primal_scene` |
| `IrishMythologyLens` | Four Cycles, HCE as Finn, ALP as Liffey, Book of Kells, Ogham, aisling | `celtic_myth`, `hce_archetype`, `alp_river`, `twin_opposition` |
| `NorseLens` | Eddas, Odin, runes, kennings, Ragnarök, Clontarf, Dublin Norse | `norse_cosmology`, `rune_encoding`, `kenning_compound`, `ragnarok_cycle` |
| `BrunianLens` | Coincidentia oppositorum, Ars Memorativa, hermetic tradition, the Nolan | `coincidentia_oppositorum`, `memory_theatre`, `hermetic_tradition`, `nolan_martyr` |

Each system prompt is several paragraphs of scholarly specifics — not a generic instruction but a detailed epistemic frame. The Kabbalistic lens prompt, for example, includes Scholem's connection to the Zohar via Joyce's Jesuit education, the four-headed shin, and the instruction to treat apparent nonsense as concealed Sod-layer content.

### `lenses/registry.py`

`ALL_LENSES` maps name strings to lens classes. `get_lens(name, graph_client=None)` instantiates one. `compose_lenses(names, weights)` builds a `ComposedLens`.

`ComposedLens` blends multiple lenses for studying *interference patterns* between incompatible frameworks. It:
- Normalises weights to sum to 1.0
- Produces attributed context blocks (one section per lens, labelled with weight)
- Computes a `superposition_score()` based on the Shannon entropy of the graph-node distribution across lenses — higher entropy means more evenly-spread activation, a proxy for genuine multi-frame reading

---

## 6. Engine

```
engine/
├── tokenizer/
│   ├── wake_tokenizer.py
│   └── morpheme_db/seed_morphemes.json
├── model/
│   └── wake_model.py
├── multipass/
│   └── runner.py
├── anamnesis/
│   └── protocol.py
└── superposition/
    └── detector.py
```

### `engine/tokenizer/wake_tokenizer.py`

Standard tokenizers destroy portmanteau structure. `WakeTokenizer` operates in three passes:

**Pass 1 — Surface splitting** (`_split_surface`): Unicode-aware regex that splits on whitespace and strips punctuation attachments. "riverrun," → "riverrun". Returns `SurfaceToken(value, span)` namedtuples so character offsets are preserved.

**Pass 2 — Portmanteau decomposition** (`_decompose_portmanteau`): Dynamic programming over the morpheme database. For a token of length n, fills a table `dp[0..n]` where `dp[j]` holds all reachable segmentations ending at character j, along with their running confidence score. The DP uses a length-weighted running average: longer morpheme matches are preferred. Returns the top 5 segmentations above a 0.1 confidence threshold.

Example: "riverrun" → ["river+run" (0.85), "river+rune" (0.72), "riv+error+un" (0.14)]

**Pass 3 — Language attribution** (`_attribute_languages`): Looks up each candidate morpheme in the morpheme DB and collects the union of attributed ISO language codes, deduplicated and ordered by confidence.

The result is a `WakeToken` dataclass: surface, character span, page, line, morpheme candidates (ranked), language candidates (ranked), portmanteau components, and a Neo4j node ID if the token is already in the graph.

**`morpheme_db/seed_morphemes.json`** contains 81 entries covering FW pages 3–10 across EN, LA, IT, DE, GA (Irish Gaelic), NON (Old Norse), HE (Hebrew), FR, EL (Greek). Each entry has confidence, language, meaning, period, and cognates.

### `engine/model/wake_model.py`

Wraps TransformerLens's `HookedTransformer` with Wake-specific convenience methods:

- `from_pretrained(model_name, hf_token)` — handles HuggingFace auth, dtype selection, device detection
- `forward_with_cache(tokens, names_filter)` — full forward pass returning logits + activation cache. The `names_filter` captures: embedding, positional embedding, residual stream pre/post at every layer, attention patterns, attention output (z), MLP output
- `get_token_entropy(logits)` — per-token Shannon entropy over the vocabulary
- `get_attention_patterns(cache, layer)` — shape `[heads, seq, seq]`
- `get_residual_stream(cache, layer, position)` — single vector `[d_model]`
- `get_mlp_output(cache, layer, position)` — MLP contribution at a position
- `get_head_output(cache, layer, head, position)` — individual head contribution
- `decode_top_k(logits, k)` — top-k predicted tokens with probabilities

Recommended model: `meta-llama/Llama-3.1-70B` (strong multilingual, large enough for superposition effects to be geometrically separable). Development: `meta-llama/Llama-3.1-8B`.

### `engine/multipass/runner.py`

The heart of the experiment loop. `MultiPassRunner.run()`:

1. For each lens, prepends the lens context to the raw passage and tokenizes
2. Runs `model.forward_with_cache()` to get logits and all activations
3. Extracts residual streams at 5 layers: early (0), 25%, 50%, 75%, late (n-1)
4. Extracts attention patterns at those same layers
5. Finds where the passage tokens begin in the input (skipping the context prefix) via `_find_passage_start_idx()`
6. Runs the language field probes at each layer for each passage token position
7. Runs `SuperpositionDetector.analyze()` at the middle layer for each token
8. Packages everything into a `PassResult` per lens

After all lenses, `_compute_contrastive_result()`:
- Builds the **divergence map**: `[n_lenses × seq_len]` matrix of per-token entropy per lens. Columns where lenses disagree are high-signal positions.
- Builds the **agreement matrix**: `[n_lenses × n_lenses]` matrix of mean cosine similarity between lens pairs' residual streams at the middle layer. Values near 1.0 mean the two lenses produce nearly identical internal representations; values near 0 mean they diverge substantially.
- Identifies **superposition tokens**: positions where the mean superposition score across lenses exceeds 1.5

`run_batch()` processes a list of passages sequentially. `save_result()` serializes to JSON (numpy arrays become lists for portability).

### `engine/anamnesis/protocol.py`

Implements the inverse retrieval protocol. Instead of asking "what does this lens extract from the passage?", it asks "what does the passage reach for on its own?".

`AnamnesisProtocol.run()`:

1. Runs the passage through the model with NO lens context — bare model, no system prompt
2. Captures the residual stream at the 65th percentile layer (past syntactic processing, into semantic integration)
3. Averages across all passage token positions → single fingerprint vector `[d_model]`
4. Computes cosine similarity between the fingerprint and pre-computed lens centroids (mean residual stream of passages well-explained by each lens)
5. Queries the Neo4j vector index: which graph nodes have embeddings closest to this fingerprint?
6. `_find_novel_connections()`: finds shortest paths in the KG between top-activated nodes, then filters to retain only paths whose relationship type sequences are NOT among the foregrounded fields of the top-3 lenses. These are the "novel connections" — semantic relationships the passage activates that none of the current lenses would predict.
7. `_generate_narrative()`: human-readable summary of which lenses were implicitly activated and how many novel connections were found

The novel connections from Experiment 3 are the primary scholarly deliverable: clusters of novel connection types across many passages define candidate new lenses — frameworks the Wake demands that haven't been formally articulated.

### `engine/superposition/detector.py`

`SuperpositionDetector.analyze()` takes probe results and the residual stream for one token and returns:

- `is_superposed` (bool): True if `n_active_fields > 2` AND `dominant_prob < 0.75`
- `active_fields`: dict of fields above the 0.35 threshold
- `dominant_field` + `dominant_probability`
- `n_active_fields`
- `residual_decomposition`: projection of the residual stream onto each probe direction (the learned logistic regression weight vector)
- `superposition_score`: `n_active_fields × (1 − dominant_prob)` — higher means more genuine multi-parse holding

`classify()` returns one of three strings:
- `"superposed"` — multiple frames above threshold, no single dominant
- `"collapsed"` — one frame clearly dominant (>0.75)
- `"opaque"` — all fields below 0.2; the model has no grip on this token at all

`compute_superposition_score()` is a standalone function using softmax entropy of probe probabilities, normalized to [0, 1]. Used in `batch_analyze()` for efficient vectorized processing.

---

## 7. Interpretability Layer

```
interpretability/
├── probes/
│   └── language_probes.py      LanguageFieldProbe, ProbeResult, SuperpositionDetector
├── attention/
│   └── head_analysis.py        AttentionHeadAnalyzer, AttentionHeadProfile
├── residual/
│   └── stream_geometry.py      ResidualStreamAnalyzer, StreamGeometryResult
└── circuits/
    └── path_patching.py        PathPatcher, PatchingResult
```

### `interpretability/probes/language_probes.py`

`LanguageFieldProbe` is a set of 19 binary logistic regression classifiers, one per semantic field. Each is trained one-vs-rest: "does this residual stream activation express `cyclic_return`? Yes or no?"

The 19 fields correspond to `SemanticField` nodes in the knowledge graph and cover the main interpretive dimensions: water/ALP, Viconian ages, Freudian mechanisms, guilt/fall, exile, body, time, language plurality, sexuality, death/rebirth, family drama, Kabbalah.

`fit(activations, labels)` trains all 19 probes on labelled residual stream samples. Training data comes from known monolingual corpora for language detection (Experiment 0) and from scholarly annotations for semantic field detection.

`predict(activation)` returns a `ProbeResult` with `predictions` (dict of field → probability), `top_field`, and `confidence`. Crucially, multiple fields can score above threshold simultaneously — that's the superposition signal.

`get_probe_direction(field)` returns the logistic regression weight vector for a field — the linear direction in residual space that the probe learned to detect. These directions are used in `SuperpositionDetector.analyze()` to decompose the residual stream into field components.

`save()/load()` use pickle for persistence between runs.

### `interpretability/attention/head_analysis.py`

`AttentionHeadAnalyzer.profile_head()` characterizes an attention head by:
- **Entropy**: low entropy = sharp, focused attention (name movers); high entropy = diffuse (general context)
- **Previous-token detection**: if `attention[i, i-1] > 0.5` on average, this is a previous-token head (involved in sequential prediction)
- **Induction detection**: induction heads attend back to the position of the previous occurrence of the current token; detected by bigram matching (simplified heuristic in this implementation)

`find_divergent_heads()` computes the mean KL divergence of attention distributions between two lens passes for each head. Heads that diverge most when the lens changes are the ones whose behavior is most lens-dependent — candidates for circuit analysis.

### `interpretability/residual/stream_geometry.py`

`ResidualStreamAnalyzer.compute_geometry()` takes a dict of `{lens_name: [seq_len, d_model]}` arrays and:

1. Concatenates all lens × position vectors
2. Fits PCA (or UMAP if available) to 2D
3. Computes per-lens centroids in 2D
4. Computes cluster separation: mean inter-centroid distance / mean intra-lens spread
5. Identifies superposition tokens: positions where the point is within threshold distance of multiple centroids simultaneously (`is_superposed()`)

`StreamGeometryResult` holds everything the UI's SuperpositionMap needs: 2D coordinates, lens centroids, superposition token indices, cluster separation score.

`compute_umap()` tries to import `umap-learn` and falls back to PCA if unavailable. UMAP is preferred because it better preserves local neighborhood structure, making the cluster/between-cluster distinction sharper.

### `interpretability/circuits/path_patching.py`

`PathPatcher` implements causal intervention: run the model with corrupted input (different lens context), but *replace* the activation at one specific `(layer, position)` with the clean-run activation. Measure the effect on a metric (typically logit difference on a target token).

`sweep_layers()` runs this for every layer at a fixed position, revealing which layers causally mediate the lens-dependent behavior. Layers with high normalized effect are part of the circuit responsible for the semantic difference between two lens readings.

`sweep_grid()` runs the full `(layer × position)` grid — expensive but gives a complete causal map of which components are responsible for a given behavior difference.

---

## 8. API

```
api/
├── main.py                 FastAPI app, CORS, lifespan, router includes
├── state.py                AppState singleton (model, graph, lenses, etc.)
├── routes/
│   ├── analysis.py         POST /analyse, GET /health, GET /lenses
│   └── graph.py            GET /graph/node, /graph/path, /graph/neighborhood
└── schemas/
    ├── requests.py         Pydantic v2 request models
    └── responses.py        Pydantic v2 response models
```

### `api/state.py`

`AppState` manages all shared resources with lazy initialization. It reads from environment variables (`NEO4J_URI`, `MODEL_NAME`, `HF_TOKEN`, `PROBES_DIR`) and only initializes components that are configured. This means the API can start and serve `/health` and `/lenses` even when the GPU model and Neo4j are not available — important for development and testing.

An `asyncio.Lock` prevents double-initialization under concurrent requests. `shutdown()` closes the Neo4j driver and releases GPU memory.

### `api/routes/analysis.py`

`POST /api/wake/analyse` is the core endpoint. It:
1. Tokenizes the passage with `WakeTokenizer`
2. Delegates to `MultiPassRunner.run()` for the full contrastive analysis
3. Optionally runs `AnamnesisProtocol.run()` if `run_anamnesis=True`
4. Formats the result into the `ContrastiveResponse` schema

`GET /api/wake/health` reports whether the model, graph, and lenses are initialized.

`GET /api/wake/lenses` lists available lens names — used by the UI to populate the lens selection checkboxes.

`POST /api/wake/lens/compose` creates a composed lens and returns its metadata, enabling the UI to define blended lens configurations.

### `api/routes/graph.py`

Graph exploration endpoints for the UI's `GraphExplorer`:

- `GET /graph/node/{surface}` — returns the full subgraph (node + all edges + neighbors) for a Wake token
- `GET /graph/path` — Cypher `shortestPath` between two tokens
- `GET /graph/neighborhood/{surface}?hops=2` — 2-hop neighborhood, grouped by node type
- `GET /graph/semantic-fields` — all semantic fields with token counts, sorted by activation strength

### Request/response schemas (`api/schemas/`)

All schemas use Pydantic v2. Key validation:
- `PassageRequest`: page ≥ 1, line ≥ 1; lenses defaults to the 4 core ones
- `GraphPathRequest`: `from_surface` and `to_surface` must differ; max_hops 1–10
- `LensComposeRequest`: weights must have same length as lenses, must sum to 1.0

---

## 9. UI

```
ui/src/
├── App.tsx                         Root layout, panel grid, loading overlay
├── store/passageStore.ts           Zustand store (single source of truth)
├── api/client.ts                   Typed Axios client
└── components/
    ├── PassageInput/               Passage form + lens selector
    ├── EntropyWaveform/            D3 per-token entropy waveform
    ├── GraphExplorer/              D3 force-directed knowledge graph
    ├── LensComparison/             Plotly divergence heatmap
    ├── SuperpositionMap/           Plotly PCA scatter (residual geometry)
    ├── AnamnesisPanel/             Anamnesis result display
    └── common/                     LoadingSpinner, ErrorBanner
```

### `store/passageStore.ts`

Single Zustand store holds the current `ContrastiveResult`, loading state, error, selected token position, and selected lens set. All components read from and write to this store; there is no local component state for analysis data. The `analyzePassage()` action calls the API, updates the store on success, and sets `error` on failure.

### `components/EntropyWaveform/`

D3 line chart showing per-token entropy as a waveform. One trace per lens, color-coded using D3's Tableau10 categorical scale. Superposition token positions are marked with vertical dashed lines and diamond markers. Uses `ResizeObserver` so the chart reflows when the panel is resized. Clicking a token position calls `setSelectedPosition()` in the store, which is then read by `SuperpositionMap` to highlight that position.

This is the first signal you look at: high entropy at a token means the model is uncertain about the next token. If all lenses produce high entropy at the same position, that position is genuinely ambiguous regardless of framing. If only some lenses produce high entropy, the ambiguity is lens-dependent.

### `components/LensComparison/`

Plotly heatmap. Rows are lenses, columns are token positions. Color encodes divergence (dark = agreement between this lens and the mean, bright = divergence). The `RdBu_r` colorscale makes disagreement immediately visible as red.

Clicking a cell opens a detail modal showing side-by-side attention patterns for the two lenses with the highest divergence at that token — the specific attention heads that are behaving differently. This is how you find the circuit-level explanation for a superposition signal.

### `components/SuperpositionMap/`

Plotly scatter showing PCA-projected residual stream vectors. Each point is one `(token, lens)` pair. Color encodes lens; shape encodes token (circle, square, diamond, cross, x for the first 5 distinct tokens, cycling). Superposition tokens are rendered larger with amber color and a star annotation showing the token string.

The geometry tells the main story:
- Points from different lenses *clustered together* → the model gives the same representation regardless of lens context (lens-invariant, probably simple tokens)
- Points from different lenses *spread into distinct clusters* → the model collapses to a single reading per lens (parse collapse)
- Points falling *between* clusters → the model holds multiple readings simultaneously (genuine superposition, the PKD result)

### `components/GraphExplorer/`

D3 force-directed simulation of the local knowledge graph around the current passage's tokens. Nodes are colored by type (WakeToken=blue, Root=green, SemanticField=amber, MythFigure=purple, KabbalahNode=gold). Edge width is proportional to the `confidence` property. Hovering shows node type, surface/name, and meaning. Clicking a token node expands it to its 2-hop neighborhood via an API call to `/graph/node/{surface}`. A filter bar shows/hides node types.

### `components/AnamnesisPanel/`

Collapsible drawer showing the anamnesis result for the current passage:

1. **Lens similarity bars** — horizontal bar chart of which lenses the bare passage fingerprint is most similar to. High similarity to "brunian" without a Brunian lens context means the text is implicitly calling for that framework.
2. **Activated graph nodes** — chip badges for the top KG nodes retrieved by fingerprint similarity. These are the concepts the model was reaching toward even without being told to.
3. **Novel connections table** — paths between activated nodes that no current lens foregrounds. Novelty score > 0.7 is highlighted. These are the candidates for new lens definitions.
4. **Narrative** — the `_generate_narrative()` string from the protocol.

---

## 10. Experiments

```
experiments/
├── README.md
├── E0_baseline_calibration.ipynb
├── E1_entropy_waveform.ipynb
├── E2_superposition_discrimination.ipynb
└── E3_anamnesis.ipynb
```

All notebooks have a `USE_REAL_MODEL = False` flag at the top. With the flag off, they generate synthetic data from numpy random seeds so the analysis pipeline, visualizations, and statistics can be explored without a GPU.

**E0 — Baseline Calibration**: Runs monolingual, monosemous passages (simple English, Latin, German, Irish, Hebrew) through the probe system. Expected: superposition scores near 0, probe predictions correct for the known language, lens agreement high. Establishes the detection floor.

**E1 — Entropy Waveform**: Iterates corpus pages, runs the model on each passage, collects per-token entropy, plots the full waveform. Expected: non-uniform distribution with identifiable high-entropy clusters. These clusters are the experimental targets for E2.

**E2 — Superposition Discrimination**: Takes the high-entropy passages from E1 and runs all 6 lenses. Computes PCA residual geometry. Classifies each token as superposed / collapsed / opaque. Runs a chi-squared test to confirm the distribution is non-uniform. The `SuperpositionMap` plot is the primary empirical output.

**E3 — Anamnesis Mapping**: Runs the anamnesis protocol on 15 passages spanning all four books. Collects novel connections, clusters them by path type using UMAP/PCA, and reports candidate new lenses. Passages from Book IV (pp. 593–628) are structurally different from Book I — the final monologue of ALP is expected to activate different implicit frameworks.

---

## 11. Tests

```
tests/
├── conftest.py                     Shared fixtures
├── test_tokenizer.py               WakeTokenizer unit tests
├── test_lenses.py                  Lens system unit tests
├── test_superposition.py           SuperpositionDetector unit tests
└── test_api.py                     FastAPI endpoint tests (TestClient)
```

All tests run without GPU, Neo4j, or network access. Fixtures in `conftest.py` provide:

- `sample_passage` — the opening of the Wake
- `mock_graph` — `MagicMock` with `.run()` returning `[]`
- `mock_wake_token_list` — pre-built list of `WakeToken` objects
- `minimal_morpheme_db` — small in-memory dict for tokenizer testing

Key test cases:
- All 6 lenses instantiate, have non-empty system prompts >100 chars, and have probe targets
- `ComposedLens` weights normalize to 1.0
- `SuperpositionDetector` correctly classifies a case with one dominant field (>0.9) as `collapsed` and a case with four fields all above 0.4 as `superposed`
- The health endpoint returns 200 even when model and graph are not initialized (lazy degradation)

---

## 12. Why Each Design Decision Was Made

**Why TransformerLens instead of plain Hugging Face transformers?**
TransformerLens provides named hook points at every layer and sublayer, residual stream decomposition, and the `run_with_cache()` interface that captures all activations in one forward pass. Without this, getting residual streams requires monkey-patching forward hooks, which breaks easily across model architectures.

**Why Neo4j instead of a simpler graph library (networkx)?**
The knowledge graph needs to be queryable at inference time — the lens system fetches graph context for every passage during analysis. NetworkX loads the full graph into RAM and has no query language. Neo4j supports Cypher (expressive, index-backed), runs as a service (same instance used by the API and offline experiments), and has vector index support for the anamnesis fingerprint queries in one unified system.

**Why one-vs-rest logistic regression for the probes?**
Logistic regression is interpretable: the weight vector is the probe direction in residual space, which can be used to decompose the residual stream into field contributions. More powerful classifiers (e.g. MLP probes) would give better accuracy but lose the geometric interpretability that is central to the superposition analysis.

**Why 6 lenses rather than 2 or 20?**
The 6 lenses cover the main documented scholarly frameworks for reading the Wake. Fewer than 4 would miss the multilingual structure. More than 8 would make the divergence map hard to read and the agreement matrix expensive. The 6 chosen are all documented in Joyce scholarship as intentional sources, not speculative: Vico is explicit in the text, Bruno is documented in Joyce's library, the Kabbalistic material is attested by Scholem's correspondence, and so on.

**Why `ComposedLens` for interference patterns?**
Some of the most interesting questions are about what happens when two *incompatible* frameworks are held simultaneously. Vico's historical cycles and Kabbalistic tikkun olam both involve repair and return, but through completely different mechanisms. Composing these lenses forces the model to integrate contexts that can't be made consistent — the resulting superposition signal should be stronger at tokens that participate in both frameworks.

**Why anamnesis runs the model without any lens context?**
The lens context shapes the model's behavior by priming a specific reading. If you want to know what the text itself demands — not what you're telling the model to find — you have to run it without priming. The fingerprint you get is the model's default internal response to the text, unfiltered by a pre-selected framework.

**Why PCA for the SuperpositionMap rather than raw residual streams?**
`d_model` for Llama-70B is 8192. You cannot visualize 8192-dimensional space. PCA gives a linear projection that maximally preserves variance — the largest geometric differences between lens residual streams. UMAP gives a non-linear projection that better preserves neighborhood structure. Both are available; PCA is the default because it requires no additional dependencies and is reproducible across runs without a random seed.
