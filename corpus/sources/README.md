# Primary Sources

This directory holds metadata and index files for the scholarly sources used in
WAKE annotations, interpretability probes, and semantic-field definitions.

No copyrighted text is stored here. The files record bibliographic data,
structural outlines, and cross-reference mappings that support the ingest
pipeline.

## Canonical Bibliography

| Short Name | Full Reference |
|---|---|
| **Vico** | Giambattista Vico, *La Scienza Nuova* (The New Science), 3rd ed. Naples, 1744. Trans. Bergin & Fisch, Cornell UP, 1948. |
| **Bruno** | Giordano Bruno, *De la Causa, Principio et Uno* (On Cause, Principle, and Unity), 1584. Trans. Robert de Lucca & Richard J. Blackwell, Cambridge UP, 1998. |
| **CampbellRobinson** | Joseph Campbell & Henry Morton Robinson, *A Skeleton Key to Finnegans Wake*, Harcourt Brace, 1944. Repr. Penguin, 1977. |
| **Hart** | Clive Hart, *Structure and Motif in Finnegans Wake*, Northwestern UP, 1962. |
| **Tindall** | William York Tindall, *A Reader's Guide to Finnegans Wake*, Farrar Straus & Giroux, 1969. |
| **McHugh** | Roland McHugh, *Annotations to Finnegans Wake*, 3rd ed., Johns Hopkins UP, 2006. |
| **Glasheen** | Adaline Glasheen, *Third Census of Finnegans Wake*, Northwestern UP, 1977. |

## Role in the WAKE Pipeline

| Source | Used For |
|---|---|
| Vico | `vico_cycle` field (1=Theocratic, 2=Aristocratic, 3=Democratic, 4=Ricorso); SemanticField nodes for Viconian themes |
| Bruno | Coincidentia oppositorum motifs; Bruno dialectic SemanticField nodes |
| CampbellRobinson | High-level structural cross-references; narrative episode mapping |
| Hart | Motif-thread cross-reference edges (`ECHOES` relationships in Neo4j) |
| Tindall | Thematic index entries mapped to SemanticField nodes |
| McHugh | Primary source for `roots[]`, `languages[]`, `semantic_fields[]`, and `notes` in every annotation JSON |
| Glasheen | Character identification; cross-reference entries for proper nouns |

## File Naming Convention

Structured index files in this directory follow the pattern:

```
{short_name}_index.json
```

Each index file maps page/line references to the corresponding entry in the
source text, enabling bidirectional lookup between Wake passages and
scholarship.
