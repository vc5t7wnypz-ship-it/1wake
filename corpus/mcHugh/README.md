# McHugh Annotations

This directory holds machine-readable versions of Roland McHugh's
*Annotations to Finnegans Wake* (3rd ed., Johns Hopkins, 2006), encoded in
a JSON schema that the ingest pipeline (`graph/ingest/mcHugh_ingest.py`) reads
directly into Neo4j.

## JSON Schema

Each annotation file covers one page of the Wake.  The top-level structure is:

```json
{
  "page": 3,
  "annotations": [
    {
      "line": 1,
      "surface": "riverrun",
      "languages": ["en", "it", "non", "la"],
      "roots": [
        {
          "language": "en",
          "form": "river",
          "gloss": "a natural watercourse",
          "confidence": 1.0
        },
        {
          "language": "it",
          "form": "rivo",
          "gloss": "stream, brook",
          "confidence": 0.9
        }
      ],
      "semantic_fields": ["water", "flow", "cycle"],
      "vico_cycle": 4,
      "kabbalah_sefirah": null,
      "cross_references": ["Anna Livia Plurabelle", "Liffey"],
      "notes": "Opening word; Vico's ricorso, the river completing the cycle."
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `page` | integer | Faber/Viking page number |
| `annotations` | array | One entry per annotated surface token on the page |
| `annotations[].line` | integer | 1-based line number within the page |
| `annotations[].surface` | string | Surface form as it appears in the text (may include trailing punctuation) |
| `annotations[].languages` | string[] | ISO 639-1 codes for all languages present in this token |
| `annotations[].roots` | object[] | Etymological decomposition (see sub-fields below) |
| `annotations[].roots[].language` | string | ISO 639-1 code of the source language |
| `annotations[].roots[].form` | string | Root/cognate form in the source language |
| `annotations[].roots[].gloss` | string | English gloss for the root |
| `annotations[].roots[].confidence` | float | Annotator confidence: 1.0 = certain, 0.5 = speculative |
| `annotations[].semantic_fields` | string[] | Thematic/semantic categories active at this token |
| `annotations[].vico_cycle` | integer\|null | Vico's historical cycle (1=theocratic, 2=aristocratic, 3=democratic, 4=ricorso) |
| `annotations[].kabbalah_sefirah` | string\|null | Kabbalistic sefirah resonance (e.g. "Malkuth", "Kether") |
| `annotations[].cross_references` | string[] | Intra-Wake motifs, characters, or places this token connects to |
| `annotations[].notes` | string | Free-text scholarly commentary |

## Language Code Conventions

Standard ISO 639-1 codes are used with the following additions:

| Code | Language |
|---|---|
| `ga` | Irish (Gaelic) |
| `non` | Old Norse |
| `la` | Latin |
| `grc` | Ancient Greek |
| `he` | Hebrew |
| `ar` | Arabic |
| `sa` | Sanskrit |
| `cy` | Welsh |
| `br` | Breton |

## File Naming

One file per page: `page_{NNN}.json` where NNN is zero-padded to three digits,
e.g. `page_003.json` for page 3.

## Source Note

McHugh's annotations are protected by copyright. The JSON files in this
directory encode only the structural relationships (language attributions,
etymological roots, semantic fields) derived from scholarship, not verbatim
text from the book. Consult the original *Annotations* for full commentary.
