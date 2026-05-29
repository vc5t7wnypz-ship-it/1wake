# Corpus Raw Text

This directory holds the plain-text source of *Finnegans Wake* (Joyce, 1939).

## File Naming

| Pattern | Contents |
|---|---|
| `fw_full.txt` | Complete text of all 17 chapters, concatenated in reading order |
| `fw_book{N}_chapter{M}.txt` | Single chapter; N is book number (1–4), M is chapter number within that book |

Example: `fw_book1_chapter1.txt` is I.1 (pages 3–29 in the standard Viking/Faber edition).

## Line Format

Every line in every file follows this prefix convention:

```
{page}.{line}: {text}
```

- `{page}` — page number in the standard 1939 Faber edition (3–628)
- `{line}` — 1-based line number within that page (1–25 for most pages)
- `{text}` — the exact line of the Wake as it appears in the printed text,
  including all original punctuation, capitalisation, and spacing

### Example

```
3.1: riverrun, past Eve and Adam's, from swerve of shore to bend
3.2: of bay, brings us by a commodius vicus of recirculation back to
3.3: Howth Castle and Environs.
```

Lines that carry over a sentence from the previous line are prefixed normally;
there is no line-continuation marker. Blank lines in the printed text are
represented as lines containing only the prefix and a single space.

## Source Note

The raw text is **not** committed to this repository for copyright reasons.
Obtain a scan or licensed digital edition and convert it to this format using
`corpus/tools/format_raw.py` (forthcoming). Page and line numbers should match
the standard Viking Press (New York) or Faber & Faber (London) pagination.
