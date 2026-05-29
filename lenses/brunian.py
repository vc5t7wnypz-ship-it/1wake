"""
Brunian lens — reads Finnegans Wake through the philosophy of Giordano Bruno
(1548–1600), martyr of the infinite universe and apostle of the coincidence
of opposites.

Joyce's documented debt to Bruno is substantial: he owned a copy of *De la
Causa, Principio et Uno* (1584), wrote a 1903 essay praising Bruno in the
*Daily Express* ("The Day of the Rabblement"), and identified Bruno's
doctrine of coincidentia oppositorum as the philosophical key to the Shem/
Shaun opposition.  In a 1924 letter, Joyce noted that "Browne and Nolan"
(the Dublin booksellers whose name he adapted in the Wake as "Browne and
Nolan") encode Bruno of Nola — "the Nolan" being Bruno's own sobriquet.
"""

from __future__ import annotations

from typing import Any

from .base import Lens, LensConfig

_SYSTEM_PROMPT = """\
You are reading Finnegans Wake through the philosophy of Giordano Bruno
of Nola (1548–1600): his doctrine of the coincidence of opposites, his
infinite universe, his memory theatre, and his martyrdom.

COINCIDENTIA OPPOSITORUM — THE COINCIDENCE OF OPPOSITES
─────────────────────────────────────────────────────────
Bruno's central philosophical doctrine, inherited from Nicholas of Cusa
(*De Docta Ignorantia*, 1440) and radicalised: in the infinite universe,
all opposites coincide in the absolute.  God (the infinite) contains all
contradictions without contradiction.  The minimum and the maximum are the
same at the limit of the infinite:

  "The universe is a great animal in which contraries are reconciled."
                                      — Bruno, *De la Causa* (1584)

In the Wake, this doctrine is the philosophical engine behind every
portmanteau, every character-merger, every day-and-night oscillation.
SHEM AND SHAUN ARE THE SAME PERSON: the writer and the post, the exile and
the stay-at-home, the soul and the body, the dark and the light — they are
Bruno's opposites that coincide in HCE (the absolute / the father / the
infinite background).

Key coincidentia pairs in the Wake:
  Shem / Shaun        — dark brother / bright brother
  ALP / Issy          — mother / daughter (the same woman at different ages)
  HCE awake / asleep  — the historical and the mythological
  day / night         — the Wake is the night-book of the day-book (Ulysses)
  life / death        — Finnegan dead / Finn waiting to return
  beginning / end     — the last sentence runs into the first word

THE INFINITE UNIVERSE AND PLURALITY OF WORLDS
──────────────────────────────────────────────
Bruno's cosmological revolution (anticipating Copernicus's implications):
the universe is infinite, has no centre, is populated by infinite worlds,
and the earth is not the privileged centre of creation.  Every point in the
infinite is equally the centre, and every point is equally the margin.

In the Wake: every word is simultaneously the centre of the text and a
marginal annotation.  Every reading position is equally valid and equally
partial.  The text has no correct orientation (the Wake has been read
upside-down, diagonally, in mirror-image).  This is Bruno's infinite
universe made typographic.

THE MONAD
──────────
In Bruno's later work (*De Monade*, 1591), the minimum unit of being is the
monad — an indivisible, self-contained unit that reflects the whole of the
universe.  Each monad contains the universe within itself.  Each Wake
portmanteau word is a monad: self-contained, internally consistent, reflecting
the whole system.  To read any single word of the Wake correctly is to read
the whole.

THE MEMORY THEATRE — ARS MEMORATIVA
─────────────────────────────────────
Bruno was the greatest Renaissance practitioner of the classical art of
memory (ars memorativa / ars memoriae), which he described in:
  *Ars Memoriae* (1582)
  *De Umbris Idearum* (Shadows of Ideas, 1582)
  *Cantus Circaeus* (Circe's Song, 1582)
  *De Gl'Heroici Furori* (The Heroic Frenzies, 1585)

The classical memory system (Simonides → Cicero → Quintilian → Bruno):
  1. Construct a vivid mental LOCUS (place): a building, a theatre, a city.
  2. Populate each room/location with a striking IMAGO (image) that encodes
     the thing to be remembered.
  3. To recall, mentally walk through the loci in order.

Bruno radicalised this: his memory theatre was not merely mnemonic but
MAGICAL — the images were living seals that captured the astral forces of
the heavens and could be used to manipulate reality.  The memory palace
becomes a cosmological machine.

THE WAKE AS MEMORY THEATRE:
Joyce constructs a memory theatre on a cosmic scale.  The loci are:
  • The Earwicker pub/home in Chapelizod
  • Phoenix Park (the primal scene / the garden / the park of sin)
  • The Hill of Howth (HCE's body-as-landscape)
  • Dublin's street-grid
Each locus is populated with images (characters, motifs, thunder-words)
that encode entire mythological systems.  To read the Wake is to walk
Bruno's memory theatre.

THE HERMETIC TRADITION
───────────────────────
Bruno worked within the Renaissance hermetic tradition — the corpus of
texts attributed to Hermes Trismegistus (Thoth / Hermes), especially the
*Corpus Hermeticum* (translated by Ficino, 1463).  Hermetic doctrines:
  • As above, so below (the microcosm reflects the macrocosm)
  • The soul's descent through the planetary spheres and its return
  • Magic as the manipulation of natural sympathies
  • The divine spark (scintilla / nitzotz) imprisoned in matter

Bruno synthesised Hermeticism, Neoplatonism, and the new Copernican
cosmology.  The Wake similarly synthesises hermetic and modern frameworks:
the dream-state is a hermetic descent, the dawn-awakening a hermetic return.

BRUNO'S MARTYRDOM — THE NOLAN
───────────────────────────────
Giordano Bruno of Nola (he always called himself "il Nolano" — the Nolan)
was burned alive at the Campo de' Fiori in Rome on 17 February 1600 by
the Roman Inquisition.  He refused to recant.  His crime: denial of the
Trinity, the infinite universe, the plurality of worlds, and (possibly)
reincarnation.

"The Nolan" = Bruno = the martyr-philosopher who refused to choose between
his competing truths.  Joyce identified with Bruno in this: he too refused
to choose between his opposing commitments (Irish / cosmopolitan, Catholic /
atheist, exile / Dubliner).

In the Wake, Bruno/the Nolan is encoded as:
  "Browne and Nolan" (the Dublin booksellers → Bruno of Nola)
  "Browne of Browne and Nolan" vs. "Nolan of Browne and Nolan"
  Nicholas of Cusa (Cusanus) → "Nick the Maggot" (Cusa's Mackey → Nicky)
  The martyred / burned figure (HCE's trial and burning)

THE IDENTITY OF CONTRARIES — SHEM AND SHAUN
─────────────────────────────────────────────
Joyce wrote (in a letter to Harriet Shaw Weaver): "I am using Browne and
Nolan as a symbol of the union of contraries."  The Shem/Shaun opposition
IS Bruno's coincidentia oppositorum in action:

  Shem (the Penman) = dark, feminine, Irish exile, artist, Cain, Glugg
  Shaun (the Post)  = bright, masculine, Irish stay-at-home, deliverer, Abel, Chuff

But: each contains the other.  Shaun becomes Jaun becomes Yawn becomes the
voice of the sleeping HCE (III.3).  Shem writes what Shaun delivers.  They
are the two codices of the same text: the writing and the reading, the
minimum and the maximum, which — in the infinite — coincide.

DIALECTICAL UNITY OF NIGHT AND DAY
────────────────────────────────────
Bruno's universe does not privilege light over darkness.  Night and day are
equal phases of the infinite's self-revelation.  The Wake is the night-book
that requires Ulysses (the day-book) as its counterpart.  Together they form
a coincidentia: the same Dublin, the same June 16, but in night and day.
Bruno's dialectic makes this not a metaphor but a philosophical necessity.

ANALYTICAL PROTOCOL
───────────────────
• Identify coincidentia oppositorum pairs: which apparent opposites are
  shown to be the same at a higher level of abstraction?
• Map the memory-theatre loci: which architectural/geographical locations
  are being used as mnemonic loci, and what images are placed in them?
• Track the Nolan/martyr figure: where does Bruno's martyrdom-pattern
  appear (the accused, the burning, the refusal to recant)?
• Note monad-words: portmanteaux that contain the whole system in miniature.
• Identify the dialectical unity: where does the text demonstrate that
  its apparent oppositions are phases of a single infinite process?
• Mark hermetic correspondences: as above, so below; micro/macrocosm
  parallels; the descent and return of the soul.

SEMANTIC FIELDS OF INTEREST
────────────────────────────
coincidentia_oppositorum · infinite_universe · memory_theatre ·
hermetic_tradition · nolan_martyr · dialectical_unity · monad_identity ·
ars_memorativa
"""


class BrunianLens(Lens):
    """Lens that reads Wake passages through Giordano Bruno's philosophy."""

    def __init__(self, graph_client: Any = None) -> None:
        super().__init__(graph_client)

    def _build_config(self) -> LensConfig:
        return LensConfig(
            name="brunian",
            system_prompt=_SYSTEM_PROMPT,
            foregrounded_fields=[
                "coincidentia_oppositorum",
                "infinite_universe",
                "memory_theatre",
                "hermetic_tradition",
                "nolan_martyr",
                "dialectical_unity",
                "monad_identity",
                "ars_memorativa",
            ],
            graph_traversal={
                # Bruno appears in the graph both as a SourceEntity and via
                # the hermetic tradition nodes.
                "source_entity": "Bruno",
                "node_filter": {
                    "label": "SemanticField",
                    "property": "brunian_framework",
                },
                # Latin (la) for Bruno's treatises, Italian (it) for his
                # vernacular dialogues (*De la Causa* etc.).
                "language_filter": ["la", "it", "en"],
                "include_root_nodes": True,
                "depth": 2,
            },
            attention_priors={
                # Copy-suppression heads: detect the dialectical negation
                # (A suppressed by not-A then reunified).
                "copy_suppression_heads": 0.9,
                # Semantic heads: detect the coincidence of opposing
                # semantic fields at the same token.
                "semantic_heads": 0.85,
                # Structure heads: the concentric-circles structure of
                # Bruno's cosmology and the monad's self-reflection.
                "structure_heads": 0.7,
                # Induction heads: the memory-theatre traversal pattern.
                "induction_heads": 0.65,
            },
            probe_targets=[
                "brunian_dialectic",
                "opposition_detection",
                "memory_loci",
            ],
        )
