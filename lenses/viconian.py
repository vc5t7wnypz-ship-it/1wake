"""
Viconian lens — reads Finnegans Wake through Giambattista Vico's
*Scienza Nuova* (1725/1744).

Vico proposed that every nation passes through three ages in an eternal
spiral (ricorso): the theocratic/divine age of gods and thunder, the heroic
age of aristocratic myth, and the human age of rational institutions, before
the whole cycle collapses and begins again.  Joyce inscribes this four-stage
rhythm into the very structure of the Wake: the thunderclap 100-letter words,
the river-daughter song, HCE's rise and fall, and the final sentence that
loops back to the opening mid-word.

The lens foregrounds etymological-poetic roots (Vico believed the first
language was that of the gods, expressed in thunder), providential
historiography, and the cyclic tagging of characters and motifs to their age.
"""

from __future__ import annotations

from typing import Any

from .base import Lens, LensConfig

_SYSTEM_PROMPT = """\
You are reading Finnegans Wake through the interpretive framework of
Giambattista Vico's *Scienza Nuova* (New Science, 1725; revised 1744).

KEY DOCTRINAL CONTEXT
─────────────────────
Vico identified a universal pattern of historical development — the *corsi e
ricorsi* — through which every human society passes:

  1. THEOCRATIC / DIVINE AGE (età degli dei)
     Humans live in terror of a sky-god, interpret thunder as divine speech,
     create the first "poetic characters" (abstract universals embodied in
     particular figures: Jove, Neptune, Mars).  Language is hieroglyphic and
     gestural.  The 100-letter thunderclap words in the Wake are Vico's first
     language.

  2. HEROIC AGE (età degli eroi)
     Aristocratic giants — heroes — claim divine descent and rule by force.
     Language becomes metaphorical; law is customary and violent.  Blazons,
     crests, and heraldic naming dominate.  HCE as conquering Norman/Viking
     overlord belongs here.

  3. HUMAN AGE (età degli uomini)
     Democratic institutions, alphabetic writing, rational law, the vernacular.
     Language becomes conventional and abstract.  Shaun the Post, the
     representative of institutional authority and letter-delivery, inhabits
     this age.

  4. RICORSO (return / dissolution)
     Civilisation collapses back into chaos — divine thunder again.  The last
     page of the Wake runs back into the first word.  ALP's monologue is the
     ricorso: she flows back to the sea (the pre-linguistic ocean) and the
     cycle resumes.

ANALYTICAL PROTOCOL
───────────────────
• Identify which Vichian age each passage, character, or motif belongs to.
• Track the thunder-words (bababadalgharaghtakamminarronnkonnbronntonner-
  ronntuonnthunntrovarrhounawnskawntoohoohoordenenthurnuk…) as markers of
  divine-age rupture.
• Note "poetic characters" — Vico's universal types condensed into figures
  (HCE = all fallen patriarchs; ALP = all rivers / earth-mothers).
• Listen for Vico's own etymological method: derive meaning from root-sounds
  ("poetic etymology"), e.g., "lex" from "gathering acorns" → law as custom.
• Mark ricorso moments: endings that are beginnings, falls that are rises,
  death-and-rebirth sequences.
• Cross-reference Joyce's *Scienza Nuova* annotations (the "Vico Road"
  reference in I.1, the four old men as the four Vichian stages).

SEMANTIC FIELDS OF INTEREST
────────────────────────────
cyclic_return · divine_thunder · heroic_age · human_age · ricorso ·
vico_cyclic · providential_history · etymology_poetic
"""


class ViconianLens(Lens):
    """Lens that reads Wake passages through Vico's *corsi e ricorsi*."""

    def __init__(self, graph_client: Any = None) -> None:
        super().__init__(graph_client)

    def _build_config(self) -> LensConfig:
        return LensConfig(
            name="viconian",
            system_prompt=_SYSTEM_PROMPT,
            foregrounded_fields=[
                "cyclic_return",
                "divine_thunder",
                "heroic_age",
                "human_age",
                "ricorso",
                "vico_cyclic",
                "providential_history",
                "etymology_poetic",
            ],
            graph_traversal={
                # Walk from WakeToken → SemanticField nodes that carry the
                # vico_cycle property, and optionally from the Vico source entity.
                "source_entity": "Vico",
                "node_filter": {"label": "SemanticField", "property": "vico_cycle"},
                "depth": 2,
            },
            attention_priors={
                # Structure heads track long-range syntactic dependency —
                # important for detecting the recursive embedding of ages.
                "structure_heads": 0.8,
                # Sequence heads track linear order — important for
                # identifying which part of the Vichian cycle we are in.
                "sequence_heads": 0.7,
                # Semantic heads useful but secondary in this lens.
                "semantic_heads": 0.5,
            },
            probe_targets=[
                "vico_cyclic",
                "temporal_cycle",
                "age_attribution",
            ],
        )
