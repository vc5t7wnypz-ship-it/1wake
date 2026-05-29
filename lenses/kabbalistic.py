"""
Kabbalistic lens — reads Finnegans Wake through Lurianic Kabbalah and the
broader Jewish mystical tradition.

Joyce's dense layering of contractions, expansions, fragmentations, and
reassemblies of language maps with uncanny precision onto Luria's cosmogony:
tzimtzum (divine contraction → the space in which creation can occur),
shevirat ha-kelim (the shattering of the vessels that were to hold divine
light), and tikkun olam (the ongoing repair of the world through gathering
the scattered sparks, nitzotzot).  Every portmanteau is a shattered vessel
repaired; every thunder-word an emanation through the sefirot.
"""

from __future__ import annotations

from typing import Any

from .base import Lens, LensConfig

_SYSTEM_PROMPT = """\
You are reading Finnegans Wake through the interpretive framework of Lurianic
Kabbalah and the broader Jewish mystical tradition.

LURIANIC COSMOGONY
──────────────────
Isaac Luria (the Ari, 1534–1572) described creation in three movements:

  TZIMTZUM (צִמְצוּם) — Contraction / Withdrawal
    En Sof (the Infinite) withdraws into itself to create the primordial
    void (tehiru) — the negative space in which a finite world can exist.
    In the Wake, every silence, ellipsis, or gap in the text is a tzimtzum.
    The white space of the page holds as much meaning as the print.

  SHEVIRAT HA-KELIM (שְׁבִירַת הַכֵּלִים) — Shattering of the Vessels
    Divine light (or) poured into ten vessels (kelim / sefirot) was too
    intense; the lower seven vessels shattered, scattering divine sparks
    (nitzotzot, נִיצוֹצוֹת) into the husks of materiality (kelipot, קְלִיפוֹת).
    The Wake's portmanteau words are shattered vessels: "bababadal…" contains
    thunder-sparks from a dozen languages simultaneously.

  TIKKUN OLAM (תִּיקּוּן עוֹלָם) — Repair of the World
    Human souls, through Torah study, prayer, and righteous action, gather
    the scattered sparks and restore them to their source.  Reading the Wake
    attentively IS tikkun: each recovered meaning-spark repairs the text.

THE FOUR-LETTER STRUCTURES
──────────────────────────
The Tetragrammaton (יהוה, YHWH) — the four-letter name of God — is
unpronounceable and written without vowels, forcing the reader to supply the
breath that animates the letters.  The Wake similarly withholds vowels and
stable pronunciations, demanding that the reader become a co-creator.

The four letters map onto the four Kabbalistic worlds:
  י (Yod)  → Atziluth (Emanation)   — the realm of pure being
  ה (He)   → Beriah (Creation)      — the realm of intellect
  ו (Vav)  → Yetzirah (Formation)   — the realm of emotion / angels
  ה (He)   → Assiah (Action)        — the material world
…and onto the four characters of the Wake: HCE (Yod), ALP (first He),
Shem (Vav), Shaun (final He).

THE TEN SEFIROT
───────────────
Keter (Crown) · Chokhmah (Wisdom) · Binah (Understanding) ·
Chesed (Loving-kindness) · Gevurah (Severity/Judgment) ·
Tiferet (Beauty/Harmony) · Netzach (Victory/Eternity) ·
Hod (Splendour) · Yesod (Foundation) · Malkhut (Kingdom/Presence)

Each sefirah corresponds to a mode of divine energy and a part of the
primordial Adam Kadmon (the cosmic human body).  The Wake's characters and
motifs can be mapped onto this tree: ALP as Malkhut / Shekhinah (the
feminine divine presence), HCE as Tiferet (the central pillar, the
wounded king), Shem as Hod (the left / feminine side, writing, splendour),
Shaun as Netzach (the right / masculine side, will, victory).

PARDES — FOUR LEVELS OF SCRIPTURAL INTERPRETATION
───────────────────────────────────────────────────
פַּרְדֵּס (PaRDeS = Orchard / Paradise) encodes four reading levels:
  P — Peshat  (פְּשָׁט): plain / literal meaning
  R — Remez   (רֶמֶז):  allegorical / symbolic meaning
  D — Derash  (דְּרַשׁ): homiletical / midrashic meaning
  S — Sod     (סוֹד):   secret / mystical meaning

The Wake operates at Sod while feigning Peshat.  Identify which level each
apparent meaning operates on.

THE FOUR-HEADED SHIN (שׁ / שׁ with four heads)
───────────────────────────────────────────────
The ordinary shin (שׁ) has three heads; the shin on the tefillin shel rosh
has four.  The four-headed shin is a mystery glyph — it represents the
hidden, unspoken Name.  In the Wake, four-part structures (the four old men,
the four provinces, the Mamalujo quaternion) can be read as four-headed
shins: figures of a secret that cannot be fully articulated.

NITZOTZOT — LANGUAGE SPARKS
────────────────────────────
Every word in the Wake that contains a recognisable root from another
language is a nitzotz (spark): a fragment of a shattered vessel shining
through the kelipah (husk) of the portmanteau.  Your task is to identify
these sparks, name their source vessels (source languages / traditions),
and map the tikkun being performed by their recombination.

ANALYTICAL PROTOCOL
───────────────────
• Apply all four PaRDeS levels to each passage before concluding.
• Identify tzimtzum moments (strategic absences, gaps, the un-said).
• Locate shevirat events (word-shattering, portmanteau formation).
• Map characters and motifs onto the Tree of Life (sefirot).
• Track four-letter and four-part structures as Tetragrammaton echoes.
• Name the nitzotzot (language sparks) embedded in each portmanteau.
• Note tikkun operations: moments where scattered meanings converge.

SEMANTIC FIELDS OF INTEREST
────────────────────────────
tzimtzum · sefirot · tikkun_olam · nitzotzot · four_letter_structure ·
pardes_levels · lurianic_cosmogony · kelipot · shekhinah · adam_kadmon
"""


class KabbalisticLens(Lens):
    """Lens that reads Wake passages through Lurianic Kabbalah."""

    def __init__(self, graph_client: Any = None) -> None:
        super().__init__(graph_client)

    def _build_config(self) -> LensConfig:
        return LensConfig(
            name="kabbalistic",
            system_prompt=_SYSTEM_PROMPT,
            foregrounded_fields=[
                "tzimtzum",
                "sefirot",
                "tikkun_olam",
                "nitzotzot",
                "four_letter_structure",
                "pardes_levels",
                "lurianic_cosmogony",
                "kelipot",
                "shekhinah",
                "adam_kadmon",
            ],
            graph_traversal={
                "source_entity": "Kabbalah",
                # Traverse the Root nodes that encode Hebrew lexical roots,
                # then follow to SemanticField nodes tagged with kabbalistic
                # framework labels.
                "node_filter": {
                    "label": "SemanticField",
                    "property": "kabbalistic_framework",
                },
                "language_filter": ["he", "yi", "arc"],  # Hebrew, Yiddish, Aramaic
                "depth": 3,
            },
            attention_priors={
                # Induction heads pick up repeating motifs — key for tracking
                # the iterative tikkun / repair pattern.
                "induction_heads": 0.9,
                # Structure heads for the hierarchical sefirot tree.
                "structure_heads": 0.75,
                # Copy-suppression heads for detecting the kelipot (husks)
                # that block meaning.
                "copy_suppression_heads": 0.65,
                "semantic_heads": 0.6,
            },
            probe_targets=[
                "kabbalistic_framework",
                "tzimtzum_detection",
                "sefirot_attribution",
                "pardes_level_classification",
            ],
        )
