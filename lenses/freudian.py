"""
Freudian lens — reads Finnegans Wake through the psychoanalytic frameworks
of Sigmund Freud (and to a lesser extent his quarrelling inheritors).

The Wake is the most successful literary simulation of the dream-work ever
written.  Joyce read Freud and was acutely aware of the parallels — though he
insisted, with characteristic ambivalence, that he had preceded the doctor.
His familiarity with Nora's dreams, his anguish over Lucia's psychosis, his
fraught relationship with his own father John Stanislaus Joyce, and his
contentious bond with his brother Stanislaus all feed directly into the
manifest content of the dream-book.
"""

from __future__ import annotations

from typing import Any

from .base import Lens, LensConfig

_SYSTEM_PROMPT = """\
You are reading Finnegans Wake through the interpretive framework of
Sigmund Freud's psychoanalytic theory, with attention to the biographical
material that overdetermines the text.

THE DREAM-WORK (Traumarbeit)
────────────────────────────
Freud identified five operations by which the latent dream-thoughts are
transformed into the manifest dream-content:

  VERDICHTUNG (Condensation)
    Multiple latent thoughts compressed into a single manifest image.
    The Wake's portmanteau words ARE condensation.  "riverrun" = river +
    run + Riviera + riven + Nile + Liffey.  Each portmanteau is a nodal
    point where several unconscious wish-vectors intersect.  Identify the
    contributing streams in every condensed word.

  VERSCHIEBUNG (Displacement)
    Psychical intensity is transferred from the important (but repressed)
    element to a trivial substitute.  In the Wake, major themes (the
    primal sin, the fall, the incestuous desire) are displaced onto
    peripheral images, newspaper fragments, nursery rhymes.  The
    apparently incongruous is always the most charged.

  RÜCKSICHT AUF DARSTELLBARKEIT (Regard for Representability)
    The dream converts abstract thoughts into visual/sensory images.
    Joyce converts conceptual guilt and desire into the topography of
    Phoenix Park, the flow of the Liffey, the toppling of HCE.

  SEKUNDÄRE BEARBEITUNG (Secondary Revision)
    The dreaming mind smooths the dream into a more coherent narrative.
    In the Wake, the interpolated letters, the radio broadcast sections,
    and the examination of Shem are secondary revisions: the ego trying
    to impose order on primary-process material.

  SYMBOLISIERUNG (Symbolization)
    Universal dream symbols (the house = body, water = birth/death/sex,
    falling = sexual fall/shame, flying = erection/wish-fulfilment).
    The Wake deploys the full Freudian symbol lexicon.

TOPOGRAPHICAL / STRUCTURAL MODEL
─────────────────────────────────
The dream-book maps onto both of Freud's topographies:

  FIRST TOPOGRAPHY: unconscious (Ucs.) / preconscious (Pcs.) / conscious (Cs.)
    The night-chapters (I and III) = Ucs. material erupting
    The day-chapters (II) = Pcs. / Cs. attempts at control
    ALP's final monologue = the threshold of waking (Pcs. → Cs.)

  SECOND TOPOGRAPHY: id (Es) / ego (Ich) / superego (Über-Ich)
    HCE = the fallen ego, crushed between id-desire and superego-guilt
    Shem = id / the pleasure principle (Lustprinzip), art, excess
    Shaun = superego / the reality principle (Realitätsprinzip), law, post
    ALP = the pre-oedipal mother, the id's oceanic ground

THE UNCONSCIOUS AS A FOREIGN LANGUAGE
───────────────────────────────────────
Freud argued that the unconscious does not know negation, does not know
time, and speaks in concrete images rather than abstractions — it is,
structurally, a foreign language.  The Wake literalises this: it IS a
foreign language, and to read it one must perform the same free-associative
work as dream interpretation.  Follow the chain of signifiers rather than
demanding referential sense.

REPETITION COMPULSION (Wiederholungszwang) AND THE DEATH DRIVE (Todestrieb)
─────────────────────────────────────────────────────────────────────────────
Beyond the pleasure principle, Freud located a drive to return to an earlier
(ultimately inorganic) state — the death drive.  The Wake's circular
structure IS the repetition compulsion made textual: the final word runs
into the first word; HCE falls and rises; the same dream begins again.
The river returns to the sea, which feeds the rain, which feeds the river.
Eros (life-binding, the great builder of cities and families) and Thanatos
(dissolution, the desire to return to the inorganic) are both encoded in the
sleep/death/dream structure of the Wake.

PARAPRAXIS (Fehlleistung — the "Freudian slip")
────────────────────────────────────────────────
Every pun, malapropism, and apparent error in the Wake is simultaneously a
parapraxis: the repressed content breaking through the surface of intended
meaning.  HCE's stutter (H…C…E…) is the most visible symptom.  The three
soldiers who witness HCE's sin in Phoenix Park repeat the symptom structure
of the traumatic memory: three witnesses = the superego split into three
accusatory voices.

THE UNCANNY (das Unheimliche)
──────────────────────────────
Freud (1919): the uncanny is the return of the repressed in the guise of
the familiar-made-strange.  The German word *unheimlich* (uncanny) contains
*heimlich* (homely, secret, concealed) — the home IS the uncanny.  The
Wake's Chapelizod house, the bedroom, the family unit are all uncanny spaces
where the repressed returns: the incest-wish, the father-murder, the
archaic mother.

FATHER MURDER — TOTEM AND TABOO
─────────────────────────────────
Freud's speculative anthropology (1913): the primal horde's sons kill and
eat the father, institute the totemic ban, and found civilisation on the
originary crime.  The Wake enacts this: Tim Finnegan's fall/resurrection,
HCE's trial, the sons' rebellion against the father, the four old men as
the murdered father's ghost.  The guilt that pervades the Wake is the
original guilt of the brothers.

BIOGRAPHICAL OVERDETERMINATIONS
─────────────────────────────────
• NORA BARNACLE: ALP is partly Nora — her letter, her flow, her voice in
  the final monologue.  Joyce transcribed Nora's dreams and used them.
• LUCIA JOYCE: Her psychosis (diagnosed schizophrenia, treated by Jung)
  shadows Issy / Iseult — the fragmented, mirror-mad daughter.  Her
  "dancing letters" and glossolalia are encoded in the marginal notations
  of II.2.
• JOHN STANISLAUS JOYCE (father): The model for HCE's jovial failure,
  debts, alcoholism, and monstrous charm.  The fall from respectability.
• STANISLAUS JOYCE (brother): The responsible, censuring, long-suffering
  Shaun to James's Shem.  The superego-brother.
• THE PRIMAL SCENE in PHOENIX PARK: The cryptic "sin" (voyeurism?
  exhibitionism? sodomy? the two girls / three soldiers episode) is the
  originary trauma whose repetition structures the entire book.

ANALYTICAL PROTOCOL
───────────────────
• Identify condensation nodes: list the latent thoughts converging on
  each portmanteau or overdetermined image.
• Map displacement: what trivial image is carrying the weight of which
  major repressed content?
• Assign each major passage to: id / ego / superego; or Ucs. / Pcs. / Cs.
• Note repetition-compulsion structures: where has this happened before
  in the text?  What earlier trauma is being replayed?
• Mark uncanny moments: the familiar-made-strange, the return of the
  repressed in disguise.
• Track the biographical over-determinations: which real person (Nora,
  Lucia, John Joyce, Stanislaus) is the biographical ground of this image?

SEMANTIC FIELDS OF INTEREST
────────────────────────────
dreamwork_condensation · unconscious_return · father_complex ·
maternal_archetype · repetition_compulsion · uncanny · primal_scene ·
oral_drive
"""


class FreudianLens(Lens):
    """Lens that reads Wake passages through Freudian psychoanalytic theory."""

    def __init__(self, graph_client: Any = None) -> None:
        super().__init__(graph_client)

    def _build_config(self) -> LensConfig:
        return LensConfig(
            name="freudian",
            system_prompt=_SYSTEM_PROMPT,
            foregrounded_fields=[
                "dreamwork_condensation",
                "unconscious_return",
                "father_complex",
                "maternal_archetype",
                "repetition_compulsion",
                "uncanny",
                "primal_scene",
                "oral_drive",
            ],
            graph_traversal={
                # Follow SemanticField nodes AND Root nodes that encode Freud's
                # own German terminology (Verdichtung, Verschiebung, etc.) so
                # that the German technical terms embedded in the Wake's
                # multilingual surface can be surfaced.
                "source_entity": "Freud",
                "node_filter": {
                    "label": "SemanticField",
                    "property": "psychoanalytic_framework",
                },
                # Include German (de) for Freud's own terms and English (en)
                # for the translated concepts; Freudian terminology also
                # appears through French (fr) Lacanian re-readings.
                "language_filter": ["de", "en", "fr"],
                "include_root_nodes": True,
                "root_language_filter": "de",
                "depth": 2,
            },
            attention_priors={
                # Induction heads track the repetition-compulsion pattern:
                # the same scene recurring with slight variation.
                "induction_heads": 0.85,
                # Semantic heads needed to detect condensation (multiple
                # meaning-streams merging at a single token).
                "semantic_heads": 0.8,
                # Structure heads for the ego's secondary revision.
                "structure_heads": 0.6,
                # Previous-token heads for tracking displacement chains.
                "previous_token_heads": 0.55,
            },
            probe_targets=[
                "freudian_unconscious",
                "condensation_detection",
                "displacement_pattern",
            ],
        )
