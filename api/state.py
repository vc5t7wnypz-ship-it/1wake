"""
Application state manager for the WAKE interpretability API.

:class:`AppState` owns all shared, long-lived resources (model, graph driver,
lenses, tokenizer, etc.) and exposes a single :meth:`initialize` method that
reads environment variables and lazily initialises each component.

Design goals
------------
- **Lazy by default**: components are only initialised when env vars are set.
  This allows the API to start (and serve health/docs endpoints) even when
  Neo4j or the HuggingFace token are not configured.
- **Thread-safe initialisation**: an ``asyncio.Lock`` prevents double-init
  under concurrent startup requests.
- **Graceful degradation**: each subsystem initialises independently;
  failure of one does not block the others.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module loads even without them.
# ---------------------------------------------------------------------------

# These are imported inside initialize() so the module can always be imported
# at test time without torch/neo4j installed.


class AppState:
    """Centralised holder of all WAKE API runtime resources.

    Attributes
    ----------
    model:
        Loaded WAKE model wrapper (``WakeModel``), or None if not yet
        initialised.
    graph:
        Neo4j ``Driver`` instance, or None.
    lenses:
        Dict mapping lens name → :class:`~lenses.base.Lens` instance.
    tokenizer:
        WakeTokenizer, or None.
    runner:
        MultiPassRunner for running contrastive lens passes, or None.
    anamnesis:
        AnamnesisProtocol instance, or None.
    probes:
        Dict mapping layer index → fitted :class:`~interpretability.probes.LanguageFieldProbe`.
    """

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.graph: Optional[Any] = None
        self.lenses: Dict[str, Any] = {}
        self.tokenizer: Optional[Any] = None
        self.runner: Optional[Any] = None
        self.anamnesis: Optional[Any] = None
        self.probes: Dict[int, Any] = {}

        self._initialised: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialise all available components.

        Reads the following environment variables:

        ``NEO4J_URI``
            Bolt URI for the Neo4j instance (default ``bolt://localhost:7687``).
        ``NEO4J_USER``
            Neo4j username (default ``neo4j``).
        ``NEO4J_PASSWORD``
            Neo4j password (default ``wakeanamnesis``).
        ``NEO4J_DATABASE``
            Neo4j database name (default ``neo4j``).
        ``MODEL_NAME``
            HuggingFace model identifier (e.g.
            ``"EleutherAI/gpt-j-6B"``).  If not set, model is not loaded.
        ``HF_TOKEN``
            HuggingFace access token for gated models.  Optional.
        ``PROBES_DIR``
            Directory containing serialised probe pickle files named
            ``layer_{n}.pkl``.  If not set, probes are not loaded.
        """
        async with self._lock:
            if self._initialised:
                return
            logger.info("AppState.initialize() starting …")

            await self._init_graph()
            await self._init_lenses()
            await self._init_model()
            await self._init_tokenizer()
            await self._init_runner()
            await self._init_anamnesis()
            await self._init_probes()

            self._initialised = True
            logger.info(
                "AppState initialised: model=%s, graph=%s, lenses=%s, probes=%s",
                "loaded" if self.model is not None else "not loaded",
                "connected" if self.graph is not None else "not connected",
                list(self.lenses.keys()),
                list(self.probes.keys()),
            )

    # ------------------------------------------------------------------
    # Component initialisers
    # ------------------------------------------------------------------

    async def _init_graph(self) -> None:
        """Connect to Neo4j if the URI env var is set."""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "wakeanamnesis")

        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]

            driver = GraphDatabase.driver(uri, auth=(user, password))
            # Quick connectivity check
            driver.verify_connectivity()
            self.graph = driver
            logger.info("Neo4j driver connected to %s", uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Neo4j not available (%s) — graph features disabled.", exc
            )
            self.graph = None

    async def _init_lenses(self) -> None:
        """Instantiate the standard four lenses (offline mode if no graph)."""
        try:
            from lenses.viconian import ViconianLens  # type: ignore[import-untyped]

            self.lenses["viconian"] = ViconianLens(graph_client=self.graph)
            logger.debug("Viconian lens initialised.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ViconianLens: %s", exc)

        # Psychoanalytic, mythological, and linguistic lenses are loaded
        # dynamically if their modules exist, otherwise skipped gracefully.
        for lens_name in ("psychoanalytic", "mythological", "linguistic"):
            try:
                module_name = f"lenses.{lens_name}"
                import importlib
                mod = importlib.import_module(module_name)
                # Convention: each lenses/<name>.py exports a class named
                # <Capitalized>Lens, e.g. PsychoanalyticLens.
                class_name = lens_name.capitalize() + "Lens"
                lens_cls = getattr(mod, class_name)
                self.lenses[lens_name] = lens_cls(graph_client=self.graph)
                logger.debug("%s lens initialised.", lens_name)
            except (ImportError, AttributeError) as exc:
                logger.debug("Lens '%s' not available: %s", lens_name, exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error loading lens '%s': %s", lens_name, exc)

    async def _init_model(self) -> None:
        """Load the transformer model if MODEL_NAME is set."""
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            logger.info("MODEL_NAME not set — model not loaded.")
            return

        hf_token = os.getenv("HF_TOKEN")

        try:
            # Import the WAKE model wrapper — path may vary as the engine
            # module is developed.  Try the expected locations in order.
            WakeModel: Any = None  # type: ignore[assignment]
            for module_path in (
                "engine.model.wake_model",
                "engine.wake_model",
            ):
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    WakeModel = mod.WakeModel
                    break
                except (ImportError, AttributeError):
                    continue

            if WakeModel is None:
                logger.warning(
                    "WakeModel class not found — falling back to direct "
                    "transformer loading."
                )
                await self._init_model_direct(model_name, hf_token)
                return

            self.model = WakeModel(model_name=model_name, hf_token=hf_token)
            await asyncio.to_thread(self.model.load)
            logger.info("WakeModel loaded: %s", model_name)

        except Exception as exc:  # noqa: BLE001
            logger.error("Model loading failed: %s", exc)
            self.model = None

    async def _init_model_direct(
        self, model_name: str, hf_token: Optional[str]
    ) -> None:
        """Fallback: load a HuggingFace model without the WakeModel wrapper."""
        try:
            import transformer_lens  # type: ignore[import-untyped]

            def _load() -> Any:
                return transformer_lens.HookedTransformer.from_pretrained(
                    model_name,
                    hf_model=None,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False,
                    tokenizer=None,
                    **({} if hf_token is None else {"use_auth_token": hf_token}),
                )

            self.model = await asyncio.to_thread(_load)
            logger.info("HookedTransformer loaded: %s", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Direct model loading failed: %s", exc)
            self.model = None

    async def _init_tokenizer(self) -> None:
        """Load the WakeTokenizer if available."""
        try:
            for module_path in (
                "engine.tokenizer.wake_tokenizer",
                "engine.wake_tokenizer",
            ):
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    cls = mod.WakeTokenizer
                    self.tokenizer = cls()
                    logger.debug("WakeTokenizer initialised from %s.", module_path)
                    return
                except (ImportError, AttributeError):
                    continue
            logger.debug("WakeTokenizer not available — tokenizer not loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tokenizer init failed: %s", exc)

    async def _init_runner(self) -> None:
        """Initialise the MultiPassRunner if the model is available."""
        if self.model is None:
            return

        try:
            for module_path in (
                "engine.multipass.runner",
                "engine.runner",
            ):
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    cls = mod.MultiPassRunner
                    self.runner = cls(
                        model=self.model,
                        lenses=self.lenses,
                        tokenizer=self.tokenizer,
                    )
                    logger.debug("MultiPassRunner initialised.")
                    return
                except (ImportError, AttributeError):
                    continue
            logger.debug("MultiPassRunner not available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Runner init failed: %s", exc)

    async def _init_anamnesis(self) -> None:
        """Initialise the AnamnesisProtocol if both model and graph are available."""
        if self.model is None or self.graph is None:
            return

        try:
            for module_path in (
                "engine.anamnesis.protocol",
                "engine.anamnesis",
            ):
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    cls = mod.AnamnesisProtocol
                    self.anamnesis = cls(model=self.model, graph=self.graph)
                    logger.debug("AnamnesisProtocol initialised.")
                    return
                except (ImportError, AttributeError):
                    continue
            logger.debug("AnamnesisProtocol not available.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anamnesis init failed: %s", exc)

    async def _init_probes(self) -> None:
        """Load serialised probe pickle files from PROBES_DIR if set."""
        from pathlib import Path

        probes_dir_str = os.getenv("PROBES_DIR")
        if not probes_dir_str:
            logger.debug("PROBES_DIR not set — probes not loaded from disk.")
            return

        probes_dir = Path(probes_dir_str)
        if not probes_dir.is_dir():
            logger.warning("PROBES_DIR '%s' does not exist.", probes_dir)
            return

        try:
            from interpretability.probes import LanguageFieldProbe
        except ImportError as exc:
            logger.warning("Could not import LanguageFieldProbe: %s", exc)
            return

        for pkl_path in sorted(probes_dir.glob("layer_*.pkl")):
            # Extract layer index from filename: "layer_12.pkl" → 12
            try:
                stem = pkl_path.stem  # "layer_12"
                layer_idx = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                logger.warning(
                    "Probe file '%s' does not match 'layer_N.pkl' pattern — skipping.",
                    pkl_path.name,
                )
                continue

            try:
                probe = await asyncio.to_thread(LanguageFieldProbe.load, pkl_path)
                self.probes[layer_idx] = probe
                logger.debug("Loaded probe for layer %d from %s.", layer_idx, pkl_path.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load probe from %s: %s", pkl_path, exc)

        logger.info("Loaded probes for layers: %s", sorted(self.probes.keys()))

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Close all open connections gracefully."""
        if self.graph is not None:
            try:
                self.graph.close()
                logger.info("Neo4j driver closed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing Neo4j driver: %s", exc)
            self.graph = None

        # Model teardown (free GPU memory if applicable)
        if self.model is not None:
            try:
                if hasattr(self.model, "unload"):
                    self.model.unload()
                elif hasattr(self.model, "cpu"):
                    # TransformerLens HookedTransformer
                    self.model.cpu()
                import torch  # type: ignore[import-untyped]
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error during model teardown: %s", exc)
            self.model = None

        self._initialised = False
        logger.info("AppState shutdown complete.")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if at least the lenses are available (minimal working state)."""
        return bool(self.lenses)

    @property
    def available_lenses(self) -> list[str]:
        """List of successfully initialised lens names."""
        return list(self.lenses.keys())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: The single shared AppState instance used by all API routes.
app_state: AppState = AppState()
