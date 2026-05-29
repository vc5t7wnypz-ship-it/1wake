"""
WAKE Interpretability API — main FastAPI application.

Exposes endpoints for:
  - Health check
  - Lens catalogue
  - Single-passage analysis
  - Batch analysis
  - Superposition map
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.state import AppState, app_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start-up / shutdown logic attached to the FastAPI app lifespan."""
    logger.info("WAKE API starting up …")
    try:
        await app_state.initialize()
    except Exception as exc:  # noqa: BLE001
        logger.warning("AppState.initialize() raised %s — continuing in degraded mode.", exc)
    yield
    logger.info("WAKE API shutting down …")
    await app_state.shutdown()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WAKE Interpretability API",
    description=(
        "Mechanistic interpretability of Finnegans Wake: "
        "superposed meaning in LLM residual streams."
    ),
    version="0.1.0",
    docs_url="/api/wake/docs",
    openapi_url="/api/wake/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get(
    "/api/wake/health",
    summary="Health check",
    tags=["system"],
)
async def health() -> Dict[str, Any]:
    """Return the current health status of the API and its subsystems.

    Returns
    -------
    dict
        - ``status``: ``"ok"`` when all subsystems are available;
          ``"degraded"`` when some are unavailable.
        - ``model``: ``"loaded"`` | ``"not_loaded"``
        - ``graph``: ``"connected"`` | ``"not_connected"``
        - ``lenses``: list of available lens names
        - ``probes``: list of available probe layer indices
    """
    return {
        "status": "ok" if app_state.is_ready else "degraded",
        "model": "loaded" if app_state.model is not None else "not_loaded",
        "graph": "connected" if app_state.graph is not None else "not_connected",
        "lenses": app_state.available_lenses,
        "probes": sorted(app_state.probes.keys()),
    }


# ---------------------------------------------------------------------------
# Lens catalogue endpoint
# ---------------------------------------------------------------------------


@app.get(
    "/api/wake/lenses",
    summary="List available lenses",
    tags=["lenses"],
)
async def list_lenses() -> List[Dict[str, Any]]:
    """Return metadata for all registered WAKE lenses.

    Each entry includes the lens name, a short description derived from
    its system prompt, the foregrounded semantic fields, and the probe
    targets.

    Returns
    -------
    list[dict]
        One dict per registered lens.
    """
    try:
        from lenses.registry import ALL_LENSES

        results: List[Dict[str, Any]] = []
        for name, cls in sorted(ALL_LENSES.items()):
            try:
                instance = cls()
                config = instance.config
                # Extract first line of system prompt as description.
                first_line = config.system_prompt.strip().splitlines()[0] if config.system_prompt else ""
                results.append({
                    "name": config.name,
                    "description": first_line,
                    "foregrounded_fields": config.foregrounded_fields,
                    "probe_targets": config.probe_targets,
                    "attention_priors": config.attention_priors,
                })
            except Exception as exc:  # noqa: BLE001
                results.append({"name": name, "error": str(exc)})
        return results
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Lens registry not available.",
        )


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/api/wake/analyse",
    summary="Analyse a Wake passage",
    tags=["analysis"],
)
async def analyse_passage(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Wake passage through the multi-pass runner under all lenses.

    Request body
    ------------
    ``passage`` (str, required):
        The Wake text to analyse.
    ``page`` (int, default 3):
        Source page number.
    ``line`` (int, default 1):
        Source line number.
    ``lenses`` (list[str], optional):
        Names of lenses to use.  Defaults to all registered lenses.
    ``layers_to_capture`` (list[int], optional):
        Model layers from which to capture residual-stream activations.

    Returns
    -------
    dict
        Serialised :class:`~engine.multipass.runner.ContrastiveResult`.
    """
    if app_state.runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.  Set MODEL_NAME env var to enable analysis.",
        )

    passage: str = body.get("passage", "")
    if not passage:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'passage' is required.",
        )

    page: int = int(body.get("page", 3))
    line: int = int(body.get("line", 1))
    layers_to_capture: Optional[List[int]] = body.get("layers_to_capture")

    # Tokenise passage
    wake_tokens: List[Any] = []
    if app_state.tokenizer is not None:
        try:
            wake_tokens = app_state.tokenizer.tokenize(passage, page=page, line=line)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tokenizer failed on passage: %s", exc)

    # Run analysis
    try:
        import asyncio

        result = await asyncio.to_thread(
            app_state.runner.run,
            passage=passage,
            wake_tokens=wake_tokens,
            page=page,
            line=line,
            layers_to_capture=layers_to_capture,
        )
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis error: {exc}",
        )

    # Serialise (arrays → lists already handled in ContrastiveResult)
    return {
        "passage": result.passage,
        "page": result.page,
        "line": result.line,
        "is_superposed": result.is_superposed,
        "dominant_field": result.dominant_field,
        "lens_agreement": result.lens_agreement,
        "entropy_profile": result.entropy_profile,
        "superposition_analysis": result.superposition_analysis,
        "n_passes": len(result.pass_results),
    }


# ---------------------------------------------------------------------------
# Superposition map endpoint
# ---------------------------------------------------------------------------


@app.get(
    "/api/wake/superposition-map",
    summary="Superposition map for a page range",
    tags=["analysis"],
)
async def superposition_map(
    page_from: int = 3,
    page_to: int = 5,
) -> Dict[str, Any]:
    """Return a superposition map for pages *page_from* through *page_to*.

    This is a placeholder endpoint; in production it streams pre-computed
    results from the corpus database.

    Returns
    -------
    dict with ``page_from``, ``page_to``, and ``tokens`` (empty list for now).
    """
    return {
        "page_from": page_from,
        "page_to": page_to,
        "tokens": [],
        "note": "Pre-computed superposition map endpoint.  Run E1/E2 experiments to populate.",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
