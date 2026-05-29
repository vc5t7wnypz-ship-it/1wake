"""
WAKE Interpretability API — main FastAPI application.

Exposes endpoints for:
  - Health check
  - Lens catalogue
  - Single-passage analysis (contrastive multi-lens)
  - Batch analysis
  - Superposition map
  - Graph queries (node, path, neighbourhood, semantic fields)
  - Lens composition
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import analysis, graph
from .state import AppState, app_state

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
        logger.info(
            "Startup complete.  Lenses available: %s",
            app_state.available_lenses,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("AppState.initialize() raised %s — continuing in degraded mode.", exc)
    yield
    logger.info("WAKE API shutting down …")
    await app_state.shutdown()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WAKE Interpretability API",
    description=(
        "Mechanistic interpretability of Finnegans Wake: "
        "superposed meaning in LLM residual streams.\n\n"
        "Run contrastive multi-lens analysis on Wake passages, inspect "
        "residual-stream geometry, trace attention-head functions, and "
        "query the Wake knowledge graph."
    ),
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React / Next.js dev server
        "http://localhost:8080",   # alternative frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log each request with its method, path, client IP, and elapsed time."""
    start = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    logger.info("→ %s %s  client=%s", request.method, request.url.path, client_host)

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.info(
            "← %s %s  status=%d  elapsed=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(
            "✗ %s %s  error=%s  elapsed=%.3fs",
            request.method,
            request.url.path,
            exc,
            elapsed,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
        )


# ---------------------------------------------------------------------------
# Routers from the routes sub-package
# ---------------------------------------------------------------------------

app.include_router(analysis.router)
app.include_router(graph.router)

# ---------------------------------------------------------------------------
# Supplementary endpoints not covered by the sub-routers
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """Minimal root endpoint — points users to /docs."""
    return {
        "name": "WAKE Interpretability API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/wake/health",
    }


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
