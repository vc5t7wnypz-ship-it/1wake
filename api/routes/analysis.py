"""
Analysis routes for the WAKE interpretability API.

Endpoints
---------
GET  /api/wake/health          — system health check
GET  /api/wake/lenses          — list available lenses
POST /api/wake/analyse         — run contrastive multi-lens analysis
POST /api/wake/lens/compose    — compose multiple lenses into a blend
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..schemas.requests import LensComposeRequest, PassageRequest
from ..schemas.responses import (
    AnamnesisResponse,
    ContrastiveResponse,
    HealthResponse,
    TokenPoint,
)
from ..state import AppState, app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wake", tags=["analysis"])


# ---------------------------------------------------------------------------
# Dependency: provide the module-level AppState
# ---------------------------------------------------------------------------

def get_state() -> AppState:
    """FastAPI dependency that returns the shared AppState singleton."""
    return app_state


# ---------------------------------------------------------------------------
# Request-logging middleware helper
# (Registered on the router via an API route interceptor.)
# ---------------------------------------------------------------------------

async def _log_request(request: Request) -> None:
    """Log incoming request method, path, and timing."""
    start = time.perf_counter()
    logger.info(
        "→ %s %s  client=%s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )
    # We can't easily log timing here without a middleware; timing is logged
    # at the handler level.
    return None


# ---------------------------------------------------------------------------
# GET /api/wake/health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the operational status of all WAKE subsystems: model, graph, "
        "and available lenses."
    ),
)
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Return current health status of the API."""
    model_loaded = state.model is not None
    graph_connected = state.graph is not None
    lenses_available = state.available_lenses

    overall_status: str
    if model_loaded and graph_connected:
        overall_status = "ok"
    elif lenses_available:
        overall_status = "degraded"
    else:
        overall_status = "error"

    return HealthResponse(
        status=overall_status,
        model_loaded=model_loaded,
        graph_connected=graph_connected,
        lenses_available=lenses_available,
    )


# ---------------------------------------------------------------------------
# GET /api/wake/lenses
# ---------------------------------------------------------------------------

@router.get(
    "/lenses",
    summary="List available lenses",
    description="Returns a list of lens names that are currently initialised and ready.",
)
async def list_lenses(
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """List all available lens names."""
    lenses_info: List[Dict[str, Any]] = []

    for name, lens in state.lenses.items():
        config = getattr(lens, "config", None)
        info: Dict[str, Any] = {"name": name}
        if config is not None:
            info["foregrounded_fields"] = getattr(
                config, "foregrounded_fields", []
            )
            info["probe_targets"] = getattr(config, "probe_targets", [])
        lenses_info.append(info)

    return {
        "lenses": lenses_info,
        "count": len(lenses_info),
    }


# ---------------------------------------------------------------------------
# POST /api/wake/analyse
# ---------------------------------------------------------------------------

@router.post(
    "/analyse",
    response_model=ContrastiveResponse,
    summary="Run contrastive multi-lens analysis",
    description=(
        "Runs the passage through each requested lens, captures residual-stream "
        "activations, runs linear probes, builds attention-divergence maps, and "
        "optionally runs the anamnesis retrieval protocol."
    ),
    status_code=status.HTTP_200_OK,
)
async def analyse(
    request_body: PassageRequest,
    request: Request,
    state: AppState = Depends(get_state),
) -> ContrastiveResponse:
    """Contrastive multi-lens analysis endpoint."""
    start_time = time.perf_counter()
    logger.info(
        "→ POST /analyse  page=%d line=%d lenses=%s",
        request_body.page,
        request_body.line,
        request_body.lenses,
    )

    # ---- Validate requested lenses are available -----------------------
    for lens_name in request_body.lenses:
        if lens_name not in state.lenses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Lens '{lens_name}' is not available.  "
                    f"Available lenses: {state.available_lenses}"
                ),
            )

    # ---- Check model availability -------------------------------------
    if state.model is None and state.runner is None:
        # Run in "graph-only" mode: no activations, probes return zeros.
        logger.warning(
            "Model not loaded — returning stub analysis (graph-only mode)."
        )
        return _stub_response(request_body)

    # ---- Run multi-pass analysis via the runner -----------------------
    try:
        if state.runner is not None:
            result = await _run_with_runner(request_body, state)
        else:
            result = await _run_direct(request_body, state)

        elapsed = time.perf_counter() - start_time
        logger.info("← POST /analyse  elapsed=%.3fs", elapsed)
        return result

    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {exc}",
        ) from exc


async def _run_with_runner(
    req: PassageRequest,
    state: AppState,
) -> ContrastiveResponse:
    """Delegate to the MultiPassRunner when available."""
    import asyncio

    runner = state.runner
    lenses = [state.lenses[name] for name in req.lenses]

    # MultiPassRunner.run_contrastive may be sync or async; handle both.
    run_fn = getattr(runner, "run_contrastive", None)
    if run_fn is None:
        raise AttributeError("MultiPassRunner has no 'run_contrastive' method.")

    if asyncio.iscoroutinefunction(run_fn):
        raw_result = await run_fn(
            passage=req.passage,
            page=req.page,
            line=req.line,
            lenses=lenses,
            layers=req.layers_to_capture,
        )
    else:
        raw_result = await asyncio.to_thread(
            run_fn,
            passage=req.passage,
            page=req.page,
            line=req.line,
            lenses=lenses,
            layers=req.layers_to_capture,
        )

    # Convert raw runner result → ContrastiveResponse
    return _convert_runner_result(raw_result, req, state)


async def _run_direct(
    req: PassageRequest,
    state: AppState,
) -> ContrastiveResponse:
    """Run analysis directly when no MultiPassRunner is available."""
    import asyncio
    import numpy as np

    model = state.model
    passage = req.passage

    # Tokenise using the model's tokenizer directly
    def _tokenize() -> Any:
        if hasattr(model, "to_tokens"):
            # TransformerLens interface
            return model.to_tokens(passage, prepend_bos=True)
        elif hasattr(model, "tokenizer"):
            return model.tokenizer(passage, return_tensors="pt")["input_ids"]
        raise RuntimeError("Cannot tokenise: model has no to_tokens or tokenizer.")

    import torch
    tokens = await asyncio.to_thread(_tokenize)

    # Get token strings
    def _to_str_tokens() -> List[str]:
        if hasattr(model, "to_str_tokens"):
            return model.to_str_tokens(tokens[0])
        return [str(i) for i in range(tokens.shape[1])]

    str_tokens: List[str] = await asyncio.to_thread(_to_str_tokens)
    seq_len = len(str_tokens)

    # For each lens, run forward pass and collect probe results
    tokens_by_lens: Dict[str, List[TokenPoint]] = {}
    lens_residuals: Dict[str, Any] = {}

    for lens_name in req.lenses:
        lens = state.lenses[lens_name]
        context = lens.get_context_for_passage(passage, [])

        def _forward(ln: str = lens_name) -> Any:
            with torch.no_grad():
                if hasattr(model, "run_with_cache"):
                    logits, cache = model.run_with_cache(tokens)
                elif hasattr(model, "forward_with_cache"):
                    logits, cache = model.forward_with_cache(tokens)
                else:
                    logits = model(tokens)
                    cache = {}
            return logits, cache

        logits, cache = await asyncio.to_thread(_forward)

        # Extract residuals at the final available layer
        layer_keys = [k for k in cache if "hook_resid_post" in k]
        token_points: List[TokenPoint] = []

        if layer_keys and state.probes:
            # Use the deepest available layer's residuals
            last_key = sorted(layer_keys)[-1]
            layer_idx = int(last_key.split(".")[1])
            residuals = cache[last_key]
            if hasattr(residuals, "cpu"):
                residuals = residuals.cpu().numpy()
            residuals = np.asarray(residuals[0])  # [seq, d]
            lens_residuals[lens_name] = residuals

            # Get probe for this layer (or nearest available)
            probe_key = min(
                state.probes.keys(), key=lambda k: abs(k - layer_idx)
            )
            probe = state.probes[probe_key]
            probe_results = probe.predict_batch(residuals)

            for i, pr in enumerate(probe_results):
                token_points.append(
                    TokenPoint(
                        position=i,
                        token=str_tokens[i] if i < len(str_tokens) else f"[{i}]",
                        entropy=pr.entropy,
                        superposition_score=pr.superposition_score,
                        active_fields=pr.active_fields,
                        top_field=pr.top_field or "unknown",
                    )
                )
        else:
            # No probes — return stub token points
            for i in range(seq_len):
                token_points.append(
                    TokenPoint(
                        position=i,
                        token=str_tokens[i] if i < len(str_tokens) else f"[{i}]",
                        entropy=0.0,
                        superposition_score=0.5,
                        active_fields=[],
                        top_field="unknown",
                    )
                )

        tokens_by_lens[lens_name] = token_points

    # ---- Build divergence and agreement matrices -----------------------
    divergence_map: List[List[float]] = _build_divergence_map(seq_len)
    agreement_matrix: List[List[float]] = _build_agreement_matrix(
        req.lenses, lens_residuals
    )

    # ---- Superposition positions ----------------------------------------
    superposition_positions = _find_superposition_positions(tokens_by_lens)

    # ---- Anamnesis -------------------------------------------------------
    anamnesis_resp: Optional[AnamnesisResponse] = None
    if req.run_anamnesis and state.anamnesis is not None:
        try:
            anamnesis_resp = await _run_anamnesis(req, state, tokens_by_lens)
        except Exception as exc:
            logger.warning("Anamnesis failed: %s", exc)

    return ContrastiveResponse(
        passage=passage,
        page=req.page,
        line=req.line,
        tokens_by_lens=tokens_by_lens,
        divergence_map=divergence_map,
        agreement_matrix=agreement_matrix,
        superposition_positions=superposition_positions,
        lens_names=req.lenses,
        anamnesis=anamnesis_resp,
    )


def _convert_runner_result(
    raw: Any,
    req: PassageRequest,
    state: AppState,
) -> ContrastiveResponse:
    """Convert a MultiPassRunner result dict to ContrastiveResponse."""
    # The runner may return a dict or already a ContrastiveResponse.
    if isinstance(raw, ContrastiveResponse):
        return raw

    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected runner result type: {type(raw)}")

    # Extract tokens_by_lens — handle both raw dicts and TokenPoint objects.
    raw_tbl = raw.get("tokens_by_lens", {})
    tokens_by_lens: Dict[str, List[TokenPoint]] = {}
    for ln, pts in raw_tbl.items():
        converted: List[TokenPoint] = []
        for pt in pts:
            if isinstance(pt, TokenPoint):
                converted.append(pt)
            elif isinstance(pt, dict):
                converted.append(TokenPoint(**pt))
        tokens_by_lens[ln] = converted

    return ContrastiveResponse(
        passage=req.passage,
        page=req.page,
        line=req.line,
        tokens_by_lens=tokens_by_lens,
        divergence_map=raw.get("divergence_map", []),
        agreement_matrix=raw.get("agreement_matrix", []),
        superposition_positions=raw.get("superposition_positions", []),
        lens_names=req.lenses,
        anamnesis=raw.get("anamnesis"),
    )


async def _run_anamnesis(
    req: PassageRequest,
    state: AppState,
    tokens_by_lens: Dict[str, List[TokenPoint]],
) -> Optional[AnamnesisResponse]:
    """Run the anamnesis protocol and return a response."""
    import asyncio

    proto = state.anamnesis
    retrieve_fn = getattr(proto, "retrieve", None)
    if retrieve_fn is None:
        return None

    if asyncio.iscoroutinefunction(retrieve_fn):
        raw = await retrieve_fn(
            passage=req.passage, page=req.page, line=req.line
        )
    else:
        raw = await asyncio.to_thread(
            retrieve_fn, passage=req.passage, page=req.page, line=req.line
        )

    if isinstance(raw, AnamnesisResponse):
        return raw
    if isinstance(raw, dict):
        return AnamnesisResponse(**raw)
    return None


def _build_divergence_map(seq_len: int) -> List[List[float]]:
    """Build a placeholder divergence map (zeros on diagonal)."""
    return [[0.0] * seq_len for _ in range(seq_len)]


def _build_agreement_matrix(
    lens_names: List[str],
    residuals: Dict[str, Any],
) -> List[List[float]]:
    """Build lens-agreement matrix from residual cosine similarities."""
    import numpy as np

    n = len(lens_names)
    mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    if len(residuals) < 2:
        return mat

    lens_list = list(residuals.keys())
    for i, ln_i in enumerate(lens_names):
        if ln_i not in residuals:
            continue
        for j, ln_j in enumerate(lens_names):
            if i == j or ln_j not in residuals:
                continue
            a = residuals[ln_i].flatten()
            b = residuals[ln_j].flatten()
            norm_a = np.linalg.norm(a) + 1e-12
            norm_b = np.linalg.norm(b) + 1e-12
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            mat[i][j] = round(sim, 4)

    return mat


def _find_superposition_positions(
    tokens_by_lens: Dict[str, List[TokenPoint]],
) -> List[int]:
    """Identify token positions where superposition score is low (< 0.4)
    across a majority of lenses."""
    if not tokens_by_lens:
        return []

    lens_names = list(tokens_by_lens.keys())
    if not lens_names:
        return []

    seq_len = len(tokens_by_lens[lens_names[0]])
    superposed: List[int] = []

    for pos in range(seq_len):
        low_super_count = 0
        for ln in lens_names:
            pts = tokens_by_lens[ln]
            if pos < len(pts) and pts[pos].superposition_score < 0.4:
                low_super_count += 1
        if low_super_count > len(lens_names) / 2:
            superposed.append(pos)

    return superposed


def _stub_response(req: PassageRequest) -> ContrastiveResponse:
    """Return a minimal stub when the model is not available."""
    tokens_by_lens: Dict[str, List[TokenPoint]] = {}
    words = req.passage.split()
    for ln in req.lenses:
        tokens_by_lens[ln] = [
            TokenPoint(
                position=i,
                token=w,
                entropy=0.0,
                superposition_score=0.5,
                active_fields=[],
                top_field="unknown",
            )
            for i, w in enumerate(words)
        ]

    n = len(req.lenses)
    agreement = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    seq = len(words)
    divergence = [[0.0] * seq for _ in range(seq)]

    return ContrastiveResponse(
        passage=req.passage,
        page=req.page,
        line=req.line,
        tokens_by_lens=tokens_by_lens,
        divergence_map=divergence,
        agreement_matrix=agreement,
        superposition_positions=[],
        lens_names=req.lenses,
        anamnesis=None,
    )


# ---------------------------------------------------------------------------
# POST /api/wake/lens/compose
# ---------------------------------------------------------------------------

@router.post(
    "/lens/compose",
    summary="Compose multiple lenses into a weighted blend",
    description=(
        "Creates a composite lens configuration by blending the system prompts "
        "and foregrounded fields of the specified lenses according to optional "
        "weights."
    ),
)
async def compose_lenses(
    req: LensComposeRequest,
    state: AppState = Depends(get_state),
) -> Dict[str, Any]:
    """Compose multiple lenses into a single weighted blend."""
    # Validate all requested lenses exist
    missing = [ln for ln in req.lenses if ln not in state.lenses]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Lenses not found: {missing}.  "
                f"Available: {state.available_lenses}"
            ),
        )

    # Determine weights
    weights = req.weights if req.weights is not None else [1.0] * len(req.lenses)
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # Build the composed lens description
    composed_fields: List[str] = []
    composed_probes: List[str] = []
    prompt_parts: List[str] = []

    for i, ln in enumerate(req.lenses):
        lens = state.lenses[ln]
        config = getattr(lens, "config", None)
        if config is None:
            continue

        w = norm_weights[i]
        prompt_parts.append(
            f"[{ln.upper()} lens weight={w:.2f}]\n{config.system_prompt}"
        )
        for f in getattr(config, "foregrounded_fields", []):
            if f not in composed_fields:
                composed_fields.append(f)
        for p in getattr(config, "probe_targets", []):
            if p not in composed_probes:
                composed_probes.append(p)

    composed_prompt = "\n\n---\n\n".join(prompt_parts)

    return {
        "lenses": req.lenses,
        "weights": norm_weights,
        "composed_prompt": composed_prompt,
        "foregrounded_fields": composed_fields,
        "probe_targets": composed_probes,
    }
