"""
WAKE MultiPass Runner — spec section 5.
=========================================
Runs a Wake passage through a language model in multiple passes, one per
active lens, then computes a contrastive result that captures the per-pass
residual activations, probe scores, and a superposition analysis.

Each *pass* uses a different lens-conditioned system prompt.  The contrastive
result captures whether the model's residual stream converges (collapsed) or
maintains genuine distributional superposition across the lens variants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

from engine.superposition.detector import (
    SuperpositionDetector,
    SuperpositionResult,
    compute_superposition_score,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PassResult:
    """Result of a single lens pass through the model.

    Attributes
    ----------
    lens_name:
        Name of the lens used in this pass.
    passage:
        The raw Wake passage text.
    page:
        Source page number.
    line:
        Source line number.
    layer_activations:
        Dict mapping layer index → numpy array of shape
        ``(seq_len, d_model)``.  Contains residual-stream activations
        captured at the requested layers.
    token_ids:
        List of integer token IDs produced by the tokenizer.
    passage_start_idx:
        Index within *token_ids* where the Wake passage text begins
        (after the system prompt / lens context).
    probe_scores:
        Dict mapping probe name → array of shape ``(seq_len,)`` or
        ``(seq_len, n_classes)``.  Populated after probe evaluation.
    attention_maps:
        Optional dict mapping layer index → attention tensor of shape
        ``(n_heads, seq_len, seq_len)``.
    superposition_results:
        Optional list of per-token :class:`SuperpositionResult` objects
        produced by running the detector on this pass's activations.
    """

    lens_name: str
    passage: str
    page: int
    line: int
    layer_activations: Dict[int, np.ndarray] = field(default_factory=dict)
    token_ids: List[int] = field(default_factory=list)
    passage_start_idx: int = 0
    probe_scores: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_maps: Dict[int, np.ndarray] = field(default_factory=dict)
    superposition_results: List[SuperpositionResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary (numpy → lists)."""
        return {
            "lens_name": self.lens_name,
            "passage": self.passage,
            "page": self.page,
            "line": self.line,
            "layer_activations": {
                str(k): v.tolist() for k, v in self.layer_activations.items()
            },
            "token_ids": self.token_ids,
            "passage_start_idx": self.passage_start_idx,
            "probe_scores": {
                k: v.tolist() for k, v in self.probe_scores.items()
            },
            "attention_maps": {
                str(k): v.tolist() for k, v in self.attention_maps.items()
            },
            "superposition_results": [
                r.to_dict() for r in self.superposition_results
            ],
        }


@dataclass
class ContrastiveResult:
    """Contrastive analysis across all lens passes for a single passage.

    Attributes
    ----------
    passage:
        The raw Wake passage text.
    page:
        Source page number.
    line:
        Source line number.
    pass_results:
        One :class:`PassResult` per lens pass.
    superposition_analysis:
        Aggregated superposition analysis dict.
    is_superposed:
        Convenience bool: True when the detector classifies the passage
        as superposed across lens passes.
    dominant_field:
        The semantic field with the highest aggregate activation across all
        passes.
    lens_agreement:
        Float in [0, 1]: fraction of lens pairs whose dominant field agrees.
        High agreement indicates model has collapsed to a single reading.
    entropy_profile:
        Per-token entropy values computed from residual-stream activations
        (one float per passage token).
    mean_superposition_score:
        Average superposition score across all token positions and passes.
    geometry:
        Optional dict containing PCA / UMAP projections for visualisation.
    """

    passage: str
    page: int
    line: int
    pass_results: List[PassResult] = field(default_factory=list)
    superposition_analysis: Dict[str, Any] = field(default_factory=dict)
    is_superposed: bool = False
    dominant_field: str = "unknown"
    lens_agreement: float = 0.0
    entropy_profile: List[float] = field(default_factory=list)
    mean_superposition_score: float = 0.0
    geometry: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "passage": self.passage,
            "page": self.page,
            "line": self.line,
            "is_superposed": self.is_superposed,
            "dominant_field": self.dominant_field,
            "lens_agreement": self.lens_agreement,
            "entropy_profile": self.entropy_profile,
            "mean_superposition_score": self.mean_superposition_score,
            "superposition_analysis": self.superposition_analysis,
            "geometry": self.geometry,
            "pass_results": [pr.to_dict() for pr in self.pass_results],
        }


# ---------------------------------------------------------------------------
# MultiPassRunner
# ---------------------------------------------------------------------------

class MultiPassRunner:
    """Run a Wake passage through a language model under multiple lens passes.

    Parameters
    ----------
    model:
        A language model compatible with HuggingFace ``AutoModelForCausalLM``
        (``output_hidden_states=True`` required) or a
        :class:`engine.model.WakeModel` instance.
    lenses:
        List of lens objects.  Each must expose a
        ``get_context_for_passage(passage, tokens)`` method that returns a
        lens-conditioned system-prompt string, and a ``config.name`` attribute.
    probes:
        Dict mapping probe name → callable.  Each callable accepts a numpy
        array of shape ``(seq_len, d_model)`` and returns scores of shape
        ``(seq_len,)`` or ``(seq_len, n_classes)``.
    tokenizer:
        A HuggingFace tokenizer (``AutoTokenizer``) compatible with *model*.
    probe_directions:
        Optional dict mapping field names to numpy probe-direction vectors
        for use with the built-in :class:`SuperpositionDetector`.
    activation_threshold:
        Minimum probe activation to consider a field active (default 0.3).
    superposition_threshold:
        Minimum superposition score to classify as superposed (default 0.4).
    """

    def __init__(
        self,
        model: Any,
        lenses: List[Any],
        probes: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
        probe_directions: Optional[Dict[str, np.ndarray]] = None,
        activation_threshold: float = 0.3,
        superposition_threshold: float = 0.4,
    ) -> None:
        self.model = model
        self.lenses = lenses
        self.probes = probes or {}
        self.tokenizer = tokenizer
        self._detector = SuperpositionDetector(
            probe_directions=probe_directions or {},
            activation_threshold=activation_threshold,
            superposition_threshold=superposition_threshold,
        )

        # Determine device from model
        self._device: Any = None
        if _TORCH_AVAILABLE and model is not None:
            try:
                self._device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self._device = torch.device("cpu")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        passage: str,
        wake_tokens: Optional[List[Any]] = None,
        page: int = 0,
        line: int = 0,
        layers_to_capture: Optional[List[int]] = None,
    ) -> ContrastiveResult:
        """Run *passage* through the model under all registered lenses.

        Parameters
        ----------
        passage:
            Raw Wake passage text.
        wake_tokens:
            Optional list of :class:`engine.tokenizer.WakeToken` objects.
            Used by lenses that need token-level metadata.
        page:
            Source page number.
        line:
            Source line number.
        layers_to_capture:
            Layer indices from which to capture residual-stream activations.
            When *None*, captures the middle and final layers.

        Returns
        -------
        :class:`ContrastiveResult`
            Full contrastive analysis across all lens passes.
        """
        if wake_tokens is None:
            wake_tokens = []

        pass_results: List[PassResult] = []

        for lens in self.lenses:
            pr = self._run_single_pass(
                lens=lens,
                passage=passage,
                wake_tokens=wake_tokens,
                page=page,
                line=line,
                layers_to_capture=layers_to_capture,
            )
            pass_results.append(pr)

        return self._compute_contrastive_result(
            passage=passage,
            page=page,
            line=line,
            pass_results=pass_results,
        )

    def run_batch(
        self,
        passages: List[Dict[str, Any]],
    ) -> List[ContrastiveResult]:
        """Run multiple passages through the model.

        Parameters
        ----------
        passages:
            List of dicts, each with keys:
            ``"passage"`` (str), ``"page"`` (int), ``"line"`` (int).
            Optional keys: ``"wake_tokens"`` (list), ``"layers_to_capture"``
            (list[int]).

        Returns
        -------
        List of :class:`ContrastiveResult`, one per input passage.
        """
        results: List[ContrastiveResult] = []
        for item in passages:
            result = self.run(
                passage=item["passage"],
                wake_tokens=item.get("wake_tokens", []),
                page=item.get("page", 0),
                line=item.get("line", 0),
                layers_to_capture=item.get("layers_to_capture"),
            )
            results.append(result)
        return results

    def save_result(self, result: ContrastiveResult, path: Path) -> None:
        """Serialise *result* to a JSON file at *path*.

        Numpy arrays are converted to nested lists for portability.

        Parameters
        ----------
        result:
            The :class:`ContrastiveResult` to save.
        path:
            Destination file path.  Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)

    @staticmethod
    def load_result(path: Path) -> ContrastiveResult:
        """Load a :class:`ContrastiveResult` from a JSON file.

        Parameters
        ----------
        path:
            Source file path created by :meth:`save_result`.

        Returns
        -------
        :class:`ContrastiveResult` with numpy arrays reconstructed.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)

        pass_results: List[PassResult] = []
        for pr_data in data.get("pass_results", []):
            layer_activations = {
                int(k): np.array(v, dtype=np.float32)
                for k, v in pr_data.get("layer_activations", {}).items()
            }
            probe_scores = {
                k: np.array(v, dtype=np.float32)
                for k, v in pr_data.get("probe_scores", {}).items()
            }
            attention_maps = {
                int(k): np.array(v, dtype=np.float32)
                for k, v in pr_data.get("attention_maps", {}).items()
            }
            pass_results.append(
                PassResult(
                    lens_name=pr_data.get("lens_name", ""),
                    passage=pr_data.get("passage", ""),
                    page=int(pr_data.get("page", 0)),
                    line=int(pr_data.get("line", 0)),
                    layer_activations=layer_activations,
                    token_ids=pr_data.get("token_ids", []),
                    passage_start_idx=int(pr_data.get("passage_start_idx", 0)),
                    probe_scores=probe_scores,
                    attention_maps=attention_maps,
                )
            )

        return ContrastiveResult(
            passage=data.get("passage", ""),
            page=int(data.get("page", 0)),
            line=int(data.get("line", 0)),
            pass_results=pass_results,
            superposition_analysis=data.get("superposition_analysis", {}),
            is_superposed=bool(data.get("is_superposed", False)),
            dominant_field=data.get("dominant_field", "unknown"),
            lens_agreement=float(data.get("lens_agreement", 0.0)),
            entropy_profile=list(data.get("entropy_profile", [])),
            mean_superposition_score=float(
                data.get("mean_superposition_score", 0.0)
            ),
            geometry=data.get("geometry"),
        )

    # ------------------------------------------------------------------
    # Single-pass execution
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        lens: Any,
        passage: str,
        wake_tokens: List[Any],
        page: int,
        line: int,
        layers_to_capture: Optional[List[int]],
    ) -> PassResult:
        """Execute one forward pass conditioned on *lens*.

        Handles both HuggingFace ``AutoModelForCausalLM`` models and the
        engine's own :class:`engine.model.WakeModel` wrapper.
        """
        # Build the lens-conditioned prompt
        try:
            lens_context: str = lens.get_context_for_passage(passage, wake_tokens)
        except (AttributeError, TypeError):
            lens_context = ""

        lens_name: str = ""
        try:
            lens_name = lens.config.name
        except AttributeError:
            try:
                lens_name = lens.name
            except AttributeError:
                lens_name = repr(lens)

        if lens_context:
            full_input = f"{lens_context}\n\nAnalyse: {passage}"
        else:
            full_input = passage

        # --- Tokenise ---
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            # Move to model device if available
            if _TORCH_AVAILABLE and self._device is not None:
                input_ids = encoding["input_ids"].to(self._device)
            else:
                input_ids = encoding["input_ids"]
            token_ids: List[int] = input_ids.squeeze(0).tolist()
        elif _TORCH_AVAILABLE:
            # Fallback: no tokenizer — use dummy single-token input
            import torch as _torch
            input_ids = _torch.zeros(1, 1, dtype=_torch.long)
            token_ids = [0]
        else:
            token_ids = []
            input_ids = None

        # --- Find passage start index ---
        passage_start = self._find_passage_start_idx(
            full_input=full_input,
            passage=passage,
            n_tokens=len(token_ids),
        )

        # --- Forward pass ---
        layer_activations: Dict[int, np.ndarray] = {}
        attention_maps: Dict[int, np.ndarray] = {}

        if self.model is not None and input_ids is not None and _TORCH_AVAILABLE:
            import torch as _torch
            with _torch.no_grad():
                try:
                    # WakeModel path
                    from engine.model.wake_model import WakeModel
                    if isinstance(self.model, WakeModel):
                        logits, cache = self.model.run_with_cache(
                            input_ids.squeeze(0)
                        )
                        n_layers = self.model.n_layers
                        if layers_to_capture is None:
                            mid = n_layers // 2
                            layers_to_capture = sorted({mid, n_layers - 1})
                        for li in layers_to_capture:
                            if 0 <= li < n_layers:
                                resid_key = f"blocks.{li}.hook_resid_post"
                                if resid_key in cache.cache_dict:
                                    layer_activations[li] = (
                                        cache[resid_key][0]
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                                attn_key = f"blocks.{li}.attn.hook_pattern"
                                if attn_key in cache.cache_dict:
                                    attention_maps[li] = (
                                        cache[attn_key][0]
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                    else:
                        # HuggingFace model path
                        outputs = self.model(
                            input_ids,
                            output_hidden_states=True,
                            output_attentions=True,
                        )
                        all_hidden = outputs.hidden_states
                        n_layers = len(all_hidden)
                        if layers_to_capture is None:
                            mid = n_layers // 2
                            layers_to_capture = sorted({mid, n_layers - 1})
                        for li in layers_to_capture:
                            if 0 <= li < n_layers:
                                layer_activations[li] = (
                                    all_hidden[li]
                                    .squeeze(0)
                                    .float()
                                    .cpu()
                                    .numpy()
                                )
                        if hasattr(outputs, "attentions") and outputs.attentions:
                            for li in layers_to_capture:
                                if 0 <= li < len(outputs.attentions):
                                    attention_maps[li] = (
                                        outputs.attentions[li]
                                        .squeeze(0)
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                except Exception:
                    # Best-effort: don't crash the whole run
                    pass

        # --- Run probes ---
        probe_scores: Dict[str, np.ndarray] = {}
        if layer_activations:
            final_layer = max(layer_activations.keys())
            act = layer_activations[final_layer]
            for probe_name, probe_fn in self.probes.items():
                try:
                    probe_scores[probe_name] = np.asarray(probe_fn(act), dtype=float)
                except Exception:
                    pass

        # --- Superposition detection on passage tokens ---
        super_results: List[SuperpositionResult] = []
        if self._detector.probe_directions and layer_activations:
            final_layer = max(layer_activations.keys())
            act = layer_activations[final_layer]
            passage_act = act[passage_start:]
            if passage_act.shape[0] > 0:
                # Create minimal token-like objects for batch_analyze
                dummy_tokens = [
                    _DummyToken(surface="_", page=page, line=line, position=i)
                    for i in range(passage_act.shape[0])
                ]
                super_results = self._detector.batch_analyze(
                    tokens=dummy_tokens,
                    residuals=passage_act,
                )

        return PassResult(
            lens_name=lens_name,
            passage=passage,
            page=page,
            line=line,
            layer_activations=layer_activations,
            token_ids=token_ids,
            passage_start_idx=passage_start,
            probe_scores=probe_scores,
            attention_maps=attention_maps,
            superposition_results=super_results,
        )

    # ------------------------------------------------------------------
    # Contrastive aggregation
    # ------------------------------------------------------------------

    def _compute_contrastive_result(
        self,
        passage: str,
        page: int,
        line: int,
        pass_results: List[PassResult],
    ) -> ContrastiveResult:
        """Aggregate per-pass results into a :class:`ContrastiveResult`.

        Steps:
        1. Average the passage-aligned activations across all passes.
        2. Collect the dominant field per pass (from probe scores).
        3. Compute lens agreement (fraction of pairs with matching dominant).
        4. Compute per-token entropy from the averaged residual stream.
        5. Compute the mean superposition score across all superposition results.
        """
        if not pass_results:
            return ContrastiveResult(
                passage=passage, page=page, line=line
            )

        # ------------------------------------------------------------------
        # 1. Average residuals across passes (aligned to shortest seq)
        # ------------------------------------------------------------------
        agg_residual: Optional[np.ndarray] = None
        for pr in pass_results:
            if not pr.layer_activations:
                continue
            last_layer = max(pr.layer_activations.keys())
            act = pr.layer_activations[last_layer]
            passage_act = act[pr.passage_start_idx:]
            if passage_act.shape[0] == 0:
                continue
            if agg_residual is None:
                agg_residual = passage_act.copy()
            else:
                min_len = min(agg_residual.shape[0], passage_act.shape[0])
                agg_residual = (
                    agg_residual[:min_len] + passage_act[:min_len]
                ) / 2.0

        # ------------------------------------------------------------------
        # 2. Dominant field per pass
        # ------------------------------------------------------------------
        dominant_per_pass: List[str] = []
        for pr in pass_results:
            if pr.probe_scores:
                means: Dict[str, float] = {}
                for name, arr in pr.probe_scores.items():
                    arr_np = np.asarray(arr)
                    means[name] = float(arr_np.mean())
                if means:
                    dominant_per_pass.append(
                        max(means, key=means.__getitem__)
                    )
                    continue
            # Fallback: use top superposition field
            if pr.superposition_results:
                fields = [
                    r.dominant_field
                    for r in pr.superposition_results
                    if r.dominant_field
                ]
                if fields:
                    from collections import Counter
                    dominant_per_pass.append(Counter(fields).most_common(1)[0][0])
                    continue
            dominant_per_pass.append("unknown")

        # ------------------------------------------------------------------
        # 3. Lens agreement
        # ------------------------------------------------------------------
        n = len(pass_results)
        if n <= 1:
            lens_agreement = 1.0
        else:
            n_pairs = n * (n - 1) // 2
            agree = sum(
                1
                for i in range(n)
                for j in range(i + 1, n)
                if dominant_per_pass[i] == dominant_per_pass[j]
            )
            lens_agreement = float(agree / n_pairs) if n_pairs > 0 else 1.0

        # ------------------------------------------------------------------
        # 4. Per-token entropy
        # ------------------------------------------------------------------
        entropy_profile: List[float] = []
        if agg_residual is not None and agg_residual.shape[0] > 0:
            for token_vec in agg_residual:
                max_v = token_vec.max()
                exp_v = np.exp(token_vec - max_v)
                probs = exp_v / (exp_v.sum() + 1e-12)
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                entropy_profile.append(entropy)

        # ------------------------------------------------------------------
        # 5. Mean superposition score
        # ------------------------------------------------------------------
        all_scores: List[float] = []
        for pr in pass_results:
            for sr in pr.superposition_results:
                all_scores.append(sr.superposition_score)
        mean_superposition_score = (
            float(np.mean(all_scores)) if all_scores else 0.0
        )
        is_superposed = mean_superposition_score >= self._detector.superposition_threshold

        # ------------------------------------------------------------------
        # Dominant field (overall)
        # ------------------------------------------------------------------
        from collections import Counter
        field_counts: Counter[str] = Counter(dominant_per_pass)
        if field_counts:
            dominant_field = field_counts.most_common(1)[0][0]
        else:
            dominant_field = "unknown"

        # ------------------------------------------------------------------
        # Superposition analysis summary dict
        # ------------------------------------------------------------------
        superposition_analysis: Dict[str, Any] = {
            "mean_superposition_score": mean_superposition_score,
            "is_superposed": is_superposed,
            "dominant_field": dominant_field,
            "lens_agreement": lens_agreement,
            "n_passes": n,
            "dominant_per_pass": dominant_per_pass,
        }

        return ContrastiveResult(
            passage=passage,
            page=page,
            line=line,
            pass_results=pass_results,
            superposition_analysis=superposition_analysis,
            is_superposed=is_superposed,
            dominant_field=dominant_field,
            lens_agreement=lens_agreement,
            entropy_profile=entropy_profile,
            mean_superposition_score=mean_superposition_score,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_passage_start_idx(
        full_input: str,
        passage: str,
        n_tokens: int,
    ) -> int:
        """Approximate the token index where *passage* begins in *full_input*.

        Uses character-offset proportion as a quick heuristic.

        Returns
        -------
        int: token index (0-based), clamped to [0, n_tokens - 1].
        """
        char_offset = full_input.find(passage)
        if char_offset <= 0 or len(full_input) == 0:
            return 0
        fraction = char_offset / len(full_input)
        idx = int(round(fraction * n_tokens))
        return max(0, min(idx, n_tokens - 1))


# ---------------------------------------------------------------------------
# Internal helper — minimal duck-typed token for batch_analyze
# ---------------------------------------------------------------------------

class _DummyToken:
    """Minimal token-like object used when real WakeTokens are unavailable."""

    __slots__ = ("surface", "page", "line", "position")

    def __init__(
        self,
        surface: str,
        page: int,
        line: int,
        position: int,
    ) -> None:
        self.surface = surface
        self.page = page
        self.line = line
        self.position = position
