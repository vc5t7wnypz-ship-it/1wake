"""
WAKE MultiPass Runner — spec section 5.

Runs a passage through a language model in multiple passes, one per active
lens, then computes a contrastive result that captures the per-pass residual
activations, probe scores, and a superposition analysis.

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
import torch

from interpretability.probes.language_probes import SuperpositionDetector

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PassResult:
    """Result of a single lens pass through the model.

    Attributes
    ----------
    lens_name:
        Name of the :class:`~lenses.base.Lens` used in this pass.
    passage:
        The raw Wake passage text.
    page:
        Source page number.
    line:
        Source line number.
    layer_activations:
        Dict mapping layer index → numpy array of shape
        ``[seq_len, d_model]``.  Contains the residual-stream activations
        captured at the requested layers.
    token_ids:
        HuggingFace tokenizer output: list of integer token IDs.
    passage_start_idx:
        Index within *token_ids* where the Wake passage text begins
        (after the system prompt).
    probe_scores:
        Dict mapping probe name → array of shape ``[seq_len, n_classes]``.
        Populated after probe evaluation; empty until then.
    attention_maps:
        Optional dict mapping layer index → attention tensor of shape
        ``[n_heads, seq_len, seq_len]``.
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
        Output of :class:`SuperpositionDetector.analyze` applied to the
        aggregated probe scores from all passes.
    is_superposed:
        Convenience bool: True when the detector classifies the passage
        as superposed.
    dominant_field:
        The semantic field with the highest aggregate activation across all
        passes.
    lens_agreement:
        Float in [0, 1]: fraction of lens pairs whose dominant field
        agrees.  High agreement → the model has collapsed to a single
        reading.
    entropy_profile:
        Per-token entropy values computed from residual-stream activations
        (shape ``[passage_len]``).
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
    geometry: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# MultiPassRunner
# ---------------------------------------------------------------------------


class MultiPassRunner:
    """Run a Wake passage through a language model under multiple lens passes.

    Parameters
    ----------
    model:
        A HuggingFace ``AutoModelForCausalLM`` (or compatible) model loaded
        with ``output_hidden_states=True``.
    lenses:
        List of :class:`~lenses.base.Lens` instances to use.  One forward
        pass is executed per lens.
    probes:
        Dict mapping probe name → callable that accepts a numpy residual-
        stream array of shape ``[seq_len, d_model]`` and returns a score
        array of shape ``[seq_len, n_classes]``.
    tokenizer:
        A HuggingFace tokenizer (``AutoTokenizer``) compatible with *model*.
    """

    def __init__(
        self,
        model: Any,
        lenses: List[Any],
        probes: Dict[str, Any],
        tokenizer: Any,
    ) -> None:
        self.model = model
        self.lenses = lenses
        self.probes = probes
        self.tokenizer = tokenizer
        self._detector = SuperpositionDetector()

        # Determine device from model (fall back to CPU)
        try:
            self._device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        passage: str,
        wake_tokens: List[Any],
        page: int,
        line: int,
        layers_to_capture: Optional[List[int]] = None,
    ) -> ContrastiveResult:
        """Run *passage* through the model under all registered lenses.

        Parameters
        ----------
        passage:
            Raw Wake passage text.
        wake_tokens:
            List of :class:`~engine.tokenizer.wake_tokenizer.WakeToken`
            objects for *passage*.
        page:
            Source page number.
        line:
            Source line number.
        layers_to_capture:
            Indices of transformer layers from which to capture residual-
            stream activations.  When *None*, captures the middle and final
            layers.

        Returns
        -------
        ContrastiveResult
            Full contrastive analysis across all lens passes.
        """
        pass_results: List[PassResult] = []

        for lens in self.lenses:
            lens_context = lens.get_context_for_passage(passage, wake_tokens)
            full_input = f"{lens_context}\n\nAnalyse: {passage}"

            # Tokenise
            encoding = self.tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids = encoding["input_ids"].to(self._device)

            # Find where the Wake passage starts within input_ids
            passage_start = self._find_passage_start(input_ids, full_input, passage)

            # Forward pass with hidden-state capture
            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            # Select layers to capture
            all_hidden = outputs.hidden_states  # tuple of [1, seq_len, d_model]
            n_layers = len(all_hidden)
            if layers_to_capture is None:
                mid = n_layers // 2
                layers_to_capture = sorted({mid, n_layers - 1})

            layer_activations: Dict[int, np.ndarray] = {}
            for layer_idx in layers_to_capture:
                if 0 <= layer_idx < n_layers:
                    layer_activations[layer_idx] = (
                        all_hidden[layer_idx].squeeze(0).cpu().float().numpy()
                    )

            # Attention maps (last requested layer)
            attention_maps: Dict[int, np.ndarray] = {}
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                for layer_idx in layers_to_capture:
                    if 0 <= layer_idx < len(outputs.attentions):
                        attention_maps[layer_idx] = (
                            outputs.attentions[layer_idx]
                            .squeeze(0)
                            .cpu()
                            .float()
                            .numpy()
                        )

            # Run probes on the final captured layer
            final_layer = max(layer_activations.keys(), default=0)
            probe_scores: Dict[str, np.ndarray] = {}
            if final_layer in layer_activations:
                act = layer_activations[final_layer]
                for probe_name, probe_fn in self.probes.items():
                    try:
                        probe_scores[probe_name] = probe_fn(act)
                    except Exception:
                        pass

            pr = PassResult(
                lens_name=lens.config.name,
                passage=passage,
                page=page,
                line=line,
                layer_activations=layer_activations,
                token_ids=input_ids.squeeze(0).tolist(),
                passage_start_idx=passage_start,
                probe_scores=probe_scores,
                attention_maps=attention_maps,
            )
            pass_results.append(pr)

        return self._compute_contrastive_result(
            passage, page, line, pass_results
        )

    def run_batch(self, passages: List[Dict[str, Any]]) -> List[ContrastiveResult]:
        """Run multiple passages through the model.

        Parameters
        ----------
        passages:
            List of dicts, each with keys:
            ``passage`` (str), ``wake_tokens`` (list), ``page`` (int),
            ``line`` (int).  Optionally ``layers_to_capture`` (list[int]).

        Returns
        -------
        List[ContrastiveResult]
            One result per input passage, in the same order.
        """
        results: List[ContrastiveResult] = []
        for item in passages:
            result = self.run(
                passage=item["passage"],
                wake_tokens=item.get("wake_tokens", []),
                page=item["page"],
                line=item["line"],
                layers_to_capture=item.get("layers_to_capture", None),
            )
            results.append(result)
        return results

    def save_result(self, result: ContrastiveResult, path: Path) -> None:
        """Serialise *result* to JSON at *path*.

        Numpy arrays are converted to nested lists so the output is
        human-readable and portable.

        Parameters
        ----------
        result:
            The :class:`ContrastiveResult` to save.
        path:
            Destination file path.  Parent directories are created if they
            do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj

        pass_results_serialised = []
        for pr in result.pass_results:
            pass_results_serialised.append({
                "lens_name": pr.lens_name,
                "passage": pr.passage,
                "page": pr.page,
                "line": pr.line,
                "layer_activations": {
                    str(k): _convert(v)
                    for k, v in pr.layer_activations.items()
                },
                "token_ids": pr.token_ids,
                "passage_start_idx": pr.passage_start_idx,
                "probe_scores": {
                    k: _convert(v) for k, v in pr.probe_scores.items()
                },
                "attention_maps": {
                    str(k): _convert(v)
                    for k, v in pr.attention_maps.items()
                },
            })

        payload = {
            "passage": result.passage,
            "page": result.page,
            "line": result.line,
            "is_superposed": result.is_superposed,
            "dominant_field": result.dominant_field,
            "lens_agreement": result.lens_agreement,
            "entropy_profile": _convert(result.entropy_profile),
            "superposition_analysis": _convert(result.superposition_analysis),
            "geometry": _convert(result.geometry) if result.geometry else None,
            "pass_results": pass_results_serialised,
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_passage_start(
        input_ids: torch.Tensor,
        full_input: str,
        passage: str,
    ) -> int:
        """Find the token index where *passage* begins in *input_ids*.

        Uses a simple character-level search: locate *passage* within
        *full_input* to get its character offset, then count the tokens
        that precede it.  Falls back to 0 if *passage* is not found.

        Parameters
        ----------
        input_ids:
            Tokenised ``full_input``, shape ``[1, seq_len]`` or ``[seq_len]``.
        full_input:
            The full prompt string that was tokenised to produce *input_ids*.
        passage:
            The passage substring to locate.

        Returns
        -------
        int
            0-based token index of the first token that belongs to *passage*.
        """
        char_offset = full_input.find(passage)
        if char_offset == -1:
            return 0

        prefix = full_input[:char_offset]
        # Count how many tokens the prefix maps to (approximate via
        # character fraction — good enough for analysis purposes).
        seq_len = input_ids.shape[-1]
        if len(full_input) == 0:
            return 0
        token_approx = int(round(len(prefix) / len(full_input) * seq_len))
        return min(token_approx, seq_len - 1)

    def _compute_contrastive_result(
        self,
        passage: str,
        page: int,
        line: int,
        pass_results: List[PassResult],
    ) -> ContrastiveResult:
        """Aggregate *pass_results* into a :class:`ContrastiveResult`.

        The method:
        1. Gathers probe scores from each pass.
        2. Runs the :class:`SuperpositionDetector` on the aggregated scores.
        3. Computes lens agreement (fraction of pairs with same dominant field).
        4. Computes per-token entropy from the last-layer residual stream.

        Parameters
        ----------
        passage:
            Raw Wake passage text.
        page:
            Source page number.
        line:
            Source line number.
        pass_results:
            Completed pass results from all lens passes.

        Returns
        -------
        ContrastiveResult
        """
        if not pass_results:
            return ContrastiveResult(
                passage=passage,
                page=page,
                line=line,
                pass_results=[],
            )

        # ------------------------------------------------------------------
        # 1. Collect dominant fields per pass (from probe scores or fallback)
        # ------------------------------------------------------------------
        dominant_fields_per_pass: List[str] = []
        agg_residual: Optional[np.ndarray] = None

        for pr in pass_results:
            # Use the last captured layer's activations for the main analysis.
            if pr.layer_activations:
                last_layer = max(pr.layer_activations.keys())
                act = pr.layer_activations[last_layer]
                # Take the passage-token sub-sequence only.
                act_passage = act[pr.passage_start_idx:]
                if agg_residual is None:
                    agg_residual = act_passage
                else:
                    # Average across passes (they may differ in seq-length
                    # due to different system prompts — align by min length).
                    min_len = min(agg_residual.shape[0], act_passage.shape[0])
                    agg_residual = (
                        agg_residual[:min_len] + act_passage[:min_len]
                    ) / 2.0

            # Collect dominant field from probe scores
            if pr.probe_scores:
                # Average across all probe outputs to get a single score vector.
                all_score_means = {}
                for probe_name, scores in pr.probe_scores.items():
                    mean_score = float(scores.mean(axis=0).max()) if scores.ndim > 1 else float(scores.mean())
                    all_score_means[probe_name] = mean_score
                if all_score_means:
                    dominant = max(all_score_means, key=all_score_means.__getitem__)
                    dominant_fields_per_pass.append(dominant)
                else:
                    dominant_fields_per_pass.append("unknown")
            else:
                dominant_fields_per_pass.append("unknown")

        # ------------------------------------------------------------------
        # 2. Lens agreement
        # ------------------------------------------------------------------
        n_passes = len(pass_results)
        if n_passes <= 1:
            lens_agreement = 1.0
        else:
            n_pairs = n_passes * (n_passes - 1) // 2
            agree = sum(
                1
                for i in range(n_passes)
                for j in range(i + 1, n_passes)
                if dominant_fields_per_pass[i] == dominant_fields_per_pass[j]
            )
            lens_agreement = float(agree / n_pairs) if n_pairs > 0 else 1.0

        # ------------------------------------------------------------------
        # 3. Superposition detection via residual-stream entropy
        # ------------------------------------------------------------------
        entropy_profile: List[float] = []
        superposition_analysis: Dict[str, Any] = {}
        is_superposed = False
        dominant_field = "unknown"

        if agg_residual is not None and agg_residual.shape[0] > 0:
            # Per-token entropy: softmax over the residual-stream vector,
            # then compute Shannon entropy.
            for token_vec in agg_residual:
                exp_v = np.exp(token_vec - token_vec.max())
                probs = exp_v / (exp_v.sum() + 1e-12)
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                entropy_profile.append(entropy)

            # Run the full SuperpositionDetector.
            # Build a synthetic ProbeResult for each pass from probe scores.
            try:
                from interpretability.probes.language_probes import ProbeResult

                synthetic_results: List[ProbeResult] = []
                for pr in pass_results:
                    if pr.probe_scores:
                        # Average probe scores across all tokens.
                        preds: Dict[str, float] = {}
                        for probe_name, scores_arr in pr.probe_scores.items():
                            arr = np.asarray(scores_arr)
                            preds[probe_name] = float(arr.mean())
                        if preds:
                            top = max(preds, key=preds.__getitem__)
                            synthetic_results.append(
                                ProbeResult(
                                    layer=0,
                                    position=0,
                                    token=passage[:30],
                                    predictions=preds,
                                    confidence=preds[top],
                                    top_field=top,
                                )
                            )

                if synthetic_results:
                    residual_for_detector = agg_residual.mean(axis=0)
                    superposition_analysis = self._detector.analyze(
                        synthetic_results,
                        residual_for_detector,
                        {},
                    )
                    is_superposed = bool(superposition_analysis.get("is_superposed", False))
                    dominant_field = superposition_analysis.get("dominant_field", "unknown")
            except Exception:
                # Detector is best-effort; never fail the run.
                is_superposed = lens_agreement < 0.5
                dominant_field = (
                    dominant_fields_per_pass[0]
                    if dominant_fields_per_pass
                    else "unknown"
                )

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
        )
