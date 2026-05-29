"""
Simplified activation patching for WAKE circuit discovery.

Path patching (also called activation patching or causal tracing) measures
the causal effect of replacing an intermediate activation from a "clean" run
with the corresponding activation from a "corrupted" run (or vice versa).

In the WAKE context:
- The **clean** input is a passage processed under one lens (full semantic
  context).
- The **corrupted** input is the same passage under a different lens or with
  a degraded context.
- Patching layer × position tells us where in the computation the model
  encodes the lens-specific semantic interpretation.

Spec reference: section 4.x (path patching utility)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# PatchingResult
# ---------------------------------------------------------------------------

@dataclass
class PatchingResult:
    """Result from a single activation-patching experiment.

    Attributes
    ----------
    patched_layer:
        The transformer layer at which the activation was replaced.
    patched_position:
        Token position (0-indexed) where the replacement was applied.
    original_logit_diff:
        Value of the metric on the **corrupted** (unpatched) run.
    patched_logit_diff:
        Value of the metric after patching in the clean activation.
    effect:
        ``patched_logit_diff − original_logit_diff``.  A positive value
        means patching in the clean activation *increased* the metric
        (this layer × position carries useful information).
    normalized_effect:
        ``effect / total_effect`` — contribution of this layer × position
        relative to all layers.  Set to 0.0 until
        :meth:`PathPatcher.sweep_layers` normalises it.
    hook_name:
        The TransformerLens hook point used, e.g. ``"hook_resid_post"``.
    """

    patched_layer: int
    patched_position: int
    original_logit_diff: float
    patched_logit_diff: float
    effect: float
    normalized_effect: float
    hook_name: str = "hook_resid_post"


# ---------------------------------------------------------------------------
# PathPatcher
# ---------------------------------------------------------------------------

class PathPatcher:
    """Causal activation patcher for WAKE circuit discovery.

    Tests the causal effect of replacing a specific activation
    (layer × position) from a *clean* run with the corresponding activation
    from a *corrupted* run (a different lens context).

    Parameters
    ----------
    model:
        The WAKE model wrapper.  Must expose:

        - ``model.model`` — the underlying TransformerLens
          ``HookedTransformer`` (or compatible object with a ``hooks``
          context manager and ``__call__``).
        - ``model.forward_with_cache(tokens)`` — returns
          ``(logits, activation_cache)``.
        - ``model.n_layers`` — number of transformer layers.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Single-patch experiment
    # ------------------------------------------------------------------

    def patch_layer_position(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        layer: int,
        position: int,
        hook_name: str,
        metric: Callable[[torch.Tensor], float],
    ) -> PatchingResult:
        """Patch a single (layer, position) activation and measure the effect.

        Procedure
        ---------
        1. Run clean tokens → collect ``clean_cache``.
        2. Run corrupted tokens → collect ``corrupted_cache`` and measure
           the *original* metric.
        3. Re-run corrupted tokens, but hook ``blocks.{layer}.{hook_name}``
           to replace position *position* with the clean activation.
        4. Measure the *patched* metric and compute the effect.

        Parameters
        ----------
        clean_tokens:
            Shape ``[1, seq_len]`` token IDs for the clean (lens A) input.
        corrupted_tokens:
            Shape ``[1, seq_len]`` token IDs for the corrupted (lens B) input.
        layer:
            Layer to patch.
        position:
            Token position to patch (0-indexed).
        hook_name:
            TransformerLens hook-point name within the layer block, e.g.
            ``"hook_resid_post"``, ``"hook_mlp_out"``,
            ``"attn.hook_result"``.
        metric:
            A callable that takes ``logits: torch.Tensor`` (shape
            ``[1, seq_len, vocab_size]``) and returns a scalar float.
            Typically the logit difference between two target tokens.

        Returns
        -------
        PatchingResult
            With ``normalized_effect`` set to 0.0; caller normalises it.
        """
        # Step 1: clean run
        _, clean_cache = self.model.forward_with_cache(clean_tokens)
        clean_activation: torch.Tensor = clean_cache[
            f"blocks.{layer}.{hook_name}"
        ]

        # Step 2: corrupted baseline
        corrupted_logits, _ = self.model.forward_with_cache(corrupted_tokens)
        original_metric = float(metric(corrupted_logits))

        # Step 3: patched run (hook replaces position in corrupted stream)
        # We capture clean_activation in the closure.
        def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN401
            # value: [batch, seq_len, d_model]
            value = value.clone()
            value[:, position] = clean_activation[:, position]
            return value

        full_hook_name = f"blocks.{layer}.{hook_name}"
        with self.model.model.hooks(fwd_hooks=[(full_hook_name, hook_fn)]):
            patched_logits = self.model.model(corrupted_tokens)

        patched_metric = float(metric(patched_logits))

        effect = patched_metric - original_metric

        return PatchingResult(
            patched_layer=layer,
            patched_position=position,
            original_logit_diff=original_metric,
            patched_logit_diff=patched_metric,
            effect=effect,
            normalized_effect=0.0,
            hook_name=hook_name,
        )

    # ------------------------------------------------------------------
    # Layer sweep
    # ------------------------------------------------------------------

    def sweep_layers(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        position: int,
        metric: Callable[[torch.Tensor], float],
        hook_name: str = "hook_resid_post",
    ) -> List[PatchingResult]:
        """Patch every layer at a fixed position and measure each effect.

        After collecting all individual results the method normalises each
        ``normalized_effect`` by the total absolute effect across all layers.

        Parameters
        ----------
        clean_tokens, corrupted_tokens:
            Shape ``[1, seq_len]``.
        position:
            Token position to patch at every layer.
        metric:
            Logit-difference metric (see :meth:`patch_layer_position`).
        hook_name:
            Hook-point name (default ``"hook_resid_post"``).

        Returns
        -------
        List[PatchingResult]
            One result per layer, in layer order.  ``normalized_effect`` is
            set on every result.
        """
        results: List[PatchingResult] = []

        for layer in range(self.model.n_layers):
            r = self.patch_layer_position(
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                layer=layer,
                position=position,
                hook_name=hook_name,
                metric=metric,
            )
            results.append(r)

        # Normalise by total absolute effect
        total = sum(abs(r.effect) for r in results) + 1e-9
        for r in results:
            r.normalized_effect = r.effect / total

        return results

    # ------------------------------------------------------------------
    # Position sweep (fixed layer, all positions)
    # ------------------------------------------------------------------

    def sweep_positions(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        layer: int,
        metric: Callable[[torch.Tensor], float],
        hook_name: str = "hook_resid_post",
    ) -> List[PatchingResult]:
        """Patch every position at a fixed layer and measure each effect.

        Parameters
        ----------
        clean_tokens, corrupted_tokens:
            Shape ``[1, seq_len]``.
        layer:
            Layer to patch at every position.
        metric:
            Logit-difference metric.
        hook_name:
            Hook-point name.

        Returns
        -------
        List[PatchingResult]
            One result per token position.  ``normalized_effect`` is set.
        """
        seq_len = clean_tokens.shape[1]
        results: List[PatchingResult] = []

        for pos in range(seq_len):
            r = self.patch_layer_position(
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                layer=layer,
                position=pos,
                hook_name=hook_name,
                metric=metric,
            )
            results.append(r)

        total = sum(abs(r.effect) for r in results) + 1e-9
        for r in results:
            r.normalized_effect = r.effect / total

        return results

    # ------------------------------------------------------------------
    # Full grid sweep (all layers × all positions)
    # ------------------------------------------------------------------

    def sweep_grid(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        metric: Callable[[torch.Tensor], float],
        hook_name: str = "hook_resid_post",
    ) -> Dict[str, Any]:
        """Patch every (layer, position) pair and return the effect matrix.

        Parameters
        ----------
        clean_tokens, corrupted_tokens:
            Shape ``[1, seq_len]``.
        metric:
            Logit-difference metric.
        hook_name:
            Hook-point name.

        Returns
        -------
        Dict with keys:
            ``"results"``       — List[List[PatchingResult]] indexed [layer][pos]
            ``"effect_matrix"`` — np.ndarray shape [n_layers, seq_len]
            ``"norm_matrix"``   — np.ndarray of normalized_effect values
        """
        import numpy as np

        seq_len = clean_tokens.shape[1]
        n_layers = self.model.n_layers
        all_results: List[List[PatchingResult]] = []
        effect_matrix = np.zeros((n_layers, seq_len))

        for layer in range(n_layers):
            row: List[PatchingResult] = []
            for pos in range(seq_len):
                r = self.patch_layer_position(
                    clean_tokens=clean_tokens,
                    corrupted_tokens=corrupted_tokens,
                    layer=layer,
                    position=pos,
                    hook_name=hook_name,
                    metric=metric,
                )
                row.append(r)
                effect_matrix[layer, pos] = r.effect
            all_results.append(row)

        total = float(np.abs(effect_matrix).sum()) + 1e-9
        norm_matrix = effect_matrix / total

        # Set normalized_effect on every result
        for layer in range(n_layers):
            for pos in range(seq_len):
                all_results[layer][pos].normalized_effect = float(
                    norm_matrix[layer, pos]
                )

        return {
            "results": all_results,
            "effect_matrix": effect_matrix,
            "norm_matrix": norm_matrix,
        }
