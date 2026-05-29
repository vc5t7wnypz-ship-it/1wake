"""
Attention-head profiling and divergence analysis for WAKE interpretability.

Each attention head in a transformer can be characterised by the statistical
structure of its attention pattern.  This module provides:

- :class:`AttentionHeadProfile` — dataclass capturing all computed statistics
  for a single head.
- :class:`AttentionHeadAnalyzer` — methods to profile one head, all heads in a
  layer, and to detect heads whose behaviour differs between two model passes
  (i.e. between two lenses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# AttentionHeadProfile
# ---------------------------------------------------------------------------

@dataclass
class AttentionHeadProfile:
    """Statistical profile of a single attention head.

    Attributes
    ----------
    layer:
        Zero-indexed transformer layer.
    head:
        Zero-indexed head index within *layer*.
    mean_entropy:
        Mean entropy of the attention distribution across source positions.
        Higher entropy → more diffuse attention (head attends broadly);
        lower entropy → sharp, localised attention.
    top_attended_positions:
        The five destination positions with highest mean attention weight
        (averaged across all source positions).  Each entry is
        ``(position_index, mean_weight)``.
    is_induction_head:
        True if heuristics indicate this head performs induction (attends to
        the position that immediately follows the previous occurrence of the
        current token).  Requires token-level analysis; set to False if that
        data is unavailable.
    is_previous_token_head:
        True if the head heavily attends to the immediately preceding token
        (position ``i − 1``) — a common motif in early layers.
    is_name_mover:
        True if heuristics suggest the head copies entity/name information
        from earlier positions.  Currently left as False by default and should
        be set externally using logit-attribution scores.
    special_function:
        One of ``"induction"``, ``"previous_token"``, ``"name_mover"``,
        ``"general"``.
    """

    layer: int
    head: int
    mean_entropy: float
    top_attended_positions: List[Tuple[int, float]]
    is_induction_head: bool
    is_previous_token_head: bool
    is_name_mover: bool
    special_function: str


# ---------------------------------------------------------------------------
# AttentionHeadAnalyzer
# ---------------------------------------------------------------------------

class AttentionHeadAnalyzer:
    """Profile and compare attention heads from cached model activations.

    Parameters
    ----------
    model:
        The WAKE model wrapper.  The analyzer accesses its ``n_layers``
        attribute and expects activation caches keyed by
        ``"blocks.{layer}.attn.hook_pattern"``.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Single-head profiling
    # ------------------------------------------------------------------

    def profile_head(
        self,
        attention_pattern: np.ndarray,
        layer: int,
        head: int,
    ) -> AttentionHeadProfile:
        """Compute a full profile for a single attention head.

        Parameters
        ----------
        attention_pattern:
            Shape ``[seq_len, seq_len]``.  ``attention_pattern[i, j]`` is the
            attention weight from source position *i* to destination position
            *j*.  Rows should sum to 1 (softmax normalised).
        layer:
            Layer index (for labelling only).
        head:
            Head index (for labelling only).

        Returns
        -------
        AttentionHeadProfile
        """
        seq_len = attention_pattern.shape[0]

        # ---- Entropy ---------------------------------------------------
        # Per-source-position entropy, then average.
        eps = 1e-9
        row_entropy = -np.sum(
            attention_pattern * np.log(attention_pattern + eps), axis=-1
        )  # [seq_len]
        mean_entropy = float(row_entropy.mean())

        # ---- Top attended positions (averaged across source positions) --
        mean_attn = attention_pattern.mean(axis=0)  # [seq_len]
        indexed = list(enumerate(mean_attn.tolist()))
        top_positions: List[Tuple[int, float]] = sorted(
            indexed, key=lambda x: x[1], reverse=True
        )[:5]

        # ---- Previous-token head heuristic -----------------------------
        # Average attention weight on position i-1 across all source positions.
        if seq_len > 1:
            prev_token_weights = [
                float(attention_pattern[i, i - 1]) for i in range(1, seq_len)
            ]
            prev_token_score = float(np.mean(prev_token_weights))
        else:
            prev_token_score = 0.0
        is_prev_token = prev_token_score > 0.5

        # ---- Induction head heuristic ----------------------------------
        # True induction detection requires token-level analysis (we need to
        # know which positions contain the same token as the current position).
        # The heuristic here looks for high attention at a consistent negative
        # offset (period detection via autocorrelation of attention patterns).
        is_induction = self._detect_induction(attention_pattern)

        # ---- Name-mover heuristic --------------------------------------
        # Name-mover heads copy entity tokens from earlier positions to the
        # query position.  Without logit attribution we cannot reliably detect
        # this; leave as False (caller can set it after logit-lens analysis).
        is_name_mover = False

        # ---- Special function ------------------------------------------
        special: str = "general"
        if is_induction:
            special = "induction"
        elif is_prev_token:
            special = "previous_token"

        return AttentionHeadProfile(
            layer=layer,
            head=head,
            mean_entropy=mean_entropy,
            top_attended_positions=top_positions,
            is_induction_head=is_induction,
            is_previous_token_head=is_prev_token,
            is_name_mover=is_name_mover,
            special_function=special,
        )

    # ------------------------------------------------------------------
    # All heads in a layer
    # ------------------------------------------------------------------

    def analyze_all_heads(
        self,
        cache: Dict[str, Any],
        layer: int,
    ) -> List[AttentionHeadProfile]:
        """Profile every head in a given layer.

        Parameters
        ----------
        cache:
            Activation cache (e.g. from TransformerLens).  Must contain the
            key ``"blocks.{layer}.attn.hook_pattern"`` with shape
            ``[batch, n_heads, seq_len, seq_len]``.
        layer:
            Which layer to profile.

        Returns
        -------
        List[AttentionHeadProfile]
            One profile per head, in order.
        """
        raw = cache[f"blocks.{layer}.attn.hook_pattern"]
        # Support both torch tensors and numpy arrays
        if hasattr(raw, "cpu"):
            pattern = raw[0].cpu().numpy()  # [n_heads, seq_len, seq_len]
        else:
            pattern = np.asarray(raw[0])

        return [
            self.profile_head(pattern[h], layer, h)
            for h in range(pattern.shape[0])
        ]

    # ------------------------------------------------------------------
    # Divergent-head detection
    # ------------------------------------------------------------------

    def find_divergent_heads(
        self,
        cache_a: Dict[str, Any],
        cache_b: Dict[str, Any],
        layer: int,
        threshold: float = 0.3,
    ) -> List[int]:
        """Find heads whose attention patterns differ significantly between
        two passes (e.g. two different lenses).

        Uses the symmetric KL divergence between the per-head attention
        distributions from the two caches.  A head is considered "divergent"
        if its mean symmetric KL exceeds *threshold*.

        Parameters
        ----------
        cache_a, cache_b:
            Activation caches from two separate forward passes.
        layer:
            Layer to compare.
        threshold:
            KL-divergence threshold above which a head is flagged as
            divergent.

        Returns
        -------
        List[int]
            Indices of divergent heads.
        """
        def _to_numpy(cache: Dict[str, Any]) -> np.ndarray:
            raw = cache[f"blocks.{layer}.attn.hook_pattern"]
            if hasattr(raw, "cpu"):
                return raw[0].cpu().numpy()
            return np.asarray(raw[0])

        pat_a = _to_numpy(cache_a)  # [n_heads, seq, seq]
        pat_b = _to_numpy(cache_b)

        n_heads = pat_a.shape[0]
        eps = 1e-9
        divergent: List[int] = []

        for h in range(n_heads):
            a = pat_a[h] + eps  # [seq, seq]
            b = pat_b[h] + eps

            # KL(A||B) + KL(B||A) summed over seq × seq, averaged over rows
            kl_ab = np.sum(a * np.log(a / b), axis=-1)   # [seq]
            kl_ba = np.sum(b * np.log(b / a), axis=-1)   # [seq]
            sym_kl = (kl_ab + kl_ba).mean()

            if float(sym_kl) > threshold:
                divergent.append(h)

        return divergent

    # ------------------------------------------------------------------
    # Induction-head heuristic (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_induction(attention_pattern: np.ndarray) -> bool:
        """Simple heuristic: check for periodic diagonal structure.

        An induction head attends to position ``i - period + 1`` (the token
        after the previous occurrence of the current token).  In a random
        sequence the period is 1, so we look for super-diagonal mass.

        This is a simplified heuristic; definitive detection requires
        constructing the prefix-match matrix from the actual token IDs.
        """
        seq_len = attention_pattern.shape[0]
        if seq_len < 4:
            return False

        # Look for elevated weight on diagonal offset +1 from the anti-diagonal
        # (i.e., attend to one position after where prev-token head attends).
        # More precisely: attention_pattern[i, i+1 - period] for period = 1..seq-1.
        # Quick proxy: high variance in which position each row attends to, but
        # with a consistent shift.

        # Measure: for each row i, what's the argmax destination?
        argmax_cols = np.argmax(attention_pattern, axis=1)  # [seq]

        # Induction heads often have argmax_col ≈ argmax_col[i-1] + 1 for i > 1
        # (they follow the token one step after where the previous match landed).
        if seq_len < 3:
            return False

        shifts = argmax_cols[1:].astype(int) - argmax_cols[:-1].astype(int)
        # If majority of shifts are +1 (the head consistently moves forward),
        # this is suggestive of induction behaviour.
        fraction_forward = float(np.mean(shifts == 1))
        return fraction_forward > 0.4

    # ------------------------------------------------------------------
    # Utility: compute attention entropy for external use
    # ------------------------------------------------------------------

    @staticmethod
    def attention_entropy(pattern: np.ndarray) -> np.ndarray:
        """Compute per-position Shannon entropy of an attention pattern.

        Parameters
        ----------
        pattern:
            Shape ``[seq_len, seq_len]`` or ``[n_heads, seq_len, seq_len]``.

        Returns
        -------
        np.ndarray
            If input is 2-D: shape ``[seq_len]`` (entropy per source position).
            If input is 3-D: shape ``[n_heads, seq_len]``.
        """
        eps = 1e-9
        return -np.sum(pattern * np.log(pattern + eps), axis=-1)

    # ------------------------------------------------------------------
    # Utility: format a profile as a human-readable string
    # ------------------------------------------------------------------

    @staticmethod
    def describe_head(profile: AttentionHeadProfile) -> str:
        """Return a one-line text description of *profile*."""
        top5 = ", ".join(
            f"{pos}({w:.3f})" for pos, w in profile.top_attended_positions[:3]
        )
        flags: List[str] = []
        if profile.is_induction_head:
            flags.append("induction")
        if profile.is_previous_token_head:
            flags.append("prev-token")
        if profile.is_name_mover:
            flags.append("name-mover")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return (
            f"L{profile.layer}H{profile.head}{flag_str}: "
            f"entropy={profile.mean_entropy:.3f}, "
            f"top_positions=[{top5}]"
        )
