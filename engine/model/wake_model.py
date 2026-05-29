"""
WAKE Engine — WakeModel
=======================
Wrapper around TransformerLens ``HookedTransformer`` for mechanistic
interpretability of Finnegans Wake token processing.

Spec reference: section 4.1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# TransformerLens is the primary dependency.  We guard the import to allow
# the module to be imported (and type-checked) in environments where the
# library is not installed, falling back to a clear ImportError at runtime.
try:
    from transformer_lens import HookedTransformer  # type: ignore[import]
    from transformer_lens.utilities import devices  # type: ignore[import]
    _TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    _TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# WakeModel
# ---------------------------------------------------------------------------

class WakeModel:
    """Thin wrapper around a TransformerLens HookedTransformer.

    Provides convenience methods for:
    - Running a forward pass with full activation caching.
    - Extracting per-layer residual streams, attention patterns, and MLP
      outputs at specific sequence positions.
    - Decoding the top-k predicted tokens from a logit tensor.

    Parameters
    ----------
    model:
        A pre-loaded :class:`HookedTransformer` instance.
    device:
        Torch device string (e.g. ``"cuda"`` or ``"cpu"``).  If *None* the
        device is inferred from the model's existing parameter placement.
    """

    def __init__(
        self,
        model: Any,  # HookedTransformer
        device: Optional[str] = None,
    ) -> None:
        if not _TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "TransformerLens is required.  Install it with: "
                "pip install transformer-lens"
            )
        self._model: Any = model  # HookedTransformer
        self.device = device or str(next(model.parameters()).device)
        self.n_layers: int = model.cfg.n_layers
        self.d_model: int = model.cfg.d_model
        self.n_heads: int = model.cfg.n_heads
        self.d_head: int = model.cfg.d_head
        self.vocab_size: int = model.cfg.d_vocab

        self._names_filter: List[str] = self._build_names_filter()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> "WakeModel":
        """Load a pretrained model from HuggingFace via TransformerLens.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier, e.g. ``"meta-llama/Llama-3-8B"``.
        device:
            Target device.  Defaults to CUDA if available, else CPU.
        hf_token:
            HuggingFace access token for gated models (e.g. Llama).  Falls
            back to the ``HF_TOKEN`` environment variable when *None*.
        dtype:
            Model weight dtype.  Use ``torch.float16`` / ``bfloat16`` to
            reduce VRAM when running on GPU.
        **kwargs:
            Additional keyword arguments forwarded to
            ``HookedTransformer.from_pretrained``.

        Returns
        -------
        :class:`WakeModel`
        """
        if not _TRANSFORMER_LENS_AVAILABLE:
            raise ImportError(
                "TransformerLens is required.  Install it with: "
                "pip install transformer-lens"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve HuggingFace token
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        load_kwargs: Dict[str, Any] = {
            "device": device,
            "dtype": dtype,
            **kwargs,
        }
        if token:
            load_kwargs["hf_token"] = token

        tl_model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
        return cls(model=tl_model, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_names_filter(self) -> List[str]:
        """Build the list of hook names whose activations are cached by default."""
        names_filter = (
            ["hook_embed", "hook_pos_embed"]
            + [f"blocks.{i}.hook_resid_pre" for i in range(self.n_layers)]
            + [f"blocks.{i}.hook_resid_post" for i in range(self.n_layers)]
            + [f"blocks.{i}.attn.hook_pattern" for i in range(self.n_layers)]
            + [f"blocks.{i}.attn.hook_z" for i in range(self.n_layers)]
            + [f"blocks.{i}.mlp.hook_post" for i in range(self.n_layers)]
        )
        return names_filter

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Run a forward pass and return (logits, cache).

        Parameters
        ----------
        tokens:
            Integer token tensor of shape ``(batch, seq_len)`` or
            ``(seq_len,)`` (will be unsqueezed automatically).
        names_filter:
            Hook names to cache.  Defaults to :attr:`_names_filter`.

        Returns
        -------
        ``(logits, cache)``
            *logits* has shape ``(batch, seq_len, vocab_size)``.
            *cache* is a TransformerLens ``ActivationCache`` object.
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        filter_list = names_filter or self._names_filter
        logits, cache = self._model.run_with_cache(
            tokens,
            names_filter=filter_list,
            return_type="logits",
        )
        return logits, cache

    # ------------------------------------------------------------------
    # Activation accessors
    # ------------------------------------------------------------------

    def get_residual(
        self,
        cache: Any,
        layer: int,
        position: int,
        which: str = "post",
    ) -> np.ndarray:
        """Extract the residual stream vector at a given layer and position.

        Parameters
        ----------
        cache:
            ActivationCache returned by :meth:`run_with_cache`.
        layer:
            Layer index (0-based).  Use -1 to access the final layer post-norm.
        position:
            Sequence position index (0-based).
        which:
            ``"pre"`` or ``"post"`` (default ``"post"``).

        Returns
        -------
        1-D numpy array of shape ``(d_model,)``.
        """
        key = f"blocks.{layer}.hook_resid_{which}"
        # cache[key] has shape (batch, seq_len, d_model)
        vec: torch.Tensor = cache[key][0, position]
        return vec.float().cpu().numpy()

    def get_mlp_output(
        self,
        cache: Any,
        layer: int,
        position: int,
    ) -> np.ndarray:
        """Extract MLP post-activation vector at *layer* and *position*.

        Parameters
        ----------
        cache:
            ActivationCache returned by :meth:`run_with_cache`.
        layer:
            Layer index (0-based).
        position:
            Sequence position index (0-based).

        Returns
        -------
        1-D numpy array of shape ``(d_mlp,)``.
        """
        key = f"blocks.{layer}.mlp.hook_post"
        # cache[key] has shape (batch, seq_len, d_mlp)
        vec: torch.Tensor = cache[key][0, position]
        return vec.float().cpu().numpy()

    def get_head_output(
        self,
        cache: Any,
        layer: int,
        head: int,
        position: int,
    ) -> np.ndarray:
        """Extract the output of a specific attention head at *position*.

        The ``hook_z`` tensor stores the per-head value-weighted sum
        (shape ``batch × seq_len × n_heads × d_head``).  This method
        returns the raw z-vector before the output projection.

        Parameters
        ----------
        cache:
            ActivationCache returned by :meth:`run_with_cache`.
        layer:
            Layer index (0-based).
        head:
            Head index (0-based).
        position:
            Sequence position index (0-based).

        Returns
        -------
        1-D numpy array of shape ``(d_head,)``.
        """
        key = f"blocks.{layer}.attn.hook_z"
        # cache[key] has shape (batch, seq_len, n_heads, d_head)
        vec: torch.Tensor = cache[key][0, position, head]
        return vec.float().cpu().numpy()

    def get_attention_pattern(
        self,
        cache: Any,
        layer: int,
    ) -> np.ndarray:
        """Return the attention pattern for a layer.

        Parameters
        ----------
        cache:
            ActivationCache returned by :meth:`run_with_cache`.
        layer:
            Layer index (0-based).

        Returns
        -------
        Numpy array of shape ``(n_heads, seq_len, seq_len)``.
        """
        key = f"blocks.{layer}.attn.hook_pattern"
        # cache[key] has shape (batch, n_heads, seq_len, seq_len)
        pattern: torch.Tensor = cache[key][0]
        return pattern.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Token operations
    # ------------------------------------------------------------------

    def decode_top_k(
        self,
        logits: torch.Tensor,
        k: int = 10,
        position: int = -1,
    ) -> List[Tuple[str, float]]:
        """Return the top-k predicted tokens with softmax probabilities.

        Parameters
        ----------
        logits:
            Logit tensor of shape ``(batch, seq_len, vocab_size)`` or
            ``(seq_len, vocab_size)`` or ``(vocab_size,)``.
        k:
            Number of top predictions to return (default 10).
        position:
            Which sequence position to read logits from when *logits* has
            a sequence dimension.  Defaults to -1 (last token).

        Returns
        -------
        List of ``(token_string, probability)`` tuples, sorted descending
        by probability.
        """
        # Normalise to 1-D
        if logits.dim() == 3:
            logits = logits[0, position]
        elif logits.dim() == 2:
            logits = logits[position]
        # Now shape is (vocab_size,)

        probs: torch.Tensor = torch.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = probs.topk(min(k, probs.shape[-1]))

        results: List[Tuple[str, float]] = []
        for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist()):
            token_str: str = self._model.to_single_str_token(idx)
            results.append((token_str, float(prob)))
        return results

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize *text* and return an integer tensor on :attr:`device`.

        Parameters
        ----------
        text:
            Input text string.

        Returns
        -------
        1-D LongTensor of token IDs (no batch dimension).
        """
        tokens: torch.Tensor = self._model.to_tokens(text, prepend_bos=True)[0]
        return tokens.to(self.device)

    def to_string(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to a string."""
        return self._model.to_string(token_ids)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Properties / passthrough
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """The underlying HookedTransformer instance."""
        return self._model

    def __repr__(self) -> str:
        return (
            f"WakeModel(n_layers={self.n_layers}, d_model={self.d_model}, "
            f"n_heads={self.n_heads}, device={self.device!r})"
        )
