"""
Residual-stream geometry analysis for WAKE interpretability.

Analyses the geometric structure of residual-stream activations across
multiple lenses, revealing whether tokens occupy distinct clusters (one per
lens — collapsed) or occupy intermediate positions between lens centroids
(superposed).

Classes
-------
StreamGeometryResult
    Dataclass holding PCA coordinates, centroids, cluster statistics, and
    the list of token positions identified as superposed.

ResidualStreamAnalyzer
    Fits PCA (or UMAP) on multi-lens residuals and computes all geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# StreamGeometryResult
# ---------------------------------------------------------------------------

@dataclass
class StreamGeometryResult:
    """Result of a residual-stream geometry analysis.

    Attributes
    ----------
    passage:
        The raw Wake text passage that was analysed.
    layer:
        Transformer layer at which residuals were extracted.
    n_lenses:
        Number of lenses included in the analysis.
    pca_coords:
        Shape ``[n_lenses * seq_len, 2]``.  Each row is the 2-D PCA
        projection of one (lens, token) residual-stream vector.  The rows
        are ordered: all tokens for lens 0, then all tokens for lens 1, etc.
    pca_variance_explained:
        Tuple ``(var_pc1, var_pc2)`` — variance explained by the first two
        principal components.
    lens_centroids:
        Dict mapping each lens name to its 2-D centroid in PCA space.
    token_lens_assignments:
        Flat list of length ``n_lenses * seq_len`` recording which lens each
        point belongs to.  Useful for colour-coding scatter plots.
    superposition_tokens:
        Token indices (within the passage) where the token's per-lens
        residuals fall *between* lens centroids rather than clustering with
        any single lens.  These are candidate superposition sites.
    cluster_separation:
        Scalar measure of cluster quality:
        mean inter-centroid distance / mean intra-cluster spread.
        Values > 1 indicate well-separated lens clusters.
    """

    passage: str
    layer: int
    n_lenses: int
    pca_coords: np.ndarray
    pca_variance_explained: Tuple[float, float]
    lens_centroids: Dict[str, np.ndarray]
    token_lens_assignments: List[str]
    superposition_tokens: List[int]
    cluster_separation: float


# ---------------------------------------------------------------------------
# ResidualStreamAnalyzer
# ---------------------------------------------------------------------------

class ResidualStreamAnalyzer:
    """Analyse the geometry of residual-stream activations across lenses.

    Parameters
    ----------
    n_components:
        Number of PCA components to retain.  The first two are used for
        visualisation; additional components are available via the fitted
        :attr:`pca` instance.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.pca: PCA = PCA(n_components=n_components)

    # ------------------------------------------------------------------
    # Primary geometry computation
    # ------------------------------------------------------------------

    def compute_geometry(
        self,
        residuals_by_lens: Dict[str, np.ndarray],
        passage_tokens: List[str],
        layer: int,
        passage: str,
    ) -> StreamGeometryResult:
        """Fit PCA and compute full geometry for a multi-lens residual set.

        Parameters
        ----------
        residuals_by_lens:
            Dict mapping lens name → ``np.ndarray`` of shape
            ``[seq_len, d_model]``.  All lenses must have the same
            ``seq_len`` and ``d_model``.
        passage_tokens:
            List of token strings (length ``seq_len``).  Used to label
            superposition sites.
        layer:
            Layer index (stored in result for reference).
        passage:
            Raw passage text (stored in result for reference).

        Returns
        -------
        StreamGeometryResult
        """
        lens_names = list(residuals_by_lens.keys())
        n_lenses = len(lens_names)

        if n_lenses == 0:
            raise ValueError("residuals_by_lens must contain at least one lens.")

        # Validate shapes
        first = next(iter(residuals_by_lens.values()))
        seq_len, d_model = first.shape
        for name, arr in residuals_by_lens.items():
            if arr.shape != (seq_len, d_model):
                raise ValueError(
                    f"Lens '{name}' has shape {arr.shape}, "
                    f"expected ({seq_len}, {d_model})."
                )

        # ---- Stack all activations [n_lenses * seq_len, d_model] --------
        stacked = np.vstack([residuals_by_lens[ln] for ln in lens_names])
        assignments = [ln for ln in lens_names for _ in range(seq_len)]

        # ---- Fit PCA -------------------------------------------------------
        n_fit_components = min(self.n_components, stacked.shape[0], stacked.shape[1])
        self.pca = PCA(n_components=n_fit_components)
        coords_full = self.pca.fit_transform(stacked)  # [N, n_components]

        # Ensure we always have exactly 2 columns for downstream consumers
        if coords_full.shape[1] < 2:
            pad = np.zeros((coords_full.shape[0], 2 - coords_full.shape[1]))
            coords_full = np.hstack([coords_full, pad])
        pca_coords = coords_full[:, :2]

        var_explained = self.pca.explained_variance_ratio_
        var1 = float(var_explained[0]) if len(var_explained) > 0 else 0.0
        var2 = float(var_explained[1]) if len(var_explained) > 1 else 0.0

        # ---- Per-lens centroids in PCA space ----------------------------
        lens_centroids: Dict[str, np.ndarray] = {}
        for i, ln in enumerate(lens_names):
            start = i * seq_len
            end = start + seq_len
            lens_centroids[ln] = pca_coords[start:end].mean(axis=0)

        # ---- Cluster separation -----------------------------------------
        cluster_separation = self._compute_cluster_separation(
            pca_coords, lens_names, seq_len
        )

        # ---- Superposition detection ------------------------------------
        superposition_tokens = self._detect_superposition_tokens(
            pca_coords, lens_names, seq_len, lens_centroids
        )

        return StreamGeometryResult(
            passage=passage,
            layer=layer,
            n_lenses=n_lenses,
            pca_coords=pca_coords,
            pca_variance_explained=(var1, var2),
            lens_centroids=lens_centroids,
            token_lens_assignments=assignments,
            superposition_tokens=superposition_tokens,
            cluster_separation=cluster_separation,
        )

    # ------------------------------------------------------------------
    # Superposition test for a single point
    # ------------------------------------------------------------------

    def is_superposed(
        self,
        point: np.ndarray,
        centroids: Dict[str, np.ndarray],
        threshold: float = 0.3,
    ) -> bool:
        """Test whether *point* lies between multiple lens centroids.

        A point is considered "superposed" if it is within *threshold* of the
        *normalised* distance to multiple centroids — i.e., it is not clearly
        dominated by any single centroid.

        The threshold operates on the normalised distance:
        ``d_i / sum(d_j)`` — if the smallest such normalised distance is
        greater than *threshold*, the point is equidistant from multiple
        centroids.

        Parameters
        ----------
        point:
            2-D PCA coordinate, shape ``[2]``.
        centroids:
            Dict mapping lens name → 2-D centroid.
        threshold:
            Normalised-distance threshold.  Default 0.3.

        Returns
        -------
        bool
        """
        if not centroids:
            return False

        dists = {
            name: float(np.linalg.norm(point - c))
            for name, c in centroids.items()
        }
        total = sum(dists.values()) + 1e-12
        normed = {name: d / total for name, d in dists.items()}

        # Sort normalised distances ascending
        sorted_dists = sorted(normed.values())
        if len(sorted_dists) < 2:
            return False

        # Superposed if the minimum distance exceeds the threshold
        # (meaning no single centroid dominates)
        return sorted_dists[0] > threshold

    # ------------------------------------------------------------------
    # UMAP projection
    # ------------------------------------------------------------------

    def compute_umap(
        self,
        residuals_by_lens: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute a 2-D UMAP projection of multi-lens residuals.

        Falls back to PCA if the ``umap-learn`` package is not installed.

        Parameters
        ----------
        residuals_by_lens:
            Dict mapping lens name → ``np.ndarray`` of shape
            ``[seq_len, d_model]``.

        Returns
        -------
        np.ndarray
            Shape ``[n_lenses * seq_len, 2]``.
        """
        stacked = np.vstack(list(residuals_by_lens.values()))

        try:
            import umap  # type: ignore[import-untyped]

            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(stacked)

        except ImportError:
            # umap-learn not available — fall back to PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(stacked)
            if coords.shape[1] < 2:
                pad = np.zeros((coords.shape[0], 2 - coords.shape[1]))
                coords = np.hstack([coords, pad])
            return coords[:, :2]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_cluster_separation(
        self,
        pca_coords: np.ndarray,
        lens_names: List[str],
        seq_len: int,
    ) -> float:
        """Compute mean inter-centroid distance / mean intra-cluster spread.

        Returns
        -------
        float
            Separation score.  Higher = more separated clusters.
        """
        n_lenses = len(lens_names)
        if n_lenses < 2:
            return 0.0

        centroids: List[np.ndarray] = []
        spreads: List[float] = []

        for i in range(n_lenses):
            start = i * seq_len
            end = start + seq_len
            cluster = pca_coords[start:end]
            c = cluster.mean(axis=0)
            centroids.append(c)
            spread = float(np.mean(np.linalg.norm(cluster - c, axis=1)))
            spreads.append(spread)

        # Mean pairwise inter-centroid distance
        inter_dists: List[float] = []
        for i in range(n_lenses):
            for j in range(i + 1, n_lenses):
                inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))

        mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
        mean_intra = float(np.mean(spreads)) if spreads else 1.0

        return mean_inter / (mean_intra + 1e-12)

    def _detect_superposition_tokens(
        self,
        pca_coords: np.ndarray,
        lens_names: List[str],
        seq_len: int,
        lens_centroids: Dict[str, np.ndarray],
    ) -> List[int]:
        """Identify token positions where per-lens points cluster between
        lens centroids (indicative of superposition).

        For each token position *t*, collect the *n_lenses* PCA points
        (one per lens) and test whether the centroid of those *n_lenses*
        points falls between the global lens centroids rather than near any
        single one.

        Returns
        -------
        List[int]
            Token indices (0-indexed within the passage) where superposition
            is detected.
        """
        n_lenses = len(lens_names)
        superposed: List[int] = []

        for t in range(seq_len):
            # Collect the n_lenses points for token t
            token_points = np.array(
                [pca_coords[i * seq_len + t] for i in range(n_lenses)]
            )  # [n_lenses, 2]
            token_centroid = token_points.mean(axis=0)

            # Is this centroid "between" the lens centroids?
            if self.is_superposed(token_centroid, lens_centroids, threshold=0.3):
                superposed.append(t)

        return superposed
