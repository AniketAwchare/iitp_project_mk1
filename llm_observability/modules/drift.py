"""
modules/drift.py — Module 2: Semantic Drift Detector.

Embeds the incoming query stream, maintains a rolling reference
distribution, and applies KS tests (per PCA dimension) to detect
when the query distribution shifts significantly.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Two-phase detector:
      Phase 1 — baseline_window_size queries → build reference distribution
      Phase 2 — each subsequent drift_window_size queries → KS test vs. baseline
    """

    def __init__(self, config):
        self.config   = config
        self.embedder = SentenceTransformer(config.embedding_model)

        self._baseline_buf: List[np.ndarray] = []
        self._baseline:     Optional[np.ndarray] = None
        self._baseline_pca: Optional[np.ndarray] = None
        self._current:      deque = deque(maxlen=config.drift_window_size)

        self.pca          = PCA(n_components=50, random_state=42)
        self.baseline_ok  = False
        self.query_count  = 0

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False)

    # ── Public API ────────────────────────────────────────────────────

    def update(self, query: str) -> Dict[str, Any]:
        """
        Feed one query into the detector.
        Returns a status dict; drift_detected is None until baseline is ready.
        """
        emb = self._embed([query])[0]
        self.query_count += 1

        if not self.baseline_ok:
            self._baseline_buf.append(emb)
            if len(self._baseline_buf) >= self.config.baseline_window_size:
                self._fit_baseline()
            return {
                "status": "building_baseline",
                "query_count": self.query_count,
                "baseline_needed": self.config.baseline_window_size,
                "drift_detected": None,
            }

        self._current.append(emb)
        if len(self._current) < self.config.drift_window_size:
            return {
                "status": "building_window",
                "query_count": self.query_count,
                "window_size": len(self._current),
                "drift_detected": None,
            }

        return self._run_ks_test()

    def reset_baseline(self) -> None:
        """Force a new baseline (e.g., after a known distribution change)."""
        self._baseline_buf.clear()
        self._baseline      = None
        self._baseline_pca  = None
        self.baseline_ok    = False
        self._current.clear()
        logger.info("Drift detector baseline reset.")

    def get_visualization_data(self) -> Dict[str, Any]:
        """PCA-2D coordinates for scatter plot in the dashboard."""
        if not self.baseline_ok:
            return {}
        result = {"baseline_2d": self._baseline_pca[:, :2].tolist()}
        if self._current:
            cur_pca = self.pca.transform(np.array(list(self._current)))
            result["current_2d"] = cur_pca[:, :2].tolist()
        return result

    # ── Internal ──────────────────────────────────────────────────────

    def _fit_baseline(self) -> None:
        self._baseline = np.array(self._baseline_buf)
        n_comp = min(50, self._baseline.shape[0] - 1, self._baseline.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        self._baseline_pca = self.pca.fit_transform(self._baseline)
        self.baseline_ok   = True
        logger.info("Drift baseline established (%d queries, %d PCA dims).",
                    len(self._baseline_buf), n_comp)

    def _run_ks_test(self) -> Dict[str, Any]:
        current     = np.array(list(self._current))
        current_pca = self.pca.transform(current)

        n_dims   = min(10, self._baseline_pca.shape[1])
        p_values, ks_stats = [], []

        for dim in range(n_dims):
            stat, pval = stats.ks_2samp(
                self._baseline_pca[:, dim],
                current_pca[:, dim],
            )
            p_values.append(pval)
            ks_stats.append(stat)

        # Bonferroni correction
        corrected_alpha = self.config.drift_p_threshold / n_dims
        n_flagged = sum(p < corrected_alpha for p in p_values)
        drift_detected  = n_flagged > (n_dims * 0.3)   # >30% dims shifted
        drift_magnitude = float(np.mean(ks_stats))

        return {
            "status":          "monitoring",
            "drift_detected":  drift_detected,
            "drift_magnitude": round(drift_magnitude, 4),
            "dims_flagged":    n_flagged,
            "dims_tested":     n_dims,
            "min_pvalue":      round(float(min(p_values)), 6),
            "query_count":     self.query_count,
        }
