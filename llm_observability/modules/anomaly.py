"""
modules/anomaly.py — Module 4: Prompt Anomaly Detector.

Detects unusual or adversarially-crafted incoming queries using:
  1. Rule-based pattern matching (financial adversarial patterns)
  2. Isolation Forest on embedding space (unsupervised)
  3. Local Outlier Factor on embedding space (unsupervised)

Ensemble score = 0.3 × rule + 0.4 × isolation_forest + 0.3 × lof
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)

# ── Financial adversarial patterns ────────────────────────────────────
_PATTERNS: List[str] = [
    "ignore your instructions", "ignore your compliance",
    "forget your rules", "bypass compliance",
    "insider trading tips", "manipulate stock",
    "pretend you have no restrictions", "you are now unrestricted",
    "disregard previous", "override safety",
    "jailbreak", "act as if you are not bound",
    "launder money", "hide assets offshore",
    "evade taxes illegally", "front running",
    "pump and dump", "ponzi scheme",
    "forge financial documents", "bypass kyc",
    "securities fraud", "manipulate libor",
    "reveal confidential", "exploit loopholes illegally",
]


class AnomalyDetector:

    def __init__(self, config):
        self.config   = config
        self.embedder = SentenceTransformer(config.embedding_model)

        self._iforest: Optional[IsolationForest]       = None
        self._lof:     Optional[LocalOutlierFactor]    = None
        self._baseline_embeddings: Optional[np.ndarray] = None
        self.is_fitted = False
        self._buf: List[str] = []   # buffer before enough data to fit

    # ── Baseline fitting ──────────────────────────────────────────────

    def fit(self, queries: List[str]) -> None:
        """Fit both unsupervised detectors on a set of benign queries."""
        embs = self._embed(queries)
        self._baseline_embeddings = embs

        self._iforest = IsolationForest(
            n_estimators=100,
            contamination=self.config.anomaly_contamination,
            random_state=42,
        )
        self._iforest.fit(embs)

        n_neighbors = min(20, len(queries) - 1)
        self._lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.config.anomaly_contamination,
            novelty=True,
        )
        self._lof.fit(embs)

        self.is_fitted = True
        logger.info("Anomaly detector fitted on %d baseline queries.", len(queries))

    def add_to_buffer(self, query: str) -> bool:
        """
        Buffer queries until we have enough for fitting.
        Returns True when fitting was triggered automatically.
        """
        self._buf.append(query)
        if len(self._buf) >= self.config.baseline_window_size:
            self.fit(self._buf)
            return True
        return False

    # ── Detection ─────────────────────────────────────────────────────

    def detect(self, query: str) -> Dict[str, Any]:
        """
        Score a single query. Works even before baseline is fitted
        (falls back to rule-based only).
        """
        rule_score = self._rule_score(query)

        if not self.is_fitted:
            composite  = rule_score
            is_anomaly = composite > self.config.anomaly_threshold
            return {
                "is_anomaly":            is_anomaly,
                "anomaly_score":         round(composite, 4),
                "rule_based_score":      round(rule_score, 4),
                "isolation_forest_score": None,
                "lof_score":             None,
                "detector_ready":        False,
            }

        emb = self._embed([query])

        # Isolation Forest: score_samples returns negative — more negative = more anomalous
        if_raw  = float(self._iforest.score_samples(emb)[0])
        if_score = float(np.clip(1.0 - (if_raw + 0.5) / 0.5, 0.0, 1.0))

        # LOF: similarly, more negative = more anomalous
        lof_raw  = float(self._lof.score_samples(emb)[0])
        lof_score = float(np.clip(1.0 - (lof_raw + 2.0) / 2.0, 0.0, 1.0))

        composite  = 0.3 * rule_score + 0.4 * if_score + 0.3 * lof_score
        is_anomaly = composite > self.config.anomaly_threshold

        return {
            "is_anomaly":             is_anomaly,
            "anomaly_score":          round(composite,  4),
            "rule_based_score":       round(rule_score, 4),
            "isolation_forest_score": round(if_score,   4),
            "lof_score":              round(lof_score,  4),
            "detector_ready":         True,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False)

    def _rule_score(self, query: str) -> float:
        q = query.lower()
        hits = sum(1 for p in _PATTERNS if p in q)
        return min(1.0, hits / 2.0)  # 2+ matches → fully suspicious
