"""
modules/retrieval.py — Module 3: Retrieval Alignment Scorer.

For RAG pipelines: measures whether the retrieved context is
relevant to the query and whether the response actually uses it.

Three sub-metrics:
  1. retrieval_relevance  — query ↔ retrieved chunks cosine similarity
  2. context_utilization  — token overlap between chunks and response
  3. faithfulness         — response embedding vs. context embedding sim

Composite = 0.4 × relevance + 0.3 × utilization + 0.3 × faithfulness
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RetrievalAlignmentScorer:

    def __init__(self, config):
        self.config   = config
        self.embedder = SentenceTransformer(config.embedding_model)

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, show_progress_bar=False)

    # ── Sub-metrics ───────────────────────────────────────────────────

    def retrieval_relevance(self, query: str, chunks: List[str]) -> float:
        """Mean cosine similarity between query embedding and each chunk."""
        if not chunks:
            return 0.0
        q_emb  = self._embed([query])
        c_embs = self._embed(chunks)
        sims   = cosine_similarity(q_emb, c_embs)[0]
        return float(np.mean(sims))

    def context_utilization(self, response: str, chunks: List[str]) -> float:
        """
        Fraction of unique content tokens (from chunks) that appear in
        the response. Proxy for how much retrieved context was used.
        """
        if not chunks:
            return 0.0
        resp_tokens  = set(response.lower().split())
        ctx_tokens   = set(" ".join(chunks).lower().split())
        # Remove stopwords (quick approximation)
        stopwords = {"the","a","an","is","in","of","to","and","or","for",
                     "on","at","by","with","from","this","that","it","be"}
        ctx_tokens  -= stopwords
        resp_tokens -= stopwords
        if not ctx_tokens:
            return 0.0
        return len(resp_tokens & ctx_tokens) / len(ctx_tokens)

    def faithfulness(self, response: str, chunks: List[str]) -> float:
        """
        Cosine similarity between response embedding and combined
        context embedding — does the response stay within context scope?
        """
        if not chunks:
            return 0.0
        ctx_text = " ".join(chunks)
        r_emb    = self._embed([response])
        c_emb    = self._embed([ctx_text])
        return float(cosine_similarity(r_emb, c_emb)[0][0])

    # ── Composite ─────────────────────────────────────────────────────

    def score(
        self, query: str, response: str, retrieved_chunks: List[str]
    ) -> Dict[str, Any]:
        """Compute all three sub-scores and composite alignment score."""
        rel  = self.retrieval_relevance(query,    retrieved_chunks)
        util = self.context_utilization(response, retrieved_chunks)
        fth  = self.faithfulness(response,        retrieved_chunks)

        composite  = 0.4 * rel + 0.3 * util + 0.3 * fth
        is_flagged = composite < self.config.alignment_threshold

        return {
            "alignment_score":      round(composite, 4),
            "retrieval_relevance":  round(rel,  4),
            "context_utilization":  round(util, 4),
            "faithfulness":         round(fth,  4),
            "is_flagged":           is_flagged,
            "threshold":            self.config.alignment_threshold,
            "n_chunks":             len(retrieved_chunks),
        }
