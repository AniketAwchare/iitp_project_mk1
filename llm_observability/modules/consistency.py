"""
modules/consistency.py — Module 1: Response Consistency Scorer.

Measures how much a system's answers vary across semantically
equivalent (paraphrased) queries. Flags low-consistency responses.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Paraphrase templates for financial queries
_TEMPLATES = [
    "{q}",
    "Can you explain: {q}",
    "In financial terms, {q}",
    "Please answer the following: {q}",
    "I would like to know: {q}",
    "Could you clarify: {q}",
]


class ConsistencyScorer:
    """
    Generates N paraphrases of a query, queries the LLM for each,
    then measures embedding cosine similarity + ROUGE-L across responses.

    Composite score = 0.6 × mean_cosine_sim + 0.4 × mean_rouge_l
    """

    def __init__(self, config):
        self.config   = config
        self.embedder = SentenceTransformer(config.embedding_model)
        self._rouge   = None   # lazy-loaded

    def _rouge_scorer(self):
        if self._rouge is None:
            import evaluate
            self._rouge = evaluate.load("rouge")
        return self._rouge

    # ── Paraphrase generation ─────────────────────────────────────────

    def generate_paraphrases(self, query: str) -> List[str]:
        n = self.config.num_paraphrases
        templates = _TEMPLATES[:n]
        return [t.format(q=query) for t in templates]

    # ── Scoring ───────────────────────────────────────────────────────

    def score_responses(self, responses: List[str]) -> Dict[str, Any]:
        """Score a list of responses (already generated) for consistency."""
        if len(responses) < 2:
            return {"consistency_score": 1.0, "cosine_sim": 1.0,
                    "rouge_l": 1.0, "is_flagged": False}

        # Cosine similarity across all embeddings
        embs   = self.embedder.encode(responses, show_progress_bar=False)
        sim_mx = cosine_similarity(embs)
        n      = len(responses)
        pairs  = [sim_mx[i][j] for i in range(n) for j in range(i + 1, n)]
        mean_cos = float(np.mean(pairs))

        # ROUGE-L (each response vs. first response as reference)
        rouge  = self._rouge_scorer()
        r_scores = []
        ref = responses[0]
        for resp in responses[1:]:
            out = rouge.compute(predictions=[resp], references=[ref])
            r_scores.append(out["rougeL"])
        mean_rouge = float(np.mean(r_scores)) if r_scores else 0.0

        composite  = 0.6 * mean_cos + 0.4 * mean_rouge
        is_flagged = composite < self.config.consistency_threshold

        return {
            "consistency_score": round(composite,  4),
            "cosine_sim":        round(mean_cos,   4),
            "rouge_l":           round(mean_rouge, 4),
            "is_flagged":        is_flagged,
            "threshold":         self.config.consistency_threshold,
            "n_responses":       len(responses),
        }

    def evaluate_query(
        self, query: str, llm_fn: Callable[[str], str]
    ) -> Dict[str, Any]:
        """
        Full pipeline: paraphrase → call LLM → score.
        llm_fn: str → str (wraps pipeline.generate or similar)
        """
        paraphrases = self.generate_paraphrases(query)
        logger.info("Running %d paraphrases for consistency check…", len(paraphrases))
        responses   = [llm_fn(p) for p in paraphrases]
        result      = self.score_responses(responses)
        result.update({"query": query, "paraphrases": paraphrases,
                        "responses": responses})
        return result
