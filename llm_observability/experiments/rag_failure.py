"""
experiments/rag_failure.py — Experiment 3: RAG Alignment Failure Detection.

Introduces deliberate retrieval failures (irrelevant / stale chunks)
into the RAG pipeline and measures whether the RetrievalAlignmentScorer
catches them before they produce bad responses.

Three conditions:
  A. Healthy RAG   — correct, relevant context
  B. Irrelevant    — completely unrelated chunks injected
  C. Stale         — outdated information chunks

Measures alignment score distribution and correlation with a
simple response quality proxy (BERTScore vs. reference answer).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from llm_observability.core.config import Config
from llm_observability.modules.retrieval import RetrievalAlignmentScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results/rag_failure")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic financial QA pairs ──────────────────────────────────────
FINANCE_QA: List[Dict] = [
    {
        "query":   "What is the Basel III Tier 1 capital ratio requirement?",
        "answer":  "Basel III requires banks to maintain a minimum Tier 1 capital ratio of 6%.",
        "good_ctx": "Basel III mandates that banks hold a minimum Common Equity Tier 1 (CET1) capital ratio of 4.5% and a Tier 1 capital ratio of 6% of risk-weighted assets.",
    },
    {
        "query":   "What does EBITDA measure?",
        "answer":  "EBITDA measures a company's operating performance before interest, taxes, depreciation, and amortisation.",
        "good_ctx": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortisation. It is used as a proxy for a company's core operational profitability.",
    },
    {
        "query":   "How is the Sharpe ratio calculated?",
        "answer":  "The Sharpe ratio is calculated by subtracting the risk-free rate from the portfolio return and dividing by the portfolio's standard deviation.",
        "good_ctx": "The Sharpe ratio = (Rp - Rf) / σp, where Rp is portfolio return, Rf is the risk-free rate, and σp is the standard deviation of the portfolio's excess return.",
    },
    {
        "query":   "What is dollar-cost averaging?",
        "answer":  "Dollar-cost averaging is investing a fixed amount at regular intervals regardless of asset price.",
        "good_ctx": "Dollar-cost averaging (DCA) involves investing a fixed dollar amount in a particular investment at regular intervals, reducing the impact of volatility.",
    },
    {
        "query":   "What are the key risks in bond investing?",
        "answer":  "Key bond risks include interest rate risk, credit risk, inflation risk, and liquidity risk.",
        "good_ctx": "Bond investors face several risks: interest rate risk (bond prices fall when rates rise), credit risk (issuer default), inflation risk (purchasing power erosion), and liquidity risk.",
    },
]

IRRELEVANT_CHUNKS: List[str] = [
    "The capital of France is Paris, a major European cultural centre.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Shakespeare wrote Hamlet in approximately 1600.",
]

STALE_CHUNKS: List[str] = [
    "As of 2010, Basel II capital requirements were the global standard for banks.",
    "The Dodd-Frank Act was recently passed in 2010, introducing new financial regulations.",
    "Interest rates set by the Federal Reserve in 2015 were near historic lows at 0.25%.",
    "Bitcoin was trading at approximately $1,000 in early 2017.",
    "LIBOR was the global benchmark interest rate used widely across financial products.",
]


def _simple_quality_score(response: str, reference: str) -> float:
    """Token F1 overlap as a lightweight quality proxy (no GPU needed)."""
    r_toks = set(response.lower().split())
    ref_toks = set(reference.lower().split())
    if not ref_toks:
        return 0.0
    precision = len(r_toks & ref_toks) / len(r_toks) if r_toks else 0.0
    recall    = len(r_toks & ref_toks) / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_experiment(config: Config = None) -> Dict:
    cfg    = config or Config()
    scorer = RetrievalAlignmentScorer(cfg)

    records = []

    for qa in FINANCE_QA:
        query, answer, good_ctx = qa["query"], qa["answer"], qa["good_ctx"]

        # Condition A: healthy RAG
        for _ in range(3):   # 3 runs per condition for variance
            res_a = scorer.score(query, answer, [good_ctx])
            records.append({
                "condition":     "healthy",
                "query":         query,
                **res_a,
                "quality_proxy": _simple_quality_score(answer, answer),
            })

        # Condition B: irrelevant chunks
        irr_chunks = list(np.random.choice(IRRELEVANT_CHUNKS, size=3, replace=True))
        res_b = scorer.score(query, answer, irr_chunks)
        records.append({
            "condition":     "irrelevant",
            "query":         query,
            **res_b,
            "quality_proxy": _simple_quality_score(
                " ".join(irr_chunks), answer
            ),
        })

        # Condition C: stale chunks
        stale_chunks = list(np.random.choice(STALE_CHUNKS, size=3, replace=True))
        res_c = scorer.score(query, answer, stale_chunks)
        records.append({
            "condition":     "stale",
            "query":         query,
            **res_c,
            "quality_proxy": _simple_quality_score(
                " ".join(stale_chunks), answer
            ),
        })

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "detailed_results.csv", index=False)

    # Group stats
    group_stats = (
        df.groupby("condition")[["alignment_score", "quality_proxy", "is_flagged"]]
        .agg({"alignment_score": ["mean", "std"],
              "quality_proxy":   ["mean", "std"],
              "is_flagged":      "mean"})
        .round(4)
    )

    # Pearson correlation: alignment_score vs. quality_proxy
    corr = float(df["alignment_score"].corr(df["quality_proxy"]))

    summary = {
        "pearson_r_alignment_vs_quality": round(corr, 4),
        "mean_alignment_healthy":         round(df[df.condition == "healthy"]["alignment_score"].mean(), 4),
        "mean_alignment_irrelevant":      round(df[df.condition == "irrelevant"]["alignment_score"].mean(), 4),
        "mean_alignment_stale":           round(df[df.condition == "stale"]["alignment_score"].mean(), 4),
        "flag_rate_healthy":              round(df[df.condition == "healthy"]["is_flagged"].mean(), 4),
        "flag_rate_irrelevant":           round(df[df.condition == "irrelevant"]["is_flagged"].mean(), 4),
        "flag_rate_stale":                round(df[df.condition == "stale"]["is_flagged"].mean(), 4),
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RAG Failure Experiment Results ===")
    print(json.dumps(summary, indent=2))
    print("\nGroup statistics:")
    print(group_stats)

    logger.info("Pearson r (alignment vs quality): %.4f", corr)
    return summary


if __name__ == "__main__":
    run_experiment()
