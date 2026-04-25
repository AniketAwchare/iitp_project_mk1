"""
experiments/adversarial.py — Experiment 2: Adversarial Prompt Detection.

Injects known adversarial financial prompts into a benign query stream
at three injection rates (5%, 10%, 20%) and measures the AnomalyDetector's
precision, recall, and F1.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from llm_observability.core.config import Config
from llm_observability.modules.anomaly import AnomalyDetector
from llm_observability.data.loaders import (
    ADVERSARIAL_FINANCIAL_PROMPTS,
    BENIGN_FINANCIAL_PROMPTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results/adversarial")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(
    injection_rate: float = 0.10,
    stream_length: int = 300,
    config: Config = None,
) -> Dict:
    """
    Args:
        injection_rate : fraction of adversarial queries in the stream
        stream_length  : total queries to evaluate
        config         : Config instance
    """
    cfg      = config or Config()
    detector = AnomalyDetector(cfg)

    # Build baseline from benign prompts (repeated to fill window)
    benign_pool     = BENIGN_FINANCIAL_PROMPTS * 20
    adversarial_pool = ADVERSARIAL_FINANCIAL_PROMPTS * 20

    # Fit on benign baseline
    baseline = list(np.random.choice(benign_pool,
                                     size=cfg.baseline_window_size,
                                     replace=True))
    detector.fit(baseline)

    # Build mixed evaluation stream
    n_adv   = int(stream_length * injection_rate)
    n_ben   = stream_length - n_adv
    stream  = (list(np.random.choice(benign_pool,     size=n_ben, replace=True)) +
               list(np.random.choice(adversarial_pool, size=n_adv, replace=True)))
    labels  = [0] * n_ben + [1] * n_adv

    # Shuffle together
    combined = list(zip(stream, labels))
    np.random.shuffle(combined)
    stream, labels = zip(*combined)

    records = []
    for query, label in zip(stream, labels):
        result = detector.detect(query)
        result["true_label"]      = label
        result["predicted_label"] = 1 if result["is_anomaly"] else 0
        records.append(result)

    df = pd.DataFrame(records)

    tp = int(((df.predicted_label == 1) & (df.true_label == 1)).sum())
    fp = int(((df.predicted_label == 1) & (df.true_label == 0)).sum())
    fn = int(((df.predicted_label == 0) & (df.true_label == 1)).sum())
    tn = int(((df.predicted_label == 0) & (df.true_label == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    summary = {
        "injection_rate": injection_rate,
        "stream_length":  stream_length,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "fpr":        round(fpr,       4),
    }

    df.to_csv(RESULTS_DIR / f"stream_rate{int(injection_rate*100)}.csv", index=False)
    with open(RESULTS_DIR / f"summary_rate{int(injection_rate*100)}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Adversarial experiment done: %s", summary)
    return summary


def run_all(config: Config = None) -> List[Dict]:
    """Run across 5%, 10%, 20% injection rates."""
    results = []
    for rate in [0.05, 0.10, 0.20]:
        logger.info("=" * 60)
        logger.info("Adversarial injection_rate=%.2f", rate)
        results.append(run_experiment(injection_rate=rate, config=config))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    print("\n=== Adversarial Detection Results ===")
    print(summary_df.to_string(index=False))
    return results


if __name__ == "__main__":
    run_all()
