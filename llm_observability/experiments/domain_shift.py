"""
experiments/domain_shift.py — Experiment 1: Domain Shift Detection.

Simulates a gradual shift in query distribution from banking → insurance/trading.
Measures how quickly the DriftDetector flags the shift.

Metrics logged:
  - detection_query_index  : first query index where drift was flagged
  - detection_latency_pct  : fraction of injection phase before detection
  - drift_magnitude        : mean KS statistic at detection point
  - precision / recall     : over the full stream
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from llm_observability.core.config import Config
from llm_observability.modules.drift import DriftDetector
from llm_observability.data.loaders import get_domain_shift_queries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results/domain_shift")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(
    injection_rate: float = 0.5,
    stream_length: int = 400,
    config: Config = None,
) -> Dict:
    """
    Args:
        injection_rate: fraction of OOD queries injected after baseline (0.1 → 0.9)
        stream_length:  total number of queries in the stream
        config:         Config instance (uses defaults if None)
    """
    cfg = config or Config()
    detector = DriftDetector(cfg)
    queries  = get_domain_shift_queries()

    banking_pool   = queries["banking"]   * 100   # repeat to fill stream
    ood_pool       = (queries["insurance"] + queries["trading"]) * 50

    # Phase 1: pure baseline queries (banking)
    baseline_phase = list(np.random.choice(banking_pool,
                                           size=cfg.baseline_window_size,
                                           replace=True))
    # Phase 2: mixed stream with injection_rate fraction OOD
    n_total = stream_length - cfg.baseline_window_size
    n_ood   = int(n_total * injection_rate)
    n_ind   = n_total - n_ood
    mixed   = (list(np.random.choice(banking_pool, size=n_ind, replace=True)) +
               list(np.random.choice(ood_pool,     size=n_ood, replace=True)))
    np.random.shuffle(mixed)

    full_stream = baseline_phase + mixed
    # Ground-truth labels: 0 = in-domain, 1 = OOD
    labels = ([0] * len(baseline_phase) +
              [0 if q in banking_pool else 1 for q in mixed])

    results_per_query = []
    detected_at = None

    for i, query in enumerate(full_stream):
        result = detector.update(query)
        result["true_label"] = labels[i]
        result["query_index"] = i
        results_per_query.append(result)

        if result.get("drift_detected") and detected_at is None:
            detected_at = i
            logger.info("Drift detected at query %d (injection_rate=%.1f)", i, injection_rate)

    # Compute precision / recall
    detection_window_start = cfg.baseline_window_size + cfg.drift_window_size
    predicted_drift = [
        1 if r.get("drift_detected") else 0
        for r in results_per_query[detection_window_start:]
    ]
    true_drift = labels[detection_window_start:]

    tp = sum(p == 1 and t == 1 for p, t in zip(predicted_drift, true_drift))
    fp = sum(p == 1 and t == 0 for p, t in zip(predicted_drift, true_drift))
    fn = sum(p == 0 and t == 1 for p, t in zip(predicted_drift, true_drift))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    detection_latency_pct = (
        (detected_at - cfg.baseline_window_size) / n_ood
        if detected_at and n_ood > 0 else 1.0
    )

    summary = {
        "injection_rate":        injection_rate,
        "stream_length":         stream_length,
        "detected_at_query":     detected_at,
        "detection_latency_pct": round(detection_latency_pct, 4),
        "precision":             round(precision, 4),
        "recall":                round(recall,    4),
        "f1":                    round(f1,        4),
    }

    # Save detailed results
    df = pd.DataFrame(results_per_query)
    df.to_csv(RESULTS_DIR / f"stream_rate{int(injection_rate*100)}.csv", index=False)

    with open(RESULTS_DIR / f"summary_rate{int(injection_rate*100)}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Experiment done: %s", summary)
    return summary


def run_all(config: Config = None) -> List[Dict]:
    """Run domain shift experiment across 3 injection rates."""
    results = []
    for rate in [0.1, 0.5, 0.9]:
        logger.info("=" * 60)
        logger.info("Running domain shift: injection_rate=%.1f", rate)
        results.append(run_experiment(injection_rate=rate, config=config))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
    print("\n=== Domain Shift Results ===")
    print(summary_df.to_string(index=False))
    return results


if __name__ == "__main__":
    run_all()
