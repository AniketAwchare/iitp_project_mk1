"""
run_experiments.py — One-shot runner for all three experiments.

Does NOT require a GPU or the full LLM to be loaded.
Modules 1–4 use SentenceTransformers (CPU-compatible).
Experiments use synthetic / dataset-based queries.

Usage:
    python run_experiments.py
    python run_experiments.py --experiment domain_shift
    python run_experiments.py --experiment adversarial
    python run_experiments.py --experiment rag_failure
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiments")


def run_domain_shift():
    from llm_observability.experiments.domain_shift import run_all
    logger.info("▶  Experiment 1: Domain Shift Detection")
    results = run_all()
    return results


def run_adversarial():
    from llm_observability.experiments.adversarial import run_all
    logger.info("▶  Experiment 2: Adversarial Prompt Detection")
    results = run_all()
    return results


def run_rag_failure():
    from llm_observability.experiments.rag_failure import run_experiment
    logger.info("▶  Experiment 3: RAG Alignment Failure Detection")
    results = run_experiment()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM observability experiments")
    parser.add_argument(
        "--experiment",
        choices=["domain_shift", "adversarial", "rag_failure", "all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    args = parser.parse_args()

    start = time.time()
    print("\n" + "=" * 65)
    print("  LLM Observability Framework — Experiment Runner")
    print("  Financial Services Domain · IIT Patna M.Tech 2025-26")
    print("=" * 65 + "\n")

    if args.experiment in ("domain_shift", "all"):
        run_domain_shift()
        print()

    if args.experiment in ("adversarial", "all"):
        run_adversarial()
        print()

    if args.experiment in ("rag_failure", "all"):
        run_rag_failure()
        print()

    elapsed = time.time() - start
    print("\n" + "=" * 65)
    print(f"  All experiments complete in {elapsed:.1f}s")
    print("  Results saved to ./results/")
    print("  Launch dashboard: streamlit run llm_observability/dashboard/app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
