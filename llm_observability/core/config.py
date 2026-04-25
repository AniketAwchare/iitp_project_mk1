"""
config.py — Central configuration for the observability framework.
All thresholds, model names, and paths are set here.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────
    model_name: str          = "mistralai/Mistral-7B-Instruct-v0.2"
    model_quantize_4bit: bool = True          # 4-bit quant for Colab T4
    device: str              = "auto"
    max_new_tokens: int      = 512
    temperature: float       = 0.1            # Near-deterministic for consistency tests

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str     = "all-MiniLM-L6-v2"

    # ── RAG ───────────────────────────────────────────────────────────
    vector_store_path: str   = "./data/financial_corpus/faiss_index"
    chunk_size: int          = 512
    chunk_overlap: int       = 64
    top_k_retrieval: int     = 3

    # ── Module 1 — Consistency ────────────────────────────────────────
    num_paraphrases: int     = 5
    consistency_threshold: float = 0.75       # Below this → flagged

    # ── Module 2 — Drift Detection ────────────────────────────────────
    baseline_window_size: int = 200           # Queries to establish baseline
    drift_window_size: int    = 50            # Rolling window for comparison
    drift_p_threshold: float  = 0.05          # KS test significance level

    # ── Module 3 — Retrieval Alignment ───────────────────────────────
    alignment_threshold: float      = 0.50
    faithfulness_threshold: float   = 0.70

    # ── Module 4 — Anomaly Detection ─────────────────────────────────
    anomaly_contamination: float = 0.10       # Expected fraction of outliers
    anomaly_threshold: float     = 0.65       # Ensemble score → flag above this

    # ── Logging & Tracking ───────────────────────────────────────────
    log_path: str                  = "./logs"
    mlflow_tracking_uri: str       = "./mlruns"
    mlflow_experiment_name: str    = "llm-observability-finance"

    # ── Domain ────────────────────────────────────────────────────────
    domain: str = "financial_services"        # banking | insurance | trading
