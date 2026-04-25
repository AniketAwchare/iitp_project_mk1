# LLM Observability Framework
### Observability & Reliability for Production LLM Systems — Financial Services
**IIT Patna · Executive M.Tech AI & Data Science 2025–26**  
*Aniket Awchare · Supervisor: Sachin Rathore*

---

## What This Is

A modular, open-source monitoring layer that runs **alongside** a deployed LLM application and detects reliability failures — without requiring ground-truth labels for every response.

Built for the **financial services domain** (banking, insurance, trading) using 100% free and open-source tools. Runs on Google Colab T4 / Kaggle P100.

---

## Architecture

```
User Query → FastAPI Endpoint (Mistral 7B 4-bit + FAISS RAG)
                    ↓ Logs every {query, context, response}
          ┌─────────────────────────────────────┐
          │     Observability Monitoring Layer   │
          │  ┌──────────────┐ ┌───────────────┐ │
          │  │ Consistency  │ │ Drift         │ │
          │  │ Scorer       │ │ Detector      │ │
          │  └──────────────┘ └───────────────┘ │
          │  ┌──────────────┐ ┌───────────────┐ │
          │  │ Retrieval    │ │ Anomaly       │ │
          │  │ Alignment    │ │ Detector      │ │
          │  └──────────────┘ └───────────────┘ │
          └──────────────┬──────────────────────┘
                         ↓ MLflow tracking
                  Streamlit Dashboard
```

---

## Project Structure

```
llm-observability-framework/
├── llm_observability/
│   ├── core/
│   │   ├── config.py        # All thresholds & paths (single source of truth)
│   │   ├── logger.py        # JSONL structured logger
│   │   └── pipeline.py      # FastAPI + LangChain RAG endpoint
│   ├── modules/
│   │   ├── consistency.py   # Module 1: Response Consistency Scorer
│   │   ├── drift.py         # Module 2: Semantic Drift Detector
│   │   ├── retrieval.py     # Module 3: Retrieval Alignment Scorer
│   │   └── anomaly.py       # Module 4: Prompt Anomaly Detector
│   ├── dashboard/
│   │   ├── interface.py     # Abstract UI contract (future React/Next.js hook)
│   │   └── app.py           # Streamlit dashboard
│   ├── experiments/
│   │   ├── domain_shift.py  # Experiment 1: domain shift detection
│   │   ├── adversarial.py   # Experiment 2: adversarial prompt detection
│   │   └── rag_failure.py   # Experiment 3: RAG alignment failure
│   └── data/
│       └── loaders.py       # FinanceBench + FiQA-2018 loaders
├── tracking/
│   └── mlflow_setup.py      # MLflow helpers
├── notebooks/               # Colab/Kaggle notebooks (one per module)
├── results/                 # Auto-generated experiment outputs
├── logs/                    # JSONL query logs
├── requirements.txt
├── setup.py
├── Dockerfile
└── run_experiments.py       # One-shot experiment runner (no GPU needed)
```

---

## Quick Start

### Option A — Google Colab / Kaggle (Recommended)

Open the notebooks in `notebooks/` — each is self-contained with `!pip install` cells.

### Option B — Local (CPU, no LLM needed for experiments)

```bash
# 1. Clone
git clone https://github.com/AniketAwchare/iitp_project_mk1.git
cd iitp_project_mk1

# 2. Install (CPU-only is fine for experiments 1–3)
pip install -r requirements.txt

# 3. Run all three experiments
python run_experiments.py

# 4. Launch dashboard
streamlit run llm_observability/dashboard/app.py
```

### Option C — Docker

```bash
docker build -t llm-observability .
docker run -p 8501:8501 -p 8000:8000 llm-observability
```

---

## The Four Modules

| Module | What it measures | Key technique |
|---|---|---|
| **Consistency Scorer** | Response variation across paraphrased queries | Embedding cosine sim + ROUGE-L |
| **Drift Detector** | Distribution shift in incoming query stream | KS test on PCA-reduced embeddings |
| **Retrieval Alignment** | RAG pipeline health (relevance + faithfulness) | Cosine sim + token overlap |
| **Anomaly Detector** | Unusual / adversarial financial prompts | Isolation Forest + LOF + rule-matching |

---

## Three Experiments

| Experiment | Failure mode | Metric |
|---|---|---|
| **Domain Shift** | Banking → Insurance/Trading query drift | Detection latency, Precision, Recall, F1 |
| **Adversarial** | Compliance bypass, jailbreak prompts | F1 @ 5/10/20% injection rates, FPR |
| **RAG Failure** | Irrelevant / stale context chunks | Alignment score vs. quality Pearson r |

**Targets:** >80% detection rate · <15% false positives · Pearson r > 0.6

---

## Tech Stack (100% Free & Open Source)

| Layer | Tool |
|---|---|
| LLM | Mistral 7B (4-bit, `transformers` + `bitsandbytes`) |
| Compute | Google Colab T4 / Kaggle P100 |
| RAG | LangChain + FAISS |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Datasets | FinanceBench, FiQA-2018 (HuggingFace) |
| Drift | scipy KS test |
| Anomaly | scikit-learn Isolation Forest + LOF |
| Tracking | MLflow (local) |
| Dashboard | Streamlit + Plotly |
| Container | Docker |

---

## Reproducing Results

```bash
# Individual experiments
python -m llm_observability.experiments.domain_shift
python -m llm_observability.experiments.adversarial
python -m llm_observability.experiments.rag_failure

# All at once
python run_experiments.py

# Results saved to ./results/
# MLflow UI: mlflow ui --port 5000
```

---

## Citation

```
Awchare, A. (2026). A Framework for Observability and Reliability in Production 
Large Language Model Systems. M.Tech Thesis, IIT Patna.
Supervisor: Sachin Rathore, Cloud Software Group.
GitHub: https://github.com/AniketAwchare/iitp_project_mk1
```
