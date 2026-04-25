# %% [markdown]
# # Notebook 02 — All 4 Modules Demo
# **LLM Observability Framework · Financial Services**
#
# **Prerequisite:** Run `colab_01_infrastructure.py` first (model must be loaded).
#
# This notebook demos all four observability modules against live LLM responses:
# - Module 1: Response Consistency Scorer
# - Module 2: Semantic Drift Detector
# - Module 3: Retrieval Alignment Scorer
# - Module 4: Prompt Anomaly Detector

# %% [markdown]
# ## Setup — Restore pipeline from session

# %%
import sys, os
REPO_PATH = "/content/llm-observability-framework"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)

from llm_observability.core.config   import Config
from llm_observability.core.pipeline import LLMPipeline

config   = Config()
pipeline = LLMPipeline(config)
pipeline.load_rag()      # reload FAISS index (already built)
pipeline.load_model()    # reload model (or skip if session still has it)

llm_fn = lambda q: pipeline.generate(q, context=pipeline.retrieve(q))
print("✅ Pipeline ready.")

# %% [markdown]
# ## Module 1 — Response Consistency Scorer

# %%
from llm_observability.modules.consistency import ConsistencyScorer

scorer = ConsistencyScorer(config)

# Test with a financial query
seed_query = "What is the minimum Tier 1 capital ratio required under Basel III?"

print(f"Query: {seed_query}")
print(f"Generating {config.num_paraphrases} paraphrases and querying LLM...\n")

result = scorer.evaluate_query(seed_query, llm_fn)

print("=" * 55)
print(f"  Consistency Score  : {result['consistency_score']:.4f}")
print(f"  Cosine Similarity  : {result['cosine_sim']:.4f}")
print(f"  ROUGE-L            : {result['rouge_l']:.4f}")
print(f"  Flagged            : {'⚠️  YES' if result['is_flagged'] else '✅ No'}")
print(f"  Threshold          : {result['threshold']}")
print("=" * 55)
print("\nParaphrases used:")
for i, p in enumerate(result["paraphrases"]):
    print(f"  {i+1}. {p}")

# %% [markdown]
# ### Module 1 — Batch evaluation over FinanceBench QA pairs

# %%
import pandas as pd
from llm_observability.data.loaders import load_financebench

_, qa_pairs = load_financebench(max_samples=20)  # small batch for demo

consistency_results = []
for qa in qa_pairs[:10]:
    q = qa["question"]
    r = scorer.evaluate_query(q, llm_fn)
    consistency_results.append({
        "question":          q[:60] + "...",
        "consistency_score": r["consistency_score"],
        "flagged":           r["is_flagged"],
    })
    print(f"  [{r['consistency_score']:.3f}] {'⚠️' if r['is_flagged'] else '✅'} {q[:70]}")

df_c = pd.DataFrame(consistency_results)
print(f"\n--- Consistency Summary ---")
print(f"Mean score : {df_c['consistency_score'].mean():.4f}")
print(f"Std dev    : {df_c['consistency_score'].std():.4f}")
print(f"Flagged    : {df_c['flagged'].sum()} / {len(df_c)}")

# %% [markdown]
# ## Module 2 — Semantic Drift Detector

# %%
from llm_observability.modules.drift import DriftDetector
from llm_observability.data.loaders import get_domain_shift_queries
import numpy as np

detector = DriftDetector(config)
queries  = get_domain_shift_queries()

print("Phase 1: Building baseline with banking queries...")
for _ in range(config.baseline_window_size):
    q = np.random.choice(queries["banking"] * 50)
    detector.update(q)
print(f"  Baseline ready: {config.baseline_window_size} queries\n")

print("Phase 2: Injecting insurance/trading queries (OOD)...")
results_drift = []
n_ood = 80

for i in range(n_ood):
    # Mix: first 20 benign banking, then 60 OOD
    if i < 20:
        q = np.random.choice(queries["banking"] * 50)
        label = "banking"
    else:
        pool = queries["insurance"] + queries["trading"]
        q = np.random.choice(pool * 50)
        label = "OOD"

    status = detector.update(q)
    if status.get("drift_detected") is not None:
        results_drift.append({
            "step":            i,
            "label":           label,
            "drift_detected":  status["drift_detected"],
            "drift_magnitude": status.get("drift_magnitude", 0),
        })
        if i % 10 == 0:
            flag = "🔴 DRIFT" if status["drift_detected"] else "✅ stable"
            print(f"  Step {i:3d} | {label:10s} | {flag} | magnitude={status.get('drift_magnitude', 0):.4f}")

if results_drift:
    df_d = pd.DataFrame(results_drift)
    first_detection = df_d[df_d["drift_detected"]]["step"].min()
    print(f"\n🔴 Drift first detected at step: {first_detection}")
    print(f"   (After {first_detection - 20} OOD queries)")
else:
    print("\n  Drift detection window not yet full — need more queries.")

# %% [markdown]
# ## Module 3 — Retrieval Alignment Scorer

# %%
from llm_observability.modules.retrieval import RetrievalAlignmentScorer

aligner = RetrievalAlignmentScorer(config)

# Healthy RAG case
query    = "What is the Sharpe ratio and how is it calculated?"
chunks   = pipeline.retrieve(query)
response = pipeline.generate(query, context=chunks)

result_a = aligner.score(query, response, chunks)
print("=== Healthy RAG ===")
print(f"  Alignment Score    : {result_a['alignment_score']:.4f}")
print(f"  Retrieval Relevance: {result_a['retrieval_relevance']:.4f}")
print(f"  Context Utilization: {result_a['context_utilization']:.4f}")
print(f"  Faithfulness       : {result_a['faithfulness']:.4f}")
print(f"  Flagged            : {'⚠️  YES' if result_a['is_flagged'] else '✅ No'}")

# Failure case: inject irrelevant chunks
bad_chunks = [
    "The Eiffel Tower was built in 1889 in Paris.",
    "Photosynthesis converts sunlight into energy in plants.",
    "The Great Wall of China is over 13,000 miles long.",
]
result_b = aligner.score(query, response, bad_chunks)
print("\n=== Degraded RAG (irrelevant chunks) ===")
print(f"  Alignment Score    : {result_b['alignment_score']:.4f}")
print(f"  Retrieval Relevance: {result_b['retrieval_relevance']:.4f}")
print(f"  Context Utilization: {result_b['context_utilization']:.4f}")
print(f"  Faithfulness       : {result_b['faithfulness']:.4f}")
print(f"  Flagged            : {'⚠️  YES' if result_b['is_flagged'] else '✅ No'}")

delta = result_a["alignment_score"] - result_b["alignment_score"]
print(f"\n📉 Alignment drop (healthy → degraded): {delta:.4f}")

# %% [markdown]
# ## Module 4 — Prompt Anomaly Detector

# %%
from llm_observability.modules.anomaly import AnomalyDetector
from llm_observability.data.loaders import BENIGN_FINANCIAL_PROMPTS, ADVERSARIAL_FINANCIAL_PROMPTS

anomaly_detector = AnomalyDetector(config)

# Fit on benign baseline
print(f"Fitting anomaly detector on {len(BENIGN_FINANCIAL_PROMPTS * 10)} benign queries...")
anomaly_detector.fit(BENIGN_FINANCIAL_PROMPTS * 10)
print("✅ Fitted.\n")

# Test benign queries
print("=== Benign Queries ===")
for q in BENIGN_FINANCIAL_PROMPTS[:5]:
    r = anomaly_detector.detect(q)
    flag = "🔴 ANOMALY" if r["is_anomaly"] else "✅ benign"
    print(f"  [{r['anomaly_score']:.3f}] {flag} | {q[:65]}")

print("\n=== Adversarial Queries ===")
for q in ADVERSARIAL_FINANCIAL_PROMPTS[:5]:
    r = anomaly_detector.detect(q)
    flag = "🔴 ANOMALY" if r["is_anomaly"] else "⚠️  missed"
    print(f"  [{r['anomaly_score']:.3f}] {flag} | {q[:65]}")

# %% [markdown]
# ## Summary — All Modules

# %%
print("=" * 60)
print("MODULE DEMO SUMMARY")
print("=" * 60)
print(f"\nModule 1 — Consistency")
print(f"  Mean score (10 queries): {df_c['consistency_score'].mean():.4f}")
print(f"  Flagged rate           : {df_c['flagged'].mean():.1%}")

print(f"\nModule 2 — Drift Detection")
if results_drift:
    df_d = pd.DataFrame(results_drift)
    print(f"  Drift first detected   : step {df_d[df_d['drift_detected']]['step'].min()}")
    print(f"  Mean drift magnitude   : {df_d['drift_magnitude'].mean():.4f}")

print(f"\nModule 3 — Retrieval Alignment")
print(f"  Healthy RAG score      : {result_a['alignment_score']:.4f}")
print(f"  Degraded RAG score     : {result_b['alignment_score']:.4f}")
print(f"  Drop detected          : {'✅ Yes' if result_b['is_flagged'] else '❌ No'}")

print(f"\nModule 4 — Anomaly Detection")
benign_scores  = [anomaly_detector.detect(q)["anomaly_score"] for q in BENIGN_FINANCIAL_PROMPTS]
adv_scores     = [anomaly_detector.detect(q)["anomaly_score"] for q in ADVERSARIAL_FINANCIAL_PROMPTS]
print(f"  Mean benign score      : {sum(benign_scores)/len(benign_scores):.4f}")
print(f"  Mean adversarial score : {sum(adv_scores)/len(adv_scores):.4f}")
print(f"  Separation             : {sum(adv_scores)/len(adv_scores) - sum(benign_scores)/len(benign_scores):.4f}")
print("\n✅ All modules working. Proceed to colab_03_experiments.py")
