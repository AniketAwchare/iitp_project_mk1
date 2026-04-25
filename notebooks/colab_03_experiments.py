# %% [markdown]
# # Notebook 03 — Run All Experiments
# **LLM Observability Framework · Financial Services**
#
# **No GPU needed** — all three experiments use SentenceTransformers (CPU OK).
#
# Experiments:
# 1. **Domain Shift Detection** — KS test flags query distribution change
# 2. **Adversarial Prompt Detection** — ensemble catches financial adversarial prompts
# 3. **RAG Alignment Failure** — alignment score separates healthy vs. degraded RAG
#
# Results are saved to `./results/` and visualised inline.

# %% [markdown]
# ## Setup

# %%
import sys, os
REPO_PATH = "/content/llm-observability-framework"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

print("✅ Ready. No GPU required for experiments.")

# %% [markdown]
# ## Experiment 1 — Domain Shift Detection

# %%
from llm_observability.experiments.domain_shift import run_all as run_domain_shift
from llm_observability.core.config import Config

config = Config()

print("Running Experiment 1: Domain Shift Detection")
print("Injection rates: 10%, 50%, 90% OOD queries\n")

ds_results = run_domain_shift(config=config)
df_ds = pd.DataFrame(ds_results)

# %%
print("\n=== Domain Shift Results Table ===")
print(df_ds[["injection_rate","precision","recall","f1","detection_latency_pct","detected_at_query"]].to_string(index=False))

# %%
# Visualise
fig1 = px.bar(
    df_ds, x="injection_rate", y=["precision","recall","f1"],
    barmode="group",
    title="Experiment 1: Domain Shift — Precision / Recall / F1",
    labels={"value":"Score","injection_rate":"OOD Injection Rate","variable":"Metric"},
    color_discrete_map={"precision":"#58a6ff","recall":"#3fb950","f1":"#f0883e"},
    template="plotly_dark",
)
fig1.add_hline(y=0.80, line_dash="dash", line_color="white",
               annotation_text="Target: 80%")
fig1.show()

# %%
fig2 = px.line(
    df_ds, x="injection_rate", y="detection_latency_pct",
    markers=True,
    title="Experiment 1: Detection Latency (fraction of OOD phase before alert)",
    labels={"detection_latency_pct":"Latency (fraction)","injection_rate":"OOD Injection Rate"},
    color_discrete_sequence=["#f0883e"],
    template="plotly_dark",
)
fig2.show()

# %%
# Check if target is met
print("\n=== Target Assessment ===")
for _, row in df_ds.iterrows():
    rate = int(row["injection_rate"] * 100)
    recall_ok = row["recall"] >= 0.80
    fpr_ok    = (1 - row["precision"]) <= 0.15
    print(f"  {rate}% injection: recall={row['recall']:.3f} {'✅' if recall_ok else '❌'} | "
          f"false_positive_rate={1-row['precision']:.3f} {'✅' if fpr_ok else '❌'}")

# %% [markdown]
# ## Experiment 2 — Adversarial Prompt Detection

# %%
from llm_observability.experiments.adversarial import run_all as run_adversarial

print("Running Experiment 2: Adversarial Prompt Detection")
print("Injection rates: 5%, 10%, 20% adversarial queries\n")

adv_results = run_adversarial(config=config)
df_adv = pd.DataFrame(adv_results)

# %%
print("\n=== Adversarial Detection Results ===")
print(df_adv[["injection_rate","precision","recall","f1","fpr"]].to_string(index=False))

# %%
fig3 = px.bar(
    df_adv, x="injection_rate", y=["precision","recall","f1"],
    barmode="group",
    title="Experiment 2: Adversarial Detection — Precision / Recall / F1",
    color_discrete_map={"precision":"#58a6ff","recall":"#3fb950","f1":"#f0883e"},
    template="plotly_dark",
)
fig3.add_hline(y=0.80, line_dash="dash", line_color="white",
               annotation_text="Target: 80%")
fig3.show()

# %%
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=df_adv["injection_rate"],
    y=df_adv["fpr"],
    name="False Positive Rate",
    marker_color="#f85149",
))
fig4.add_hline(y=0.15, line_dash="dash", line_color="#f0883e",
               annotation_text="Target FPR < 15%")
fig4.update_layout(
    title="Experiment 2: False Positive Rate by Injection Rate",
    xaxis_title="Adversarial Injection Rate",
    yaxis_title="FPR",
    template="plotly_dark",
)
fig4.show()

# %%
print("\n=== Target Assessment ===")
for _, row in df_adv.iterrows():
    rate = int(row["injection_rate"] * 100)
    recall_ok = row["recall"] >= 0.80
    fpr_ok    = row["fpr"] <= 0.15
    print(f"  {rate}% injection: recall={row['recall']:.3f} {'✅' if recall_ok else '❌'} | "
          f"FPR={row['fpr']:.3f} {'✅' if fpr_ok else '❌'}")

# %% [markdown]
# ## Experiment 3 — RAG Alignment Failure Detection

# %%
from llm_observability.experiments.rag_failure import run_experiment as run_rag

print("Running Experiment 3: RAG Alignment Failure Detection")
print("Conditions: Healthy | Irrelevant chunks | Stale chunks\n")

rag_summary = run_rag(config=config)

# %%
df_rag = pd.read_csv("./results/rag_failure/detailed_results.csv")

fig5 = px.box(
    df_rag, x="condition", y="alignment_score",
    color="condition",
    color_discrete_map={"healthy":"#3fb950","irrelevant":"#f85149","stale":"#f0883e"},
    title="Experiment 3: Alignment Score by RAG Condition",
    template="plotly_dark",
    points="all",
)
fig5.show()

# %%
fig6 = px.scatter(
    df_rag, x="alignment_score", y="quality_proxy",
    color="condition",
    color_discrete_map={"healthy":"#3fb950","irrelevant":"#f85149","stale":"#f0883e"},
    title=f"Experiment 3: Alignment Score vs. Response Quality (Pearson r = {rag_summary['pearson_r_alignment_vs_quality']:.4f})",
    trendline="ols",
    template="plotly_dark",
)
fig6.show()

# %%
print("\n=== RAG Alignment Results ===")
print(f"  Pearson r (alignment vs quality): {rag_summary['pearson_r_alignment_vs_quality']:.4f} {'✅' if rag_summary['pearson_r_alignment_vs_quality'] > 0.6 else '❌'} (target > 0.6)")
print(f"  Mean alignment — Healthy    : {rag_summary['mean_alignment_healthy']:.4f}")
print(f"  Mean alignment — Irrelevant : {rag_summary['mean_alignment_irrelevant']:.4f}")
print(f"  Mean alignment — Stale      : {rag_summary['mean_alignment_stale']:.4f}")
print(f"  Flag rate — Healthy         : {rag_summary['flag_rate_healthy']:.1%}")
print(f"  Flag rate — Irrelevant      : {rag_summary['flag_rate_irrelevant']:.1%}")
print(f"  Flag rate — Stale           : {rag_summary['flag_rate_stale']:.1%}")

# %% [markdown]
# ## Combined Results Summary

# %%
print("\n" + "=" * 65)
print("EXPERIMENT SUMMARY — ALL THREE CONDITIONS")
print("=" * 65)

print("\n[Experiment 1 — Domain Shift]")
best_ds = df_ds.loc[df_ds["f1"].idxmax()]
print(f"  Best F1: {best_ds['f1']:.4f} at {int(best_ds['injection_rate']*100)}% injection rate")
print(f"  Mean recall across rates: {df_ds['recall'].mean():.4f}")

print("\n[Experiment 2 — Adversarial Detection]")
best_adv = df_adv.loc[df_adv["f1"].idxmax()]
print(f"  Best F1: {best_adv['f1']:.4f} at {int(best_adv['injection_rate']*100)}% injection rate")
print(f"  Mean FPR across rates: {df_adv['fpr'].mean():.4f}")

print("\n[Experiment 3 — RAG Alignment]")
print(f"  Pearson r: {rag_summary['pearson_r_alignment_vs_quality']:.4f}")
print(f"  Alignment drop (healthy→irrelevant): "
      f"{rag_summary['mean_alignment_healthy'] - rag_summary['mean_alignment_irrelevant']:.4f}")

print("\nAll results saved to ./results/")
print("Launch dashboard: streamlit run llm_observability/dashboard/app.py")
