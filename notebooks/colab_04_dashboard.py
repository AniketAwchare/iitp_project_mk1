# %% [markdown]
# # Notebook 04 — Launch Streamlit Dashboard
# **LLM Observability Framework · Financial Services**
#
# This notebook launches the Streamlit dashboard in Google Colab
# using a public tunnel (via `pyngrok` or `localtunnel`).
#
# **Prerequisite:** Run `colab_03_experiments.py` first to generate results.

# %% [markdown]
# ## Option A — Launch via pyngrok (recommended)

# %%
import subprocess, sys

# Install pyngrok
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
print("✅ pyngrok installed")

# %%
import sys, os
REPO_PATH = "/content/llm-observability-framework"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)

# %%
from pyngrok import ngrok
import threading

# NOTE: For a persistent tunnel, sign up free at ngrok.com and set your auth token:
# ngrok.set_auth_token("YOUR_TOKEN_HERE")
# Without auth token, tunnels are temporary and have bandwidth limits.

def run_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "llm_observability/dashboard/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ])

# Start Streamlit in background thread
thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

import time
time.sleep(5)   # wait for Streamlit to start

# Open ngrok tunnel
public_url = ngrok.connect(8501)
print("=" * 55)
print(f"✅ Dashboard live at: {public_url}")
print("=" * 55)
print("\nOpen the URL above in your browser.")
print("The dashboard shows:")
print("  • Live Monitor     — module gauges")
print("  • Experiment 1     — domain shift charts")
print("  • Experiment 2     — adversarial detection charts")
print("  • Experiment 3     — RAG alignment charts")
print("  • Query Log        — logged queries + CSV export")

# %% [markdown]
# ## Option B — localtunnel (no signup required)

# %%
# If pyngrok fails, use localtunnel as fallback

subprocess.run(["npm", "install", "-g", "localtunnel"], check=True)

# Start Streamlit
thread2 = threading.Thread(target=run_streamlit, daemon=True)
thread2.start()
time.sleep(5)

# Open localtunnel
result = subprocess.run(
    ["lt", "--port", "8501"],
    capture_output=True, text=True, timeout=10
)
print("Localtunnel URL:", result.stdout.strip())

# %% [markdown]
# ## Option C — Render dashboard output inline (no tunnel needed)
#
# If tunnels are unavailable, display charts directly in the notebook:

# %%
import json
from pathlib import Path
import pandas as pd
import plotly.express as px

print("=== Inline Dashboard Output ===\n")

# Domain shift
ds_path = Path("./results/domain_shift/all_results.csv")
if ds_path.exists():
    df_ds = pd.read_csv(ds_path)
    print("Experiment 1 — Domain Shift:")
    print(df_ds[["injection_rate","precision","recall","f1"]].to_string(index=False))
    fig = px.bar(df_ds, x="injection_rate", y=["precision","recall","f1"],
                 barmode="group", title="Domain Shift Results",
                 template="plotly_dark")
    fig.show()

# Adversarial
adv_path = Path("./results/adversarial/all_results.csv")
if adv_path.exists():
    df_adv = pd.read_csv(adv_path)
    print("\nExperiment 2 — Adversarial Detection:")
    print(df_adv[["injection_rate","precision","recall","f1","fpr"]].to_string(index=False))
    fig2 = px.bar(df_adv, x="injection_rate", y=["precision","recall","f1"],
                  barmode="group", title="Adversarial Detection Results",
                  template="plotly_dark")
    fig2.show()

# RAG alignment
rag_path = Path("./results/rag_failure/summary.json")
if rag_path.exists():
    with open(rag_path) as f:
        rag_s = json.load(f)
    print("\nExperiment 3 — RAG Alignment:")
    for k, v in rag_s.items():
        print(f"  {k}: {v}")

# %% [markdown]
# ## Session Complete
#
# All four notebooks have been executed:
#
# | Notebook | Status |
# |---|---|
# | 01 Infrastructure | ✅ LLM loaded, FAISS index built |
# | 02 Modules Demo | ✅ All 4 modules tested live |
# | 03 Experiments | ✅ All 3 experiments run, results saved |
# | 04 Dashboard | ✅ Streamlit live (or inline charts) |
#
# **Next step:** Write the thesis — open `colab_05_thesis_data.py` to generate
# all tables and figures in thesis-ready format.
