"""
dashboard/app.py — Streamlit Monitoring Dashboard.

Run with:
    streamlit run llm_observability/dashboard/app.py

Displays real-time metrics for all four observability modules and
experiment results loaded from the ./results directory.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Observability — Financial Services",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0e1117; }
  .metric-card {
    background: #1c1f26; border-radius: 12px; padding: 1.2rem;
    border: 1px solid #2e3340; margin-bottom: 0.5rem;
  }
  .metric-title { font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 0.3rem; }
  .metric-value { font-size: 2.0rem; font-weight: 700; color: #e6edf3; }
  .flag-ok   { color: #3fb950; }
  .flag-warn { color: #f0883e; }
  .flag-crit { color: #f85149; }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; }
  .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

RESULTS_DIR = Path("./results")


# ── Helpers ───────────────────────────────────────────────────────────

def gauge(value: float, title: str, threshold_low: float = 0.5,
          threshold_high: float = 0.75) -> go.Figure:
    color = ("#3fb950" if value >= threshold_high
             else "#f0883e" if value >= threshold_low else "#f85149")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        title={"text": title, "font": {"color": "#8b949e", "size": 13}},
        number={"font": {"color": color, "size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#444"},
            "bar":  {"color": color},
            "bgcolor": "#1c1f26",
            "bordercolor": "#2e3340",
            "steps": [
                {"range": [0.0, threshold_low],  "color": "#2d1115"},
                {"range": [threshold_low, threshold_high], "color": "#2d2110"},
                {"range": [threshold_high, 1.0], "color": "#102d14"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=20, r=20, t=40, b=20), height=200,
    )
    return fig


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/LLM-Observability-58a6ff?style=for-the-badge", width=220)
    st.markdown("### ⚙️ Config")
    consistency_threshold = st.slider("Consistency threshold", 0.0, 1.0, 0.75, 0.05)
    alignment_threshold   = st.slider("Alignment threshold",   0.0, 1.0, 0.50, 0.05)
    anomaly_threshold     = st.slider("Anomaly threshold",     0.0, 1.0, 0.65, 0.05)
    st.divider()
    st.markdown("**Domain:** Financial Services  \n**Model:** Mistral 7B (4-bit)  \n**Stack:** 100% Open Source")


# ── Main header ───────────────────────────────────────────────────────

st.markdown("# 📊 LLM Observability Dashboard")
st.markdown("**Financial Services · Banking · Insurance · Trading**")
st.divider()

tabs = st.tabs([
    "🔴 Live Monitor",
    "📈 Experiment 1 — Domain Shift",
    "🛡️ Experiment 2 — Adversarial",
    "🔗 Experiment 3 — RAG Alignment",
    "📋 Query Log",
])


# ══════════════════════════════════════════════════════════════════════
# Tab 1 — Live Monitor
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Real-Time Module Metrics")
    st.caption("Metrics update as queries flow through the pipeline. "
               "Run the pipeline and refresh.")

    # -- Load latest log record if available ---
    log_files = sorted(Path("./logs").glob("queries_*.jsonl")) if Path("./logs").exists() else []
    latest_record = {}
    if log_files:
        lines = log_files[-1].read_text().strip().split("\n")
        if lines and lines[-1]:
            latest_record = json.loads(lines[-1])

    col1, col2, col3, col4 = st.columns(4)

    # Dummy values for demo if no live data
    c_score = latest_record.get("metadata", {}).get("consistency_score", 0.82)
    d_mag   = latest_record.get("metadata", {}).get("drift_magnitude",   0.12)
    a_score = latest_record.get("metadata", {}).get("alignment_score",   0.71)
    an_score= latest_record.get("metadata", {}).get("anomaly_score",     0.08)

    with col1:
        st.plotly_chart(gauge(c_score, "Consistency Score",
                              consistency_threshold - 0.1, consistency_threshold),
                        use_container_width=True)
    with col2:
        drift_val = 1.0 - min(d_mag * 5, 1.0)   # invert: higher magnitude = lower score
        st.plotly_chart(gauge(drift_val, "Drift Health (inverted)",
                              0.3, 0.6),
                        use_container_width=True)
    with col3:
        st.plotly_chart(gauge(a_score, "RAG Alignment Score",
                              alignment_threshold - 0.1, alignment_threshold),
                        use_container_width=True)
    with col4:
        anomaly_health = 1.0 - an_score
        st.plotly_chart(gauge(anomaly_health, "Anomaly Health (inverted)",
                              0.3, 0.6),
                        use_container_width=True)

    st.divider()
    # Status flags
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        ok = c_score >= consistency_threshold
        st.markdown(f"{'✅' if ok else '⚠️'} **Consistency** {'OK' if ok else 'FLAGGED'}")
    with f2:
        ok = d_mag < 0.3
        st.markdown(f"{'✅' if ok else '🔴'} **Drift** {'Stable' if ok else 'DETECTED'}")
    with f3:
        ok = a_score >= alignment_threshold
        st.markdown(f"{'✅' if ok else '⚠️'} **RAG Alignment** {'OK' if ok else 'FLAGGED'}")
    with f4:
        ok = an_score < anomaly_threshold
        st.markdown(f"{'✅' if ok else '🔴'} **Anomaly** {'None' if ok else 'DETECTED'}")

    if latest_record:
        st.divider()
        st.markdown("#### Latest Query")
        st.code(latest_record.get("query", "—"), language=None)
        st.markdown("**Response:**")
        st.info(latest_record.get("response", "—"))


# ══════════════════════════════════════════════════════════════════════
# Tab 2 — Domain Shift Experiment
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Experiment 1: Domain Shift Detection")
    st.caption("Queries shift from Banking → Insurance / Trading at increasing rates.")

    all_path = RESULTS_DIR / "domain_shift" / "all_results.csv"
    df_ds = load_csv(all_path)

    if df_ds.empty:
        st.warning("No results yet. Run: `python -m llm_observability.experiments.domain_shift`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_ds, x="injection_rate", y=["precision", "recall", "f1"],
                         barmode="group", title="Precision / Recall / F1 by Injection Rate",
                         color_discrete_map={"precision": "#58a6ff", "recall": "#3fb950", "f1": "#f0883e"},
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(df_ds, x="injection_rate", y="detection_latency_pct",
                           title="Detection Latency (fraction of injection phase)",
                           markers=True, template="plotly_dark",
                           color_discrete_sequence=["#58a6ff"])
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_ds.style.highlight_max(axis=0, color="#1a3a1a"), use_container_width=True)

    # Stream-level detail
    for rate_label in ["10", "50", "90"]:
        path = RESULTS_DIR / "domain_shift" / f"stream_rate{rate_label}.csv"
        df_s = load_csv(path)
        if not df_s.empty and "drift_magnitude" in df_s.columns:
            with st.expander(f"📊 Stream detail — {rate_label}% injection rate"):
                fig3 = px.line(df_s.reset_index(), x="query_index", y="drift_magnitude",
                               color_discrete_sequence=["#f0883e"], template="plotly_dark",
                               title=f"Drift Magnitude over Time ({rate_label}% OOD)")
                fig3.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Tab 3 — Adversarial Detection
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Experiment 2: Adversarial Prompt Detection")
    st.caption("Known financial adversarial prompts injected at 5%, 10%, 20% rates.")

    df_adv = load_csv(RESULTS_DIR / "adversarial" / "all_results.csv")

    if df_adv.empty:
        st.warning("No results yet. Run: `python -m llm_observability.experiments.adversarial`")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df_adv, x="injection_rate", y=["precision", "recall", "f1"],
                         barmode="group", title="Detection Performance by Injection Rate",
                         color_discrete_map={"precision": "#58a6ff", "recall": "#3fb950", "f1": "#f0883e"},
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(df_adv, x="injection_rate", y="fpr",
                          title="False Positive Rate",
                          color_discrete_sequence=["#f85149"], template="plotly_dark")
            fig2.add_hline(y=0.15, line_dash="dash", line_color="#f0883e",
                           annotation_text="Target FPR < 15%")
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df_adv, use_container_width=True)

        # Confusion matrix for 10% rate
        summary_10 = load_json(RESULTS_DIR / "adversarial" / "summary_rate10.json")
        if summary_10:
            st.markdown("#### Confusion Matrix (10% injection rate)")
            cm_data = pd.DataFrame({
                "Predicted Benign":     [summary_10.get("tn", 0), summary_10.get("fn", 0)],
                "Predicted Adversarial":[summary_10.get("fp", 0), summary_10.get("tp", 0)],
            }, index=["Actual Benign", "Actual Adversarial"])
            st.dataframe(cm_data, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Tab 4 — RAG Alignment
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Experiment 3: RAG Alignment Failure Detection")
    st.caption("Three conditions: Healthy RAG · Irrelevant chunks · Stale chunks")

    df_rag = load_csv(RESULTS_DIR / "rag_failure" / "detailed_results.csv")
    summary_rag = load_json(RESULTS_DIR / "rag_failure" / "summary.json")

    if df_rag.empty:
        st.warning("No results yet. Run: `python -m llm_observability.experiments.rag_failure`")
    else:
        if summary_rag:
            corr = summary_rag.get("pearson_r_alignment_vs_quality", 0)
            st.metric("Pearson r — Alignment Score vs. Response Quality", f"{corr:.4f}",
                      help="Target: > 0.6 to validate that alignment score predicts quality")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df_rag, x="condition", y="alignment_score",
                         color="condition",
                         color_discrete_map={"healthy": "#3fb950",
                                             "irrelevant": "#f85149",
                                             "stale": "#f0883e"},
                         title="Alignment Score Distribution by Condition",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df_rag, x="alignment_score", y="quality_proxy",
                              color="condition",
                              color_discrete_map={"healthy": "#3fb950",
                                                  "irrelevant": "#f85149",
                                                  "stale": "#f0883e"},
                              title="Alignment Score vs. Response Quality",
                              trendline="ols", template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1c1f26")
            st.plotly_chart(fig2, use_container_width=True)

        flag_cols = ["condition", "alignment_score", "faithfulness",
                     "retrieval_relevance", "context_utilization", "is_flagged"]
        available = [c for c in flag_cols if c in df_rag.columns]
        st.dataframe(df_rag[available], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# Tab 5 — Query Log
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Query Log")
    log_path = Path("./logs")
    records = []
    if log_path.exists():
        for lf in sorted(log_path.glob("queries_*.jsonl")):
            for line in lf.read_text().strip().split("\n"):
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if records:
        df_log = pd.DataFrame(records)[["timestamp", "query", "response"]].iloc[::-1]
        st.dataframe(df_log, use_container_width=True, height=500)

        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download log CSV", csv, "query_log.csv", "text/csv")
    else:
        st.info("No queries logged yet. Run the pipeline to start generating data.")

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption("LLM Observability Framework · IIT Patna M.Tech Thesis 2025-26 · "
           "Aniket Awchare · Supervisor: Sachin Rathore · 100% Open Source")
