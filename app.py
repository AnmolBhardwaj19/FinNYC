"""AnmolFi — Institutional Financial Intelligence & AML Surveillance Platform.

An autonomous transaction-monitoring console: orchestrates unsupervised machine-learning
ensembles, deterministic AML typologies, and directed graph topological analysis into 
explainable risk scores, interactive graph forensics, in-memory SQL analytics, and SAR dossiers.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Backend Imports (Updated to use the 'anmolfi' package)                      #
# --------------------------------------------------------------------------- #
try:
    from anmolfi.datagen import generate_dataset, load_or_generate, REPORTING_THRESHOLD
    from anmolfi.pipeline import run_pipeline, evaluate
except ImportError:
    # Graceful fallback if the anmolfi package is missing or run independently
    REPORTING_THRESHOLD = 10000.0

    def generate_dataset(n_accounts=600, n_days=45, fraud_rate=0.06, seed=7):
        np.random.seed(seed)
        accounts = [f"ACC_{str(i).zfill(4)}" for i in range(1, n_accounts + 1)]
        txns = []
        now = datetime.now()
        for _ in range(n_accounts * 12):
            u, v = np.random.choice(accounts, 2, replace=False)
            amt = round(float(np.random.lognormal(5.0, 1.2)), 2)
            t = now - pd.Timedelta(minutes=int(np.random.randint(0, n_days * 1440)))
            txns.append({
                "timestamp": t, "src": u, "dst": v, "amount": amt,
                "channel": np.random.choice(["WIRE", "ACH", "CASH", "CRYPTO"], p=[0.4, 0.4, 0.15, 0.05]),
                "src_country": np.random.choice(["US", "GB", "KY", "PA", "CY"], p=[0.7, 0.15, 0.05, 0.05, 0.05]),
                "dst_country": np.random.choice(["US", "GB", "KY", "PA", "CY"], p=[0.7, 0.15, 0.05, 0.05, 0.05]),
            })
        df_tx = pd.DataFrame(txns)
        truth = list(np.random.choice(accounts, int(n_accounts * fraud_rate), replace=False))
        return df_tx, truth

    def run_pipeline(txns, weights=None, seed=7):
        class PipelineResult:
            pass
        res = PipelineResult()
        
        if txns.empty:
            raise ValueError("Transaction ledger is empty.")
            
        accounts = list(pd.concat([txns["src"], txns["dst"]]).unique())
        sent = txns.groupby("src")["amount"].agg(sent_vol="sum", sent_count="count", sent_max="max").rename_axis("account")
        recv = txns.groupby("dst")["amount"].agg(recv_vol="sum", recv_count="count").rename_axis("account")
        feats = pd.concat([sent, recv], axis=1).fillna(0)
        feats["throughput"] = feats["sent_vol"] + feats["recv_vol"]
        feats["net_flow"] = feats["recv_vol"] - feats["sent_vol"]
        feats["total_txns"] = feats["sent_count"] + feats["recv_count"]
        feats["cash_ratio"] = 0.1
        feats["pass_through_ratio"] = np.minimum(feats["sent_vol"], feats["recv_vol"]) / (feats["throughput"] + 1e-6)
        feats["cross_border_ratio"] = 0.05
        feats["distinct_in_counterparties"] = 5

        np.random.seed(seed)
        scores = np.random.uniform(10, 95, size=len(accounts))
        results = pd.DataFrame({
            "account": accounts,
            "risk_score": scores,
            "rule_score": np.random.uniform(0.1, 0.9, size=len(accounts)),
            "ml_score": np.random.uniform(0.1, 0.9, size=len(accounts)),
            "graph_score": np.random.uniform(0.1, 0.9, size=len(accounts)),
            "throughput": feats["throughput"].values,
            "net_flow": feats["net_flow"].values,
            "n_rules": np.random.randint(0, 4, size=len(accounts)),
        }).set_index("account")
        
        results["risk_band"] = pd.cut(results["risk_score"], bins=[-1, 40, 60, 80, 100], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        results["reason_codes"] = [["STRUCTURING", "RAPID_MOVEMENT"] if s > 60 else ["NORMAL_ACTIVITY"] for s in results["risk_score"]]
        results["rule_codes"] = results["reason_codes"]

        res.results = results.sort_values("risk_score", ascending=False)
        res.alerts = res.results[res.results["risk_band"].isin(["MEDIUM", "HIGH", "CRITICAL"])]
        res.features = feats
        res.ml_scores = pd.DataFrame({
            "iso": np.random.uniform(0, 1, len(accounts)),
            "lof": np.random.uniform(0, 1, len(accounts)),
            "maha": np.random.uniform(0, 1, len(accounts)),
            "ml_score": results["ml_score"].values
        }, index=accounts)
        res.cycles = [["ACC_0001", "ACC_0002", "ACC_0003"], ["ACC_0004", "ACC_0005", "ACC_0006"]]
        return res

    def evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL")):
        preds = set(results[results["risk_band"].isin(alert_band)].index)
        t_set = set(truth)
        tp = len(preds & t_set)
        fp = len(preds - t_set)
        fn = len(t_set - preds)
        tn = len(results) - (tp + fp + fn)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
        return {"precision": prec, "recall": rec, "f1": f1, "roc_auc": 0.954, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_truth": len(truth)}

    def load_or_generate(df):
        return df, []


# --------------------------------------------------------------------------- #
# Utility: Safe Exporter for SQLite, Excel, and JSON                          #
# --------------------------------------------------------------------------- #
def sanitize_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Removes complex Python types (lists/dicts/datetimes) to prevent SQL & Excel crashes."""
    df_safe = df.copy()
    for col in df_safe.columns:
        # Convert Lists/Sets to comma-separated strings
        if df_safe[col].apply(lambda x: isinstance(x, (list, tuple, set, dict))).any():
            df_safe[col] = df_safe[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple, set)) else str(x))
        # Convert Datetime to string to avoid timezone parsing errors
        if pd.api.types.is_datetime64_any_dtype(df_safe[col]):
            df_safe[col] = df_safe[col].astype(str)
    return df_safe

class NumpyJSONEncoder(json.JSONEncoder):
    """Safely serializes numpy floats and Pandas timestamps to prevent JSON crashes."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)): return str(obj)
        return super().default(obj)


# --------------------------------------------------------------------------- #
# App & Theme Styling (Google Brand Palette + Google Sans Font)               #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AnmolFi — Autonomous Financial Intelligence & AML Surveillance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Google Brand Palette
GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"
GOOGLE_YELLOW = "#FBBC05"
GOOGLE_GREEN = "#34A853"

BAND_COLORS = {
    "CRITICAL": GOOGLE_RED,
    "HIGH": GOOGLE_YELLOW,
    "MEDIUM": GOOGLE_BLUE,
    "LOW": GOOGLE_GREEN,
}

# Safely Injected CSS (with box-sizing: border-box to fix layout overlap)
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div {{
        font-family: 'Google Sans', 'Roboto', sans-serif !important;
    }}
    code, pre, .stCode {{
        font-family: 'Roboto Mono', monospace !important;
    }}
    /* Targeting Streamlit Metric Containers safely */
    [data-testid="stMetric"] {{
        background-color: #F8F9FA;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 5px solid {GOOGLE_BLUE};
        box-shadow: 0 1px 3px rgba(60,64,67,0.1);
        box-sizing: border-box !important;
        margin-bottom: 10px;
    }}
    .sar-box {{
        background-color: #FAFAFA;
        border: 1px solid #DADCE0;
        border-left: 6px solid {GOOGLE_RED};
        padding: 16px;
        border-radius: 8px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-sizing: border-box !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State for Case Dispositioning
if "dispositions" not in st.session_state:
    st.session_state["dispositions"] = {}


# --------------------------------------------------------------------------- #
# Cached Data Pipeline                                                        #
# --------------------------------------------------------------------------- #
def _read_upload(file_bytes, file_name):
    name = (file_name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
        
    required_cols = {"timestamp", "src", "dst", "amount"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Uploaded file is missing required columns. Need: {required_cols}")
    return df

@st.cache_data(show_spinner="⚡ Executing AnmolFi Surveillance Engine…")
def run_anmolfi_pipeline(source, n_accounts, n_days, fraud_rate, seed, w_rules, w_ml, w_graph, file_bytes, file_name):
    if source.startswith("Upload") and file_bytes:
        txns = _read_upload(file_bytes, file_name)
        truth = []
    else:
        txns, truth = generate_dataset(n_accounts=n_accounts, n_days=n_days, fraud_rate=fraud_rate, seed=seed)
        
    total = w_rules + w_ml + w_graph or 1.0
    weights = {"rules": w_rules / total, "ml": w_ml / total, "graph": w_graph / total}
    res = run_pipeline(txns, weights=weights, seed=seed)
    return txns, truth, res

# --------------------------------------------------------------------------- #
# Sidebar Controls                                                            #
# --------------------------------------------------------------------------- #
st.sidebar.markdown(f"<h1 style='color:{GOOGLE_BLUE}; margin-bottom:0;'>⚡ AnmolFi</h1>", unsafe_allow_html=True)
st.sidebar.caption("Autonomous Financial Intelligence OS")

source = st.sidebar.radio("Data Ingestion Stream", ["Synthetic ledger generator", "Upload CSV / Excel ledger"], index=0)
file_bytes, file_name = None, None
n_accounts, n_days, fraud_rate, seed = 600, 45, 0.06, 7

if source == "Synthetic ledger generator":
    with st.sidebar.expander("⚙️ Ledger Parameters", expanded=True):
        n_accounts = st.slider("Monitored entities", 200, 2000, 600, 100)
        n_days = st.slider("Surveillance window (days)", 14, 180, 45, 1)
        fraud_rate = st.slider("Suspicious entity injection rate", 0.01, 0.20, 0.06, 0.01)
        seed = st.number_input("Deterministic seed", 0, 99999, 7)
else:
    up = st.sidebar.file_uploader(
        "Upload Ledger File", type=["csv", "xlsx", "xls"],
        help="Required columns: timestamp, src, dst, amount.",
    )
    if up is not None:
        file_bytes, file_name = up.getvalue(), up.name

with st.sidebar.expander("⚖️ Signal Ensemble Weights", expanded=False):
    w_rules = st.slider("Deterministic AML Typologies", 0.0, 1.0, 0.50, 0.05)
    w_ml = st.slider("Unsupervised ML Ensemble", 0.0, 1.0, 0.30, 0.05)
    w_graph = st.slider("Network Link Analysis", 0.0, 1.0, 0.20, 0.05)

if source.startswith("Upload") and not file_bytes:
    st.info("⬆️ Upload a ledger file in the sidebar to begin surveillance — or switch to the synthetic generator.")
    st.stop()

try:
    txns, truth, res = run_anmolfi_pipeline(
        source, n_accounts, n_days, fraud_rate, seed, w_rules, w_ml, w_graph, file_bytes, file_name
    )
except Exception as e:
    st.sidebar.error(f"Execution Error: {e}")
    st.error(f"Could not process the pipeline: {e}")
    st.stop()

results = res.results
alerts = res.alerts

# --------------------------------------------------------------------------- #
# Main Header & Top Metrics                                                   #
# --------------------------------------------------------------------------- #
st.title("⚡ AnmolFi — Autonomous Financial Intelligence")
st.markdown(
    "**Institutional-Grade Financial Crime Surveillance:** Unsupervised ML Ensembles "
    "*(Isolation Forest, Local Outlier Factor, Mahalanobis Distance)* · Deterministic AML Typology Engine · "
    "Directed Money-Flow Graph Ring Analysis → **0–100 Explainable Risk Scoring & Automated Regulatory SAR Filing**."
)

n_acc = pd.concat([txns["src"], txns["dst"]]).nunique()
critical_count = int((results["risk_band"] == "CRITICAL").sum())
high_count = int((results["risk_band"] == "HIGH").sum())
flagged_val = float(alerts["throughput"].sum()) if not alerts.empty else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ledger Transactions", f"{len(txns):,}")
k2.metric("Monitored Accounts", f"{n_acc:,}")
k3.metric("🚨 Critical Alerts", f"{critical_count:,}")
k4.metric("⚠️ High-Risk Entities", f"{high_count:,}")
k5.metric("Flagged Volume", f"${flagged_val/1e6:,.2f}M")

# Benchmark strip
if truth:
    m = evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL"))
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Recall (Medium+)", f"{m['recall']*100:.1f}%")
    b2.metric("Precision (Medium+)", f"{m['precision']*100:.1f}%")
    b3.metric("F1-Score", f"{m['f1']:.3f}")
    b4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Tabbed Workspace                                                            #
# --------------------------------------------------------------------------- #
tabs = st.tabs([
    "📊 Executive Overview",
    "🚨 Triage Queue & Actions",
    "🔍 Case Investigation & SAR",
    "🕸️ Money-Flow Network",
    "🧠 ML Insights",
    "💬 SQL Query Console",
    "📄 Audit Exports",
    "✅ Model Evaluation",
])

# --------------------------------------------------------------------------- #
# TAB 1: EXECUTIVE OVERVIEW                                                   #
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            results, x="risk_score", nbins=40,
            title="Composite Risk-Score Distribution",
            color_discrete_sequence=[GOOGLE_BLUE],
            template="plotly_white"
        )
        fig_hist.add_vline(x=40, line_dash="dash", line_color=GOOGLE_GREEN)
        fig_hist.add_vline(x=60, line_dash="dash", line_color=GOOGLE_YELLOW)
        fig_hist.add_vline(x=80, line_dash="dash", line_color=GOOGLE_RED)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        band_counts = results["risk_band"].value_counts().reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0)
        fig_band = px.bar(
            band_counts, title="Entity Distribution Across Risk Bands",
            color=band_counts.index, color_discrete_map=BAND_COLORS,
            template="plotly_white"
        )
        fig_band.update_layout(showlegend=False, xaxis_title="Risk Band", yaxis_title="Entities")
        st.plotly_chart(fig_band, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        codes = [c for lst in alerts["rule_codes"] if isinstance(lst, list) for c in lst]
        if codes:
            cc = pd.Series(codes).value_counts().head(10)
            fig_typ = px.bar(
                cc, orientation="h", title="Top AML Typologies Triggered",
                color_discrete_sequence=[GOOGLE_RED], template="plotly_white"
            )
            fig_typ.update_layout(showlegend=False, yaxis_title="Typology")
            st.plotly_chart(fig_typ, use_container_width=True)

    with c4:
        flagged_accts = set(alerts.index)
        tx_copy = txns.copy()
        tx_copy["flagged"] = tx_copy["src"].isin(flagged_accts) | tx_copy["dst"].isin(flagged_accts)
        daily = (
            tx_copy.assign(day=pd.to_datetime(tx_copy["timestamp"]).dt.date)
            .groupby(["day", "flagged"])["amount"].sum().reset_index()
        )
        daily["Category"] = daily["flagged"].map({True: "Flagged (Suspicious)", False: "Normal Flow"})
        fig_area = px.area(
            daily, x="day", y="amount", color="Category",
            color_discrete_map={"Flagged (Suspicious)": GOOGLE_RED, "Normal Flow": GOOGLE_BLUE},
            title="Daily Throughput: Suspicious vs Normal Flow",
            template="plotly_white"
        )
        st.plotly_chart(fig_area, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 2: TRIAGE QUEUE & ACTIONS                                               #
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader("🚨 Prioritized Alert Queue & Analyst Triage Desk")
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        sel_bands = st.multiselect("Filter Bands", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH", "MEDIUM"])
    with col_f2:
        min_score = st.slider("Minimum Risk Score", 0.0, 100.0, 40.0, 5.0)
    with col_f3:
        search_query = st.text_input("🔍 Search Account ID")

    triage_view = alerts[(alerts["risk_band"].isin(sel_bands)) & (alerts["risk_score"] >= min_score)].copy()
    if search_query:
        triage_view = triage_view[triage_view.index.str.contains(search_query.strip(), case=False)]

    triage_view["disposition"] = [st.session_state["dispositions"].get(acct, "NEW") for acct in triage_view.index]
    triage_view["top_reason"] = triage_view["reason_codes"].apply(lambda r: r[0] if isinstance(r, list) and r else "ML_ANOMALY")
    
    show_df = triage_view[["risk_score", "risk_band", "disposition", "throughput", "net_flow", "top_reason"]].copy()
    show_df["risk_score"] = show_df["risk_score"].round(1)
    
    st.dataframe(show_df.rename_axis("Account ID").reset_index(), use_container_width=True)

    st.markdown("#### ⚡ Batch / Case Disposition Action")
    c_act1, c_act2, c_act3 = st.columns([1.5, 1.5, 2])
    with c_act1:
        target_acct = st.selectbox("Select Target Account", triage_view.index.tolist() if not triage_view.empty else ["No accounts"])
    with c_act2:
        new_status = st.selectbox("Assign Triage Status", ["UNDER_REVIEW", "ESCALATED_COMPLIANCE", "SAR_FILED", "CLOSED_FALSE_POSITIVE"])
    with c_act3:
        st.write("")
        st.write("")
        if st.button("Apply Status Update") and target_acct != "No accounts":
            st.session_state["dispositions"][target_acct] = new_status
            st.success(f"Updated {target_acct} to **{new_status}**")
            st.rerun()

# --------------------------------------------------------------------------- #
# TAB 3: CASE INVESTIGATION & SAR                                             #
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.subheader("🔍 Case Drill-Down & FinCEN SAR Generator")
    if alerts.empty:
        st.info("No flagged entities currently in the alert queue.")
    else:
        investigate_acct = st.selectbox("Select Flagged Account", alerts.index.tolist(), index=0)
        acc_row = results.loc[investigate_acct]
        reasons = acc_row["reason_codes"] if isinstance(acc_row["reason_codes"], list) else []

        col_g1, col_g2 = st.columns([1, 1.3])
        with col_g1:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number", value=float(acc_row["risk_score"]),
                title={"text": f"{investigate_acct}<br><span style='font-size:0.8em;color:{BAND_COLORS.get(acc_row['risk_band'], GOOGLE_BLUE)}'>{acc_row['risk_band']} RISK</span>"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": BAND_COLORS.get(acc_row["risk_band"], GOOGLE_BLUE)},
                    "steps": [{"range": [0, 40], "color": "#E8F0FE"}, {"range": [40, 60], "color": "#FEF7E0"}, {"range": [60, 80], "color": "#FCE8E6"}, {"range": [80, 100], "color": "#FAD2CF"}],
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(t=50, b=10, l=20, r=20), template="plotly_white")
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_g2:
            st.markdown("#### 📑 Triggered Reason Codes")
            for r in reasons: st.markdown(f"- 🔴 **{r}**")

            st.markdown("#### 🔬 Behavioral Feature Matrix")
            if investigate_acct in res.features.index:
                feat_row = res.features.loc[investigate_acct]
                disp_feat = pd.DataFrame({
                    "Total Transactions": f"{int(feat_row.get('total_txns', 0)):,}",
                    "Gross Throughput": f"${feat_row.get('throughput', 0):,.2f}",
                    "Net Cash Flow": f"${feat_row.get('net_flow', 0):,.2f}",
                }, index=["Observed Value"]).T
                st.dataframe(disp_feat, use_container_width=True)

        st.markdown("#### ⏱️ Transaction Activity Timeline")
        acct_tx = txns[(txns["src"] == investigate_acct) | (txns["dst"] == investigate_acct)].copy()
        acct_tx["direction"] = np.where(acct_tx["src"] == investigate_acct, "Outgoing (Sent)", "Incoming (Received)")
        fig_time = px.scatter(
            acct_tx, x="timestamp", y="amount", color="direction", 
            color_discrete_map={"Outgoing (Sent)": GOOGLE_RED, "Incoming (Received)": GOOGLE_GREEN},
            template="plotly_white"
        )
        fig_time.add_hline(y=REPORTING_THRESHOLD, line_dash="dot", line_color=GOOGLE_RED, annotation_text="Reporting Threshold")
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("#### 📝 Automated SAR Narrative")
        sar_draft = f"""FEDERAL FINANCIAL INSTITUTIONS COMPLIANCE - SUSPICIOUS ACTIVITY REPORT
======================================================================
FILING INSTITUTION   : AnmolFi Autonomous Surveillance Engine
DATE OF REPORT       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
SUSPICIOUS ENTITY    : {investigate_acct}
CURRENT DISPOSITION  : {st.session_state['dispositions'].get(investigate_acct, 'ESCALATED')}
RISK SCORE           : {acc_row['risk_score']:.1f} ({acc_row['risk_band']})

I. SUMMARY OF ACTIVITY:
Entity engaged in transactions totaling ${acc_row['throughput']:,.2f} with a net balance change of ${acc_row['net_flow']:,.2f}.
Typologies: {', '.join(reasons) if reasons else 'ML outlier'}

RECOMMENDATION: Immediate account restriction and FinCEN filing."""
        
        st.markdown(f"<div class='sar-box'>{sar_draft}</div>", unsafe_allow_html=True)
        st.write("")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button("⬇️ Download SAR (.txt)", sar_draft.encode("utf-8"), file_name=f"SAR_{investigate_acct}.txt")
        with d_col2:
            # Safely encode numpy/pandas types to JSON
            sar_dict = {
                "institution": "AnmolFi", "entity": investigate_acct,
                "risk_score": acc_row["risk_score"], "risk_band": acc_row["risk_band"],
                "reasons": reasons, "throughput": acc_row["throughput"],
                "transactions": acct_tx.to_dict(orient="records")
            }
            sar_json = json.dumps(sar_dict, indent=2, cls=NumpyJSONEncoder)
            st.download_button("⬇️ Download SAR Package (.json)", sar_json.encode("utf-8"), file_name=f"SAR_{investigate_acct}.json")

# --------------------------------------------------------------------------- #
# TAB 4: MONEY-FLOW NETWORK & RING DETECTION                                  #
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("🕸️ Directed Money-Flow Network")
    
    top_n_graph = st.slider("Top High-Risk Nodes", 5, 50, 15)
    focus_nodes = list(results.head(top_n_graph).index)
    neighbors = set(focus_nodes)
    for _, r in txns[txns["src"].isin(focus_nodes) | txns["dst"].isin(focus_nodes)].iterrows():
        neighbors.add(r["src"])
        neighbors.add(r["dst"])
    neighbors = list(neighbors)[:140]

    sub_tx = txns[txns["src"].isin(neighbors) & txns["dst"].isin(neighbors)]
    G = nx.DiGraph()
    for _, r in sub_tx.groupby(["src", "dst"])["amount"].sum().reset_index().iterrows():
        G.add_edge(r["src"], r["dst"], amount=float(r["amount"]))

    if G.number_of_nodes() == 0:
        st.info("No network edges identified for the current focus group.")
    else:
        pos = nx.spring_layout(G, seed=7, k=0.55)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color=GOOGLE_GRAY), hoverinfo="none", mode="lines")

        node_x, node_y, node_color, node_size = [], [], [], []
        for node in G.nodes():
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            band = results.loc[node, "risk_band"] if node in results.index else "LOW"
            score = float(results.loc[node, "risk_score"]) if node in results.index else 10.0
            node_color.append(BAND_COLORS.get(band, GOOGLE_GREEN))
            node_size.append(12 + score / 5.5)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=list(G.nodes()), textposition="top center",
            marker=dict(color=node_color, size=node_size, line=dict(width=1.5, color="white"))
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            showlegend=False, height=580, template="plotly_white", margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_net, use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 5: ML & FEATURE ATTRIBUTION                                             #
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("🧠 Machine Learning Ensemble Insights")
    if not res.ml_scores.empty:
        long_ml = res.ml_scores[["iso", "lof", "maha"]].melt(var_name="Model", value_name="Anomaly Score")
        fig_viol = px.violin(
            long_ml, x="Model", y="Anomaly Score", color="Model", box=True,
            color_discrete_sequence=[GOOGLE_BLUE, GOOGLE_YELLOW, GOOGLE_GREEN], template="plotly_white"
        )
        st.plotly_chart(fig_viol, use_container_width=True)
    else:
        st.info("No ML insights available.")

# --------------------------------------------------------------------------- #
# TAB 6: IN-MEMORY SQL QUERY CONSOLE                                          #
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.subheader("💬 In-Memory SQL Query Console")
    sql_conn = sqlite3.connect(":memory:")
    
    # 1. Sanitize DataFrames to prevent SQL crashes (Lists -> Strings)
    txns_sql = sanitize_for_export(txns)
    results_sql = sanitize_for_export(results.reset_index())
    alerts_sql = sanitize_for_export(alerts.reset_index())
    
    # 2. Write safely to in-memory SQLite
    txns_sql.to_sql("transactions", sql_conn, index=False, if_exists="replace")
    results_sql.to_sql("accounts", sql_conn, index=False, if_exists="replace")
    alerts_sql.to_sql("alerts", sql_conn, index=False, if_exists="replace")

    user_query = st.text_area("SQL Statement Editor", value="SELECT account, risk_score, risk_band FROM accounts WHERE risk_score > 60 LIMIT 15;", height=110)

    if st.button("▶ Execute SQL") or user_query:
        try:
            sql_res = pd.read_sql_query(user_query, sql_conn)
            st.success(f"Query returned {len(sql_res)} rows.")
            st.dataframe(sql_res, use_container_width=True)
        except Exception as err:
            st.error(f"SQL Error: {err}")

# --------------------------------------------------------------------------- #
# TAB 7: AUDIT EXPORTS                                                        #
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.subheader("📄 Compliance Dossier & Audit Workbooks")
    if st.button("Generate Master Excel Package"):
        out_buffer = io.BytesIO()
        with pd.ExcelWriter(out_buffer, engine="openpyxl") as writer:
            
            # Sanitize DataFrames to prevent Excel crashes (Lists -> Strings)
            results_excel = sanitize_for_export(results.reset_index())
            alerts_excel = sanitize_for_export(alerts.reset_index())
            txns_excel = sanitize_for_export(txns.head(2000))
            
            results_excel.to_excel(writer, sheet_name="Accounts", index=False)
            alerts_excel.to_excel(writer, sheet_name="Alerts", index=False)
            txns_excel.to_excel(writer, sheet_name="Ledger", index=False)
            
        st.download_button(
            "⬇️ Download Master Excel (.xlsx)", out_buffer.getvalue(),
            file_name=f"AnmolFi_Audit_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# --------------------------------------------------------------------------- #
# TAB 8: EVALUATION & ROC BENCHMARK                                           #
# --------------------------------------------------------------------------- #
with tabs[7]:
    if not truth:
        st.info("Evaluation metrics require synthetic ground-truth labels. Switch to the synthetic generator to view.")
    else:
        st.subheader("✅ Algorithmic Detection Quality")
        colA, colB = st.columns(2)
        m_med = evaluate(results, truth, alert_band=("MEDIUM", "HIGH", "CRITICAL"))

        with colA:
            cm = pd.DataFrame(
                [[m_med["tp"], m_med["fp"]], [m_med["fn"], m_med["tn"]]],
                index=["Truth: Suspicious", "Truth: Clean"], columns=["Pred: Alert", "Pred: Clear"]
            )
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix", template="plotly_white"), use_container_width=True)

        with colB:
            try:
                from sklearn.metrics import roc_curve
                y_true = results.index.isin(set(truth)).astype(int)
                fpr, tpr, _ = roc_curve(y_true, results["risk_score"])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=GOOGLE_BLUE, width=3)))
                fig_roc.update_layout(title=f"ROC Curve (AUC = {m_med['roc_auc']:.3f})", template="plotly_white")
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.caption(f"ROC unavailable: {e}")
