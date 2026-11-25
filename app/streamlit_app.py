"""PrivFedXAI demo. Run from the repo root: python -m streamlit run app/streamlit_app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from app import backend as B
from src.data.preprocessing import SCALE_MV

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]   # verified order
NAMES = {"cnn": "Plain CNN", "attn": "Attention (centralized)", "fl": "Federated (FL)"}
PANEL3_CAPTION = ("Attention weights show where the model's per-class pooling places weight along the 10 s time "
                  "axis (one band for all 12 leads, because the first convolution mixes leads). Agreement with "
                  "reference ECG regions on fold 9 varied by class (mean IoU about 0.44 for CD, about 0.01 for "
                  "HYP), and those regions were a design choice, not a clinical standard. This is a description "
                  "of the model, not clinical localization.")
PANEL5_CAPTION = ("Mean |SHAP| per lead for the centralized attention model, computed once from 20 true-positive "
                  "fold-9 records per class (it does not change with the record selected on the left). One seed, "
                  "one background draw. The profile mostly follows signal amplitude (Spearman 0.78-0.84); after "
                  "accounting for amplitude, V1 and aVF stand out. It is not evidence of class-specific lead "
                  "importance, and nothing here is claimed about timing or clinical relevance.")
EPS = {"ε = 1": 1, "ε = 5": 5, "ε = 10": 10, "ε = ∞ (no DP)": float("inf")}


def table_model(eps):
    return "fl" if eps == float("inf") else f"dp_eps{eps}"


def fmt(v):
    return "n/a" if pd.isna(v) else f"{v:.3f}"

st.set_page_config(page_title="PrivFedXAI", layout="wide")
d = B.load_demo()
C = d["classes"]


def true_labels(i):
    return [c for c, v in zip(C, d["y"][i]) if v == 1]


def draw_leads(ax, sig):                       # sig: (12, 1000) in mV
    step = 2.2 * np.percentile(np.abs(sig), 99)
    t = np.arange(sig.shape[1]) / 100
    for i, lead in enumerate(sig):
        ax.plot(t, lead - i * step, lw=0.7, color="k")
    ax.set_yticks([-i * step for i in range(12)])
    ax.set_yticklabels(LEADS)
    ax.set_xlim(0, 10)
    ax.set_xlabel("time (s)")


k = st.sidebar.selectbox(
    "Demo record (fold 9)", range(len(d["ecg_id"])),
    format_func=lambda i: f"{int(d['ecg_id'][i])}  {'+'.join(true_labels(i))}")
x = d["x"][k]                                  # (12, 1000), already preprocessed and scaled

st.title("PrivFedXAI")
tabs = st.tabs(["1 Data input", "2 Prediction", "3 Attention", "4 Privacy", "5 SHAP"])

with tabs[0]:
    st.subheader("Data input")
    st.write(f"ecg_id {int(d['ecg_id'][k])} · PTB-XL fold 9 · true label(s): {', '.join(true_labels(k))}")
    fig, ax = plt.subplots(figsize=(11, 5))
    draw_leads(ax, x.numpy() * SCALE_MV)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Preprocessed signal as the models see it: 100 Hz, 10 s, 0.5-40 Hz band-pass, shown in mV.")

with tabs[1]:
    st.subheader("Prediction")
    S = B.scores(x)
    df = pd.DataFrame({NAMES[m]: p for m, p in S.items()}, index=C).round(3)
    df["true label"] = ["yes" if v == 1 else "" for v in d["y"][k]]
    st.dataframe(df)
    for m, p in S.items():
        flagged = ", ".join(c for c, v in zip(C, p) if v > 0.5) or "none"
        st.write(f"**{NAMES[m]}** flags: {flagged}")
    st.caption("Model score = sigmoid of the model output, cut at 0.5 (strict >). Scores are not calibrated "
               "probabilities; no calibration was fitted. On fold 9 the models flag somewhat more records than the "
               "true prevalence; the largest gap is the federated model for HYP (27% flagged vs 12.5% prevalence).")

with tabs[2]:
    st.subheader("Attention")
    a = B.attention(x)                                   # (125, 5), centralized attention model
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    draw_leads(ax0, x.numpy() * SCALE_MV)
    ax0.set_xlabel("")
    ax1.imshow(a.T, aspect="auto", extent=(-0.04, 9.96, 4.5, -0.5), vmin=0, vmax=a.max(), cmap="viridis")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([c + (" *" if v == 1 else "") for c, v in zip(C, d["y"][k])])
    ax1.set_xlim(0, 10)
    ax1.set_xlabel("time (s)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Max attention weight {a.max():.3f} (uniform would be {1 / 125:.3f}). * = true label.")
    st.caption(PANEL3_CAPTION)

with tabs[3]:
    st.subheader("Privacy budget")
    label = st.select_slider("Privacy budget (smaller ε = stronger privacy)", options=list(EPS),
                             value="ε = ∞ (no DP)")
    eps = EPS[label]

    m = B.fold10_metrics(table_model(eps)).loc[["auc", "f1", "mcc", "ppr"]]
    st.markdown("**Fold-10 test set (2,158 records), pooled macro**")
    st.table(pd.DataFrame({
        "value": [fmt(v) for v in m["value"]],
        "95% CI (record bootstrap)": [f"[{fmt(lo)}, {fmt(hi)}]" if pd.notna(lo) else "n/a"
                                      for lo, hi in zip(m["lo"], m["hi"])],
        "min-max over 3 runs": [f"{fmt(a)} to {fmt(b)}" if pd.notna(a) else "n/a"
                                for a, b in zip(m["vmin"], m["vmax"])],
    }, index=m.index.str.upper()))

    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (name, mod) in enumerate([("ε = 1", "dp_eps1"), ("ε = 5", "dp_eps5"),
                                     ("ε = 10", "dp_eps10"), ("no DP", "fl")]):
        r = B.fold10_metrics(mod).loc["auc"]
        ax.errorbar(i, r["value"], yerr=[[r["value"] - r["lo"]], [r["hi"] - r["value"]]], fmt="o", capsize=4,
                    color="tab:red" if mod == table_model(eps) else "tab:blue")
        ax.annotate(name, (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -14),
                    textcoords="offset points", ha="center")
    ax.set_xticks([])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("pooled macro AUC (95% CI)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Selected demo record: model scores at this ε**")
    R = B.dp_scores(x, eps)
    sc = pd.DataFrame({("FL, no DP" if n == "fl" else f"run {n[-1]}"): p for n, p in R.items()}, index=C).round(3)
    st.dataframe(sc)
    for n, p in R.items():
        who = "FL, no DP" if n == "fl" else f"run {n[-1]}"
        st.write(f"**{who}** flags: {', '.join(c for c, v in zip(C, p) if v > 0.5) or 'none'}")
    st.caption("Metrics come from the frozen fold-10 table (each model scored once). For DP rows the value is the "
               "mean of three independent runs and min-max shows the run-to-run spread; the interval covers record "
               "sampling only. PPR = predicted-positive rate (a point estimate, no interval). DP-SGD (Opacus) as "
               "configured: the drop from no-DP includes gradient clipping and Poisson sampling, not only noise, "
               "and privacy is per record, not per patient. The scores are for the selected fold-9 record; all "
               "three runs are shown and none is picked as best.")

with tabs[4]:
    st.subheader("SHAP lead importance")
    st.image("results/shap/attn_lead_importance_fold9.png")
    st.caption(PANEL5_CAPTION)
    with st.expander("Table: mean |SHAP| per lead and class"):
        st.dataframe(pd.read_csv("results/shap/attn_lead_importance_fold9.csv"))

st.divider()
st.caption("Research demo, not a medical device. Demo records come from PTB-XL fold 9 (validation, not training). "
           "Models were trained on folds 1-8; any record from those folds was seen in training.")