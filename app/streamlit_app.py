"""
Streamlit demo for the Network Traffic Anomaly Detection learning project.

Run it from the repo root (the folder that contains app/, artifacts/, data/):
    streamlit run app/streamlit_app.py

This app only DISPLAYS results. It reads:
  - data/processed/network_traffic_cleaned.csv   (from notebooks 01-02)
  - artifacts/*.csv                              (saved by notebook 04)
It never trains a model, so it opens instantly and can't change your results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent      # repo root, wherever you launch from
DATA_PATH = ROOT / "data" / "processed" / "network_traffic_cleaned.csv"
ART = ROOT / "artifacts"

# From notebook 04, experiment 3: the 95th percentile of "best of 14 features"
# under shuffled labels. Update it if you re-run that experiment.
LUCK_95 = 0.592

st.set_page_config(page_title="Network Traffic Anomaly - Learning Project", layout="wide")


# -------------------------------------------------------------- helpers ----
@st.cache_data
def load_csv(path):
    """Read a CSV once and remember it (Streamlit re-runs the script on every click)."""
    return pd.read_csv(path)


def require(path):
    """Show a friendly message instead of a crash if a file is missing."""
    if not Path(path).exists():
        st.error(f"Missing file: {path}. Run notebook 04 (last cell) and put the output in artifacts/.")
        st.stop()
    return load_csv(path)


# ---------------------------------------------------------------- tabs -----
def tab_overview():
    st.subheader("Dataset overview")
    df = require(DATA_PATH).copy()
    df["label"] = df["label"].astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(df))
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Malicious share", f"{df['label'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        counts = df["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Normal", "Malicious"], counts.values, color=["steelblue", "indianred"])
        ax.set_ylabel("Records")
        ax.set_title("Class balance")
        st.pyplot(fig)
        st.caption("9 Normal for every 1 Malicious: a model that always says 'Normal' scores ~90% accuracy.")

    with right:
        feature = st.selectbox(
            "Feature to compare across classes",
            [c for c in df.columns if c not in ("label", "src_port")],
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df.loc[df["label"] == 0, feature], bins=20, alpha=0.55, density=True,
                label="Normal", color="steelblue")
        ax.hist(df.loc[df["label"] == 1, feature], bins=20, alpha=0.55, density=True,
                label="Malicious", color="indianred")
        ax.set_title(feature)
        ax.legend()
        st.pyplot(fig)
        st.caption("If the two histograms almost completely overlap, this feature cannot tell the classes apart.")


def tab_model_comparison():
    st.subheader("Model comparison (repeated 5x5 cross-validation, training data)")
    cmp = require(ART / "cv_model_comparison.csv")
    st.dataframe(cmp.round(3), hide_index=True)

    short = [m.split(" (")[0] for m in cmp["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, mean_col, std_col, ref, title in [
        (axes[0], "ROC-AUC mean", "ROC-AUC std", 0.5, "ROC-AUC (0.5 = random)"),
        (axes[1], "AP mean", "AP std", 0.10, "Average Precision (0.10 = random)"),
    ]:
        ax.bar(short, cmp[mean_col], yerr=cmp[std_col], capsize=4, color="steelblue")
        ax.axhline(ref, color="red", ls="--", label="random baseline")
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Error bars = spread across 25 folds. If a bar is within its error bar of the red line, "
               "there is no evidence that model learned anything.")


def tab_thresholds():
    st.subheader("Threshold explorer (out-of-fold probabilities)")
    oof = require(ART / "oof_probabilities.csv")
    options = {"Logistic Regression": "lr_oof_proba", "Random Forest (simple)": "rf_oof_proba"}
    name = st.selectbox("Model", list(options))
    y = oof["y_true"].to_numpy().astype(int)
    p = oof[options[name]].to_numpy()

    thr = st.slider("Decision threshold (flag as Malicious when probability >= this)", 0.0, 1.0, 0.5, 0.01)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    if auc < 0.6:
        st.warning(f"Out-of-fold ROC-AUC = {auc:.3f} (random = 0.5). Moving the slider just trades "
                   "missed attacks for false alarms along the random diagonal; it does not make the model better.")
    else:
        st.info(f"Out-of-fold ROC-AUC = {auc:.3f}, AP = {ap:.3f} (random AP = {y.mean():.2f}).")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall (attacks caught)", f"{recall:.2f}")
    m2.metric("Precision", f"{precision:.2f}")
    m3.metric("False-alarm rate", f"{fpr:.2f}")
    m4.metric("F1", f"{f1:.2f}")

    cm = pd.DataFrame([[tn, fp], [fn, tp]],
                      index=["Actual Normal", "Actual Malicious"],
                      columns=["Predicted Normal", "Predicted Malicious"])
    st.dataframe(cm)

    fpr_c, tpr_c, _ = roc_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr_c, tpr_c, label=f"model (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "r--", label="random guessing")
    axes[0].scatter([fpr], [recall], color="black", zorder=5, label="current threshold")
    axes[0].set_xlabel("False-alarm rate"); axes[0].set_ylabel("Recall"); axes[0].set_title("ROC curve")
    axes[0].legend()

    axes[1].hist(p[y == 0], bins=30, alpha=0.55, color="steelblue", label="Normal")
    axes[1].hist(p[y == 1], bins=30, alpha=0.55, color="indianred", label="Malicious")
    axes[1].axvline(thr, color="black", ls="--", label="threshold")
    axes[1].set_xlabel("Predicted probability of Malicious"); axes[1].set_title("Probability distributions")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("A useful model pushes the red histogram to the right of the blue one. Here they overlap almost fully.")


def tab_features():
    st.subheader("Does any single feature carry signal? (training data)")
    fa = require(ART / "feature_auc.csv").sort_values("AUC (oriented)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fa["feature"], fa["AUC (oriented)"], color="steelblue")
    ax.axvline(0.5, color="black", lw=1, label="no signal (0.5)")
    ax.axvline(LUCK_95, color="red", ls="--", label="what the best of 14 features reaches by luck (95%)")
    ax.set_xlim(0.45, max(0.65, fa["AUC (oriented)"].max() + 0.02))
    ax.set_xlabel("Single-feature ROC-AUC (oriented)")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.caption("A feature only counts as informative if it clearly passes the red line. See notebook 04, experiment 3.")


# ----------------------------------------------------------------- page ----
st.title("Network Traffic Anomaly Detection - Learning Project")
st.warning("Educational demo. The models here perform close to random and are NOT a reliable "
           "security tool. This app exists to show what the experiments found, honestly.")

tabs = st.tabs(["Dataset", "Model comparison", "Threshold explorer", "Feature signal"])
with tabs[0]:
    tab_overview()
with tabs[1]:
    tab_model_comparison()
with tabs[2]:
    tab_thresholds()
with tabs[3]:
    tab_features()
