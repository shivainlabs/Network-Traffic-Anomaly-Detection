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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent      # repo root, wherever you launch from
DATA_PATH = ROOT / "data" / "processed" / "network_traffic_cleaned.csv"
ART = ROOT / "artifacts"

# From notebook 04, experiment 3: the 95th percentile of "best of 14 features"
# under shuffled labels. Update it if you re-run that experiment.
LUCK_95 = 0.592

NORMAL_COLOR = "#4C78A8"
MAL_COLOR = "#E45756"

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


def show_fig(fig):
    """Display a Plotly chart; works on both older and newer Streamlit versions."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.plotly_chart(fig)


FEATURE_INFO = {
    "packet_size": "How big each network packet is.",
    "inter_arrival_time": "The time gap between one packet and the next.",
    "dst_port": "Which 'door' on the receiving computer the packet goes to (53 = name lookups, 80 = web, 443 = secure web).",
    "packet_count_5s": "How many packets were sent in a 5-second window, a 'how busy is the traffic' measure.",
    "spectral_entropy": "How irregular or 'spread out' the traffic pattern is over time (from a wavelet analysis).",
    "frequency_band_energy": "How much of the traffic's rhythm sits in certain frequency bands (from a wavelet analysis).",
}


def describe_feature(name):
    """Plain-English meaning of a column name."""
    if name in FEATURE_INFO:
        return FEATURE_INFO[name]
    if name.startswith("protocol_type_"):
        return f"1 if the packet used the {name.split('_')[-1]} protocol, otherwise 0."
    if name.startswith("src_ip_"):
        return f"1 if the packet came from {name.replace('src_ip_', '')}, otherwise 0."
    if name.startswith("dst_ip_"):
        return f"1 if the packet was sent to {name.replace('dst_ip_', '')}, otherwise 0."
    if name.startswith("tcp_flags_"):
        return f"1 if the connection flag was {name.replace('tcp_flags_', '')}, otherwise 0."
    return "A measurement taken from the network traffic."


# ------------------------------------------------------------- sidebar -----
def sidebar_glossary():
    st.sidebar.header("Quick glossary")
    st.sidebar.caption("Stuck on a word? Open it here.")
    terms = {
        "Model": "A computer program that learns patterns from labelled examples, then guesses the label for new examples.",
        "Label": "The right answer for each record: 0 = normal traffic, 1 = malicious (an attack).",
        "Feature": "One measurement about a record, such as packet size. The model uses features to decide.",
        "Suspicion score": "The number (0 to 1) a model gives each record. Higher = 'more likely an attack'.",
        "Cut-off (threshold)": "The line you choose. Records scoring above it get flagged as attacks.",
        "Accuracy": "Share of all guesses that were right. Misleading here because 90% of traffic is normal.",
        "Recall (attacks caught)": "Of all real attacks, how many did we catch?",
        "Precision": "Of everything we flagged, how much was a real attack?",
        "False alarm": "A normal record wrongly flagged as an attack.",
        "ROC-AUC": "Pick one random attack and one random normal record: how often does the model give the attack the higher score? 0.5 = coin flip, 1.0 = perfect.",
        "Average Precision (AP)": "How pure the model's top picks are. Random guessing scores about the share of attacks (0.10 here).",
        "Cross-validation": "Split the training data into 5 parts, train on 4, test on the 5th, rotate, and repeat. Gives many scores instead of one lucky one.",
        "Out-of-fold": "Each record was scored by a model that never saw that record in training, like grading an exam on unseen questions.",
    }
    for term, text in terms.items():
        with st.sidebar.expander(term):
            st.write(text)


# ---------------------------------------------------------------- tabs -----
def tab_start(cmp):
    st.subheader("What is this project about?")
    st.markdown(
        """
**The goal.** Computers send each other small messages called *packets*. Most are normal, but a few
come from attacks. Can we teach a computer program, called a **model**, to spot the attacks?

**How a model learns.** We show it many example records that are already labelled *normal* or
*malicious*. Each record has 16 measurements (packet size, timing, port, and so on). The model searches for
patterns that tell the two groups apart. Afterwards it gives every record a **suspicion score** from 0 to 1.

**What we did.**
1. **Looked at the data** to see what we have (tab 2).
2. **Cleaned and prepared it** (notebooks 01 and 02).
3. **Trained three models** (Logistic Regression, Random Forest, XGBoost) **and a "do-nothing" baseline** to compare against (tab 3).
4. **Tested them fairly** on data they had not seen (tabs 3 and 4).
5. **Asked: is the result real, or just luck?** (tab 5 and notebook 04).
"""
    )

    base = cmp[cmp["model"].str.startswith("Dummy")].iloc[0]
    real = cmp[~cmp["model"].str.startswith("Dummy")]
    best = real.loc[real["ROC-AUC mean"].idxmax()]

    st.subheader("What happened?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score from pure guessing", f"{base['ROC-AUC mean']:.2f}", help="ROC-AUC. 0.5 = a coin flip.")
    c2.metric("Best model we trained", f"{best['ROC-AUC mean']:.2f}", help=f"{best['model']} (ROC-AUC, cross-validated)")
    c3.metric("Normal ups and downs", f"+/-{best['ROC-AUC std']:.2f}",
              help="How much that score changes from one test round to the next.")
    if best["ROC-AUC mean"] - base["ROC-AUC mean"] <= best["ROC-AUC std"]:
        st.markdown(
            """
The models are **no better than guessing**. Their scores differ from a coin flip by less than the normal
ups and downs between test rounds.

**Why?** A model can only find patterns that exist. The experiments in notebook 04 suggest the 16
measurements in this dataset barely differ between normal and malicious traffic. Trying fancier models
would not fix that.
"""
        )
    else:
        st.markdown(
            "The best model scores above guessing by more than the normal ups and downs. "
            "Check tabs 3 and 5 to see whether that is a real, useful gain."
        )
    st.success(
        "This is still a useful result. Knowing *why* a model fails, and proving it with a fair test, "
        "is a core machine learning skill. Use the tabs above to see the evidence."
    )


def tab_data():
    st.subheader("The data: what are we working with?")
    df = require(DATA_PATH).copy()
    df["label"] = df["label"].astype(int)
    df["Traffic type"] = df["label"].map({0: "Normal", 1: "Malicious"})
    n, n_mal = len(df), int(df["label"].sum())

    st.markdown(
        "Each **row** is one network record. The **label** column is the right answer: "
        "`0` = normal traffic, `1` = malicious. The other columns are the measurements (**features**) "
        "a model can look at."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", n)
    c2.metric("Measurements (features)", df.shape[1] - 2)
    c3.metric("Share that are attacks", f"{n_mal / n:.0%}")
    with st.expander("Look at the raw table (first 10 rows)"):
        st.dataframe(df.drop(columns=["Traffic type"]).head(10))

    st.markdown("---")
    st.markdown("#### Problem 1: attacks are rare")
    counts = df["Traffic type"].value_counts().reindex(["Normal", "Malicious"])
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=[NORMAL_COLOR, MAL_COLOR],
        text=[f"{v} ({v / n:.0%})" for v in counts.values], textposition="outside",
        hovertemplate="%{x}: %{y} records<extra></extra>",
    ))
    fig.update_layout(title="How many records of each type?", yaxis_title="Number of records",
                      height=340, margin=dict(t=50, b=20))
    show_fig(fig)
    lazy_acc = 1 - n_mal / n
    st.info(
        f"A lazy 'model' that says **Normal** for every record is right **{lazy_acc:.0%}** of the time "
        f"but catches **0 of the {n_mal}** attacks. That is why we cannot judge models by accuracy alone."
    )

    st.markdown("---")
    st.markdown("#### Problem 2: do attacks look different from normal traffic?")
    st.markdown(
        "Pick a measurement. If attacks and normal traffic **look different** on it, a model can use it. "
        "If the two groups look the **same**, it is useless for telling them apart."
    )
    features = [c for c in df.columns if c not in ("label", "Traffic type", "src_port")]
    feature = st.selectbox("Measurement to look at", features)
    st.caption(describe_feature(feature))

    if df[feature].nunique() <= 6:
        prop = pd.crosstab(df[feature], df["Traffic type"], normalize="columns")
        fig = go.Figure()
        for group, color in [("Normal", NORMAL_COLOR), ("Malicious", MAL_COLOR)]:
            fig.add_bar(x=[str(v) for v in prop.index], y=prop[group].values, name=group,
                        marker_color=color,
                        hovertemplate="value %{x}: %{y:.1%} of " + group + " records<extra></extra>")
        fig.update_layout(barmode="group", title=f"{feature}: share of each group with each value",
                          xaxis_title="Value", yaxis_title="Share of that group", yaxis_tickformat=".0%",
                          height=360, margin=dict(t=50, b=20))
        show_fig(fig)
        st.markdown("**How to read it:** if the blue and red bars are about the same height for every value, "
                    "this feature cannot tell the groups apart.")
    else:
        fig = go.Figure()
        for group, color in [("Normal", NORMAL_COLOR), ("Malicious", MAL_COLOR)]:
            fig.add_histogram(x=df.loc[df["Traffic type"] == group, feature], name=group,
                              marker_color=color, opacity=0.6, histnorm="probability density", nbinsx=25)
        fig.update_layout(barmode="overlay", title=f"{feature}: how values are spread in each group",
                          xaxis_title=f"{feature} (rescaled to 0-1)", yaxis_title="Density",
                          height=360, margin=dict(t=50, b=20))
        show_fig(fig)
        m0 = df.loc[df["label"] == 0, feature].mean()
        m1 = df.loc[df["label"] == 1, feature].mean()
        st.markdown(f"Average for **Normal**: {m0:.2f}. Average for **Malicious**: {m1:.2f}.")
        st.markdown("**How to read it:** if the blue and red shapes sit almost on top of each other, "
                    "this feature cannot tell the groups apart. (Values were rescaled to 0-1 in the "
                    "cleaning step, so compare the shapes, not the units.)")


def tab_models(cmp):
    st.subheader("The models: who did better than guessing?")
    st.markdown("We trained three models and compared them with a baseline that does not learn anything.")
    st.table(pd.DataFrame({
        "Contestant": ["Baseline (Dummy)", "Logistic Regression", "Random Forest", "XGBoost"],
        "Plain-English idea": [
            "Ignores all measurements and gives everyone the same score. This is the bar to beat.",
            "Adds up the measurements with weights, like a scoring checklist.",
            "Many simple decision trees ('is packet size above X?') that vote together.",
            "Trees built one after another, each fixing the mistakes of the previous ones.",
        ],
    }))

    st.markdown("#### How were they scored?")
    st.markdown(
        "Each model was tested with **repeated cross-validation**: the training data is split into 5 parts, "
        "the model trains on 4 and is graded on the 5th, rotating through all 5 parts, and the whole process "
        "is repeated 5 times. That gives 25 scores per model, so we see the **average** and the **ups and downs**."
    )
    metric = st.radio("Which score do you want to see?", ["ROC-AUC", "Average Precision"], horizontal=True)
    if metric == "ROC-AUC":
        mean_col, std_col = "ROC-AUC mean", "ROC-AUC std"
        st.markdown("**ROC-AUC:** pick one random attack and one random normal record. How often does the model "
                    "give the attack the higher suspicion score? **0.5 = coin flip, 1.0 = perfect.**")
    else:
        mean_col, std_col = "AP mean", "AP std"
        st.markdown("**Average Precision:** how many of the model's top picks are real attacks? "
                    "Guessing scores about the **share of attacks** (0.10 here), not 0.5.")

    base_row = cmp[cmp["model"].str.startswith("Dummy")].iloc[0]
    baseline = float(base_row[mean_col])
    short = [m.split(" (")[0] for m in cmp["model"]]
    fig = go.Figure(go.Bar(
        x=short, y=cmp[mean_col], error_y=dict(type="data", array=cmp[std_col], visible=True),
        marker_color=["#9E9E9E"] + [NORMAL_COLOR] * (len(cmp) - 1),
        hovertemplate="%{x}<br>average %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color=MAL_COLOR,
                  annotation_text=f"pure guessing = {baseline:.2f}", annotation_position="top left")
    fig.update_layout(title=f"{metric} (average of 25 test rounds; whiskers show the ups and downs)",
                      yaxis_title=metric, height=400, margin=dict(t=60, b=20))
    show_fig(fig)

    def verdict(row):
        if row["model"].startswith("Dummy"):
            return "Baseline (guessing)"
        diff = row[mean_col] - baseline
        if abs(diff) <= row[std_col]:
            return "Same as guessing (difference is within normal ups and downs)"
        return "Better than guessing" if diff > 0 else "Worse than guessing"

    table = cmp[["model", mean_col, std_col]].copy()
    table.columns = ["Model", "Average score", "Ups and downs (+/-)"]
    table["Verdict"] = cmp.apply(verdict, axis=1)
    st.dataframe(table.round(3), hide_index=True)
    st.info("If a model's bar is within its whiskers of the red line, we have no evidence it learned anything real.")


def tab_cutoff():
    st.subheader("Try the cut-off: what does 'flagging attacks' really look like?")
    st.markdown(
        "After training, a model gives every record a **suspicion score** from 0 to 1. To raise alarms we must "
        "choose a **cut-off**: records scoring at or above it are flagged as attacks. "
        "**Drag the slider** and watch the trade-off: a low cut-off flags a lot (catches more attacks "
        "but raises many false alarms), a high cut-off flags little (few false alarms but more attacks missed)."
    )
    st.caption("These scores are 'out-of-fold': each of the 800 training records was scored by a model "
               "that never saw it during training, like grading an exam on unseen questions.")

    oof = require(ART / "oof_probabilities.csv")
    options = {"Logistic Regression": "lr_oof_proba", "Random Forest": "rf_oof_proba"}
    name = st.radio("Model", list(options), horizontal=True)
    y = oof["y_true"].to_numpy().astype(int)
    p = oof[options[name]].to_numpy()
    n, attacks = len(y), int(y.sum())
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    weak = auc < 0.6

    thr = st.slider("Cut-off: flag a record as an attack when its score is at least...", 0.0, 1.0, 0.5, 0.01)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    flagged = tp + fp
    precision = tp / flagged if flagged else 0.0
    recall = tp / attacks if attacks else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # What would random flagging do? (hypergeometric mean and spread)
    expected = flagged * attacks / n
    spread = (flagged * (attacks / n) * (1 - attacks / n) * (n - flagged) / (n - 1)) ** 0.5 if n > 1 else 0.0

    st.markdown("#### What this cut-off does")
    st.markdown(
        f"At a cut-off of **{thr:.2f}** the model flagged **{flagged} of {n}** records. "
        f"It caught **{tp} of the {attacks}** real attacks and raised **{fp} false alarms**. "
        f"Flagging the same number of records **completely at random** would be expected to catch about "
        f"**{expected:.0f}** attacks."
    )
    if flagged == 0:
        st.info("Nothing is flagged at this cut-off, so nothing is caught and there are no false alarms.")
    elif abs(tp - expected) <= 2 * spread:
        st.warning("The model caught about as many attacks as random flagging would. "
                   "Moving the slider only trades missed attacks for false alarms; it does not make the model smarter.")
    elif tp > expected:
        st.success("The model caught clearly more attacks than random flagging would.")
    else:
        st.warning("The model caught fewer attacks than random flagging would.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attacks caught", f"{tp} of {attacks}", help="Recall: " + f"{recall:.0%}" + " of real attacks.")
    m2.metric("Attacks missed", fn)
    m3.metric("False alarms", fp, help=f"{fpr:.0%} of normal records were wrongly flagged.")
    m4.metric("Flags that were real", f"{precision:.0%}",
              help=f"Precision. Random flagging gives about {attacks / n:.0%}, the share of attacks.")

    left, right = st.columns(2)
    with left:
        z = [[tn, fp], [fn, tp]]
        labels = [[f"Correctly cleared<br>{tn}", f"False alarm<br>{fp}"],
                  [f"Attack missed<br>{fn}", f"Attack caught<br>{tp}"]]
        fig = go.Figure(go.Heatmap(
            z=z, x=["Model says Normal", "Model says Attack"], y=["Really Normal", "Really Attack"],
            text=labels, texttemplate="%{text}", colorscale="Blues", showscale=False, hoverinfo="skip"))
        fig.update_layout(title="Right and wrong calls", height=340, margin=dict(t=50, b=20),
                          yaxis=dict(autorange="reversed"))
        show_fig(fig)
        st.caption("Top-left and bottom-right are correct calls. The other two boxes are the mistakes.")
    with right:
        fig = go.Figure()
        fig.add_histogram(x=p[y == 0], name="Really Normal", marker_color=NORMAL_COLOR, opacity=0.6, nbinsx=30)
        fig.add_histogram(x=p[y == 1], name="Really Attack", marker_color=MAL_COLOR, opacity=0.6, nbinsx=30)
        fig.add_vline(x=thr, line_dash="dash", line_color="black", annotation_text="cut-off")
        fig.update_layout(barmode="overlay", title="Suspicion scores the model gave",
                          xaxis_title="Suspicion score (0 to 1)", yaxis_title="Number of records",
                          height=340, margin=dict(t=50, b=20))
        show_fig(fig)
        st.caption("A useful model pushes the red pile to the right of the blue pile."
                   + (" Here the two piles overlap almost completely." if weak else ""))

    st.markdown("#### The big picture: the ROC curve")
    fpr_c, tpr_c, _ = roc_curve(y, p)
    fig = go.Figure()
    fig.add_scatter(x=fpr_c, y=tpr_c, mode="lines", name=f"{name} (ROC-AUC {auc:.2f})",
                    line=dict(color=NORMAL_COLOR, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                    line=dict(color=MAL_COLOR, dash="dash"))
    fig.add_scatter(x=[fpr], y=[recall], mode="markers", name="Your current cut-off",
                    marker=dict(size=13, color="black"))
    fig.update_layout(title="Share of attacks caught vs share of normal records falsely flagged",
                      xaxis_title="False alarms (share of normal records wrongly flagged)",
                      yaxis_title="Attacks caught (share of real attacks)", height=420,
                      margin=dict(t=50, b=20))
    show_fig(fig)
    st.markdown(
        "Every cut-off is one point on the blue line. The red dashed line is what random guessing gives: "
        "catch 30% of attacks, falsely flag 30% of normal records. A good model's line bulges toward the "
        "**top-left corner**." + (" Here the blue line hugs the red one." if weak else "")
    )
    st.caption(f"Out-of-fold ROC-AUC = {auc:.3f} (0.5 = guessing). Average Precision = {ap:.3f} "
               f"(guessing = {attacks / n:.2f}).")


def tab_features():
    st.subheader("Which measurements matter?")
    fa = require(ART / "feature_auc.csv").sort_values("AUC (oriented)")
    st.markdown(
        "For each measurement we asked: *if I look at this one feature alone, can it tell attacks from normal "
        "traffic?* The score runs from **0.5 (useless) to 1.0 (perfect)**."
    )
    st.markdown(
        f"**A trap:** we tried {len(fa)} features, and with that many tries one will look good by luck. So we scrambled the "
        "answers at random 1,000 times and measured how high the luckiest feature gets. That is the **red line**. "
        "A feature only counts if it clearly passes it."
    )
    colors = [("#2E8B57" if v > LUCK_95 else "#9E9E9E") for v in fa["AUC (oriented)"]]
    fig = go.Figure(go.Bar(
        x=fa["AUC (oriented)"], y=fa["feature"], orientation="h", marker_color=colors,
        hovertext=[describe_feature(f) for f in fa["feature"]],
        hovertemplate="%{y}: %{x:.3f}<br>%{hovertext}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_color="black", annotation_text="useless (0.5)", annotation_position="bottom")
    fig.add_vline(x=LUCK_95, line_dash="dash", line_color=MAL_COLOR,
                  annotation_text=f"luck line ({LUCK_95:.2f})", annotation_position="top")
    fig.update_layout(title="How well each measurement alone separates attacks from normal traffic",
                      xaxis_title="Single-feature score (0.5 = useless, 1.0 = perfect)",
                      xaxis_range=[0.45, max(0.65, float(fa["AUC (oriented)"].max()) + 0.02)],
                      height=520, margin=dict(t=60, b=20))
    show_fig(fig)

    passing = int((fa["AUC (oriented)"] > LUCK_95).sum())
    if passing == 0:
        st.warning(f"**0 of {len(fa)} features** clearly beat luck (green bars would mean they do). "
                   "That is why no model, simple or fancy, could do well: they all work from the same weak information.")
    else:
        st.info(f"**{passing} of {len(fa)} features** pass the luck line (green). "
                "Even so, a single weak feature rarely gives a useful model.")
    st.caption("Hover over a bar to see what the measurement means. See notebook 04, experiment 3.")


# ----------------------------------------------------------------- page ----
st.title("Network Traffic Anomaly Detection - Learning Project")
st.warning("Educational demo. The models here perform close to random and are NOT a reliable "
           "security tool. This app exists to show what the experiments found, honestly.")
sidebar_glossary()

cmp = require(ART / "cv_model_comparison.csv")

tabs = st.tabs(["1. Start here", "2. The data", "3. The models", "4. Try the cut-off", "5. Which measurements matter?"])
with tabs[0]:
    tab_start(cmp)
with tabs[1]:
    tab_data()
with tabs[2]:
    tab_models(cmp)
with tabs[3]:
    tab_cutoff()
with tabs[4]:
    tab_features()
