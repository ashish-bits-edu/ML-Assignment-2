"""
2025ac05119 - Ashish Takawale
Streamlit app for ML Assignment 2 - Obesity Level Classification.

Lets a user upload a test CSV, pick one of the 5 trained models (or compare
all of them at once), and view the resulting metrics, confusion matrix and
classification report. The models themselves are trained separately by
model/train_models.py and just loaded here.

Run with: streamlit run app.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Obesity Level Classifier",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_artifacts():
    """Load the saved pipelines once per session instead of on every rerun.

    Streamlit reruns this whole script on every widget interaction, and
    the Random Forest pickle alone is a few MB, so caching this keeps the
    app responsive instead of re-reading every .pkl from disk each time a
    dropdown changes.
    """
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_info = json.load(f)
    metrics_train = pd.read_csv(os.path.join(MODEL_DIR, "metrics.csv"), index_col=0)
    return models, label_encoder, feature_info, metrics_train


models, label_encoder, feature_info, metrics_train = load_artifacts()
FEATURE_COLS = feature_info["all_features"]
TARGET_COL = feature_info["target"]

st.title("Obesity Level Classification")
st.caption(
    "M.Tech ML Assignment 2 - Multi-class classification on the UCI "
    "*Estimation of Obesity Levels* dataset, comparing 5 ML models."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"], help="Upload test_data.csv or similar."
)

model_choice = st.sidebar.selectbox(
    "Select model", list(models.keys()) + ["Compare all models"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "Dataset: UCI Obesity Levels (2,111 rows, 16 features, 7 classes)\n\n"
    "Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, "
    "Random Forest"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def compute_metrics(y_true_enc, y_pred_enc, y_proba, n_classes):
    """Same 6 metrics as the training script, computed here on whatever
    data the user just uploaded rather than the fixed train/test split."""
    labels = list(range(n_classes))
    auc_kwargs = dict(multi_class="ovr", average="macro", labels=labels)
    try:
        auc = roc_auc_score(y_true_enc, y_proba, **auc_kwargs)
    except ValueError:
        # Happens if the uploaded CSV only contains a subset of the 7
        # classes - AUC needs at least two classes present to be defined.
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true_enc, y_pred_enc),
        "AUC": auc,
        "Precision": precision_score(
            y_true_enc, y_pred_enc, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true_enc, y_pred_enc, average="macro", zero_division=0
        ),
        "F1": f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred_enc),
    }


def run_model(name, df, has_target):
    """Predict with the chosen pipeline, and evaluate against the target
    column if the uploaded CSV happens to include it (test_data.csv does,
    a plain feature-only CSV wouldn't)."""
    pipeline = models[name]
    X = df[FEATURE_COLS]
    y_pred_enc = pipeline.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    result = {"pred_labels": y_pred_labels}

    if has_target:
        y_proba = pipeline.predict_proba(X)
        y_true_enc = label_encoder.transform(df[TARGET_COL])
        result["metrics"] = compute_metrics(
            y_true_enc, y_pred_enc, y_proba, len(label_encoder.classes_)
        )
        result["y_true_enc"] = y_true_enc
        result["y_pred_enc"] = y_pred_enc

    return result


def plot_confusion_matrix(y_true_enc, y_pred_enc, class_names):
    cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Class names are long (e.g. "Overweight_Level_II"), so rotate the
    # x-axis labels or they overlap and become unreadable.
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.info(
        "Upload a CSV (e.g. `test_data.csv` from the repo) using the sidebar "
        "to get started. The file should contain the 16 feature columns; "
        "including the `NObeyesdad` target column enables full metric "
        "evaluation, confusion matrix and classification report."
    )
    st.subheader("Expected columns")
    st.code(", ".join(FEATURE_COLS + [TARGET_COL]))
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded CSV: {e}")
    st.stop()

missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

has_target = TARGET_COL in df.columns

st.subheader("Uploaded data preview")
st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

if has_target:
    # Quick sanity-check plot before diving into model results: how are
    # the 7 classes actually distributed in whatever got uploaded. Useful
    # since test_data.csv is a random 20% split and worth confirming it
    # isn't skewed towards one class before trusting the metrics below.
    st.markdown("#### Class distribution in uploaded data")
    class_counts = df[TARGET_COL].value_counts().reindex(
        label_encoder.classes_, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 3.5))
    class_counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

if model_choice != "Compare all models":
    # Single-model view: metrics, confusion matrix, classification report,
    # and the raw predictions.
    st.subheader(f"Results - {model_choice}")
    result = run_model(model_choice, df, has_target)

    if has_target:
        m = result["metrics"]
        cols = st.columns(6)
        for col, (k, v) in zip(cols, m.items()):
            col.metric(k, f"{v:.4f}")

        st.markdown("#### Confusion Matrix")
        fig = plot_confusion_matrix(
            result["y_true_enc"], result["y_pred_enc"], label_encoder.classes_
        )
        st.pyplot(fig)

        st.markdown("#### Classification Report")
        report = classification_report(
            result["y_true_enc"],
            result["y_pred_enc"],
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    else:
        st.warning(
            f"No `{TARGET_COL}` column found - showing predictions only "
            "(metrics need ground-truth labels)."
        )

    st.markdown("#### Predictions")
    pred_df = df.copy()
    pred_df["Predicted_" + TARGET_COL] = result["pred_labels"]
    st.dataframe(pred_df, use_container_width=True)

else:
    # "Compare all models" view: run every pipeline on the same uploaded
    # data and lay the results out side by side.
    st.subheader("Comparison across all 5 models")
    if not has_target:
        st.warning(
            f"No `{TARGET_COL}` column found in the uploaded CSV - metric "
            "comparison requires ground-truth labels."
        )
        st.stop()

    rows = {}
    all_results = {}
    for name in models:
        r = run_model(name, df, has_target=True)
        rows[name] = r["metrics"]
        all_results[name] = r

    comp_df = pd.DataFrame(rows).T.round(4)
    comp_df.index.name = "ML Model Name"
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("#### Metric comparison chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    best_model = comp_df["F1"].idxmax()
    st.success(f"Best model on this data by F1 score: {best_model}")

    st.markdown("#### Confusion matrix per model")
    tabs = st.tabs(list(models.keys()))
    for tab, name in zip(tabs, models.keys()):
        with tab:
            r = all_results[name]
            fig = plot_confusion_matrix(
                r["y_true_enc"], r["y_pred_enc"], label_encoder.classes_
            )
            st.pyplot(fig)

st.markdown("---")
with st.expander("Reference: metrics computed during training (full test split)"):
    st.dataframe(metrics_train, use_container_width=True)
