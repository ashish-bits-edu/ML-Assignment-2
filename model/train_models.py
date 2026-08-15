"""
Training script for ML Assignment 2 - Obesity Level Classification.

Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical
Condition (UCI ML Repository, id=544), 7-class target.

Loads the raw dataset, does a stratified 80/20 train/test split (the raw
test split is written out to ../test_data.csv for the Streamlit app), fits
a shared preprocessing pipeline (one-hot encoding for the categorical
columns, standard scaling for the numeric ones) around each of the 5
classifiers, and evaluates all of them on the held-out split. Fitted
pipelines, the label encoder and a metrics.csv summary get saved under
model/ for the app and the README comparison table to use.

Run with: python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Fixed seed everywhere so the split and the models are reproducible between
# runs (and so the numbers in the README match what this script actually
# produces).
RANDOM_STATE = 42

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = HERE

# Column groups, used by the ColumnTransformer below. Kept as module-level
# constants (instead of inferring dtypes automatically) because a couple of
# the numeric-looking columns in this dataset (FCVC, NCP, CH2O, FAF, TUE)
# are actually ordinal survey responses, not raw measurements - scaling
# them still makes sense for kNN/Logistic Regression, so they're grouped
# with the numeric features.
CATEGORICAL_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
TARGET_COL = "NObeyesdad"


def load_data():
    """Read the raw UCI CSVs and merge features + target into one frame."""
    X = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "targets.csv"))
    df = pd.concat([X, y], axis=1)

    # A part of this dataset was generated with SMOTE by the original
    # authors, which leaves a handful of exact duplicate rows behind.
    # Dropping them avoids the same row ending up in both the train and
    # test split after train_test_split.
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor():
    """Shared preprocessing used by every model in the pipeline below.

    StandardScaler on the numeric columns matters most for kNN (distance
    based) and Logistic Regression (gradient based); it doesn't hurt the
    tree-based models either. OneHotEncoder handles the categorical
    survey answers, with unknown categories ignored at inference time so
    a slightly different CSV uploaded to the Streamlit app doesn't crash
    the app.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def get_models():
    """The 5 classifiers required by the assignment.

    max_depth is capped for the Decision Tree and Random Forest to keep
    them from just memorising the training split (Height/Weight alone are
    close to sufficient to determine the label, so an unconstrained tree
    overfits almost immediately). n_neighbors=9 for kNN and n_estimators
    for the forest were picked after a couple of manual trials, not an
    exhaustive grid search - good enough for comparing the 5 model
    families against each other. GaussianNB has no hyperparameters to
    tune.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE
        ),
    }


def evaluate(pipeline, X_test, y_test_enc, n_classes):
    """Compute the 6 metrics the assignment asks for, on one fitted model.

    Precision/Recall/F1 use macro averaging - the 7 classes are fairly
    balanced here (see README), so macro (unweighted mean over classes)
    is a fair summary rather than being dominated by the larger classes.
    AUC is one-vs-rest since sklearn doesn't expose a single native
    multi-class AUC.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test_enc, y_pred),
        "AUC": roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test_enc, y_pred),
    }
    return metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y_raw = df[TARGET_COL]

    # Encode the target to integers once, up front, so every model and
    # every metric function below works with the same label ordering.
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    # stratify=y_enc keeps the class proportions in the test split close
    # to the full dataset (272-351 rows per class), which matters more
    # than usual here since a couple of classes are already on the
    # smaller side.
    X_train, X_test, y_train_enc, y_test_enc, y_train_raw, y_test_raw = train_test_split(
        X, y_enc, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # Write out the raw (unprocessed) test split as test_data.csv - this
    # is the file required by the assignment and the one the Streamlit
    # app expects to be uploaded, so preprocessing happens identically
    # for train and "new" data via the saved pipelines below.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test_raw.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    results = {}
    for name, clf in get_models().items():
        # Preprocessing + classifier bundled into one Pipeline object so
        # the app only has to load a single .pkl per model and call
        # .predict()/.predict_proba() directly on raw feature columns.
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", clf)]
        )
        pipeline.fit(X_train, y_train_enc)

        metrics = evaluate(pipeline, X_test, y_test_enc, n_classes)
        results[name] = metrics

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{fname}.pkl"))
        print(f"{name:30s} -> {metrics}")

    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Small metadata file so app.py doesn't need to hardcode the column
    # lists a second time.
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(
            {
                "numeric": NUMERIC_COLS,
                "categorical": CATEGORICAL_COLS,
                "target": TARGET_COL,
                "all_features": NUMERIC_COLS + CATEGORICAL_COLS,
            },
            f,
            indent=2,
        )

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "ML Model Name"
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"))
    print("\nSaved metrics.csv:\n", metrics_df)


if __name__ == "__main__":
    main()
