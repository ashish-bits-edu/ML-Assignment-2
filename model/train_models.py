"""
Training script for ML Assignment 2.

Dataset : Estimation of Obesity Levels Based on Eating Habits and Physical
          Condition (UCI ML Repository, id=544)
Task    : Multi-class classification (7 obesity level classes)

This script:
  1. Loads the raw dataset (data/features.csv + data/targets.csv).
  2. Performs a stratified 80/20 train/test split.
  3. Saves the RAW (unprocessed) test split to ../test_data.csv - this is
     the file uploaded to the Streamlit app for evaluation.
  4. Builds a shared preprocessing pipeline (one-hot encode categoricals,
     scale numeric columns) wrapped around each of the 5 classifiers.
  5. Trains all 5 models and computes Accuracy, AUC, Precision, Recall,
     F1 and MCC on the held-out test split.
  6. Saves each fitted pipeline (model/*.pkl), the target label encoder,
     and a metrics.csv summary used to build the README comparison table.
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

RANDOM_STATE = 42

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = HERE

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
    X = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "targets.csv"))
    df = pd.concat([X, y], axis=1)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )


def get_models():
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
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test_enc, y_pred),
        "AUC": roc_auc_score(
            y_test_enc, y_proba, multi_class="ovr", average="macro"
        ),
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

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    X_train, X_test, y_train_enc, y_test_enc, y_train_raw, y_test_raw = train_test_split(
        X, y_enc, y_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # Save RAW (unprocessed) test split for the Streamlit app / grading.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test_raw.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    results = {}
    for name, clf in get_models().items():
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
