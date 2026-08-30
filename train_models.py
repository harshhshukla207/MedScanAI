#!/usr/bin/env python3
"""
train_models.py — Clinical Lab Report Analysis: XGBoost Model Training

Trains 6 individual XGBClassifier models for:
  1. Diabetes
  2. Anemia
  3. CKD (Chronic Kidney Disease)
  4. Cardiovascular Risk
  5. Liver Disease
  6. Thyroid Disorder

Reads comprehensive_clinical_data.csv, reclassifies binary targets into
3 severity classes (0=Normal, 1=Mild, 2=Chronic) using clinical thresholds
from parameters.csv, handles class imbalance via sample_weight, and saves
each model to saved_models/ as .joblib.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "comprehensive_clinical_data.csv"
MODEL_DIR = Path(__file__).parent / "saved_models"

# Each entry: (model_filename, target_column, display_name)
DISEASE_MODELS = [
    ("diabetes_model.joblib",   "Target_Diabetes",  "Diabetes"),
    ("anemia_model.joblib",     "Target_Anemia",    "Anemia"),
    ("ckd_model.joblib",        "Target_CKD",       "CKD"),
    ("cardio_model.joblib",     "Target_Cardio",    "Cardiovascular Risk"),
    ("liver_model.joblib",      "Target_Liver",     "Liver Disease"),
    ("thyroid_model.joblib",    "Target_Thyroid",    "Thyroid Disorder"),
]

# All target column names (to exclude from features)
TARGET_COLS = [m[1] for m in DISEASE_MODELS]

# Severity class labels
SEVERITY_LABELS = ["Normal", "Mild", "Chronic"]


# ---------------------------------------------------------------------------
# Target Reclassification (Binary → 3-Class)
# ---------------------------------------------------------------------------
def reclassify_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reclassify binary targets into 3 severity classes using clinical
    thresholds from each disease's primary diagnostic indicator.

    Classes: 0 = Normal, 1 = Mild, 2 = Chronic
    Thresholds sourced from parameters.csv label definitions.
    """
    df = df.copy()

    # --- Diabetes (HbA1c, HIGH_BAD) ---
    # Normal: HbA1c < 5.7 | Mild: 5.7–7.99 | Chronic: >= 8.0
    df["Target_Diabetes"] = np.select(
        [df["HbA1c"] < 5.7,
         (df["HbA1c"] >= 5.7) & (df["HbA1c"] < 8.0),
         df["HbA1c"] >= 8.0],
        [0, 1, 2], default=0
    )

    # --- Anemia (Hemoglobin, LOW_BAD) ---
    # Normal: Hgb >= 12.0 | Mild: 7.0–11.99 | Chronic: < 7.0
    df["Target_Anemia"] = np.select(
        [df["Hemoglobin"] >= 12.0,
         (df["Hemoglobin"] >= 7.0) & (df["Hemoglobin"] < 12.0),
         df["Hemoglobin"] < 7.0],
        [0, 1, 2], default=0
    )

    # --- CKD (eGFR, LOW_BAD) ---
    # Normal: eGFR >= 60 | Mild: 15–59.99 | Chronic: < 15
    df["Target_CKD"] = np.select(
        [df["eGFR"] >= 60,
         (df["eGFR"] >= 15) & (df["eGFR"] < 60),
         df["eGFR"] < 15],
        [0, 1, 2], default=0
    )

    # --- Cardiovascular Risk (LDL, HIGH_BAD) ---
    # Normal: LDL < 130 | Mild: 130–189.99 | Chronic: >= 190
    df["Target_Cardio"] = np.select(
        [df["LDL"] < 130,
         (df["LDL"] >= 130) & (df["LDL"] < 190),
         df["LDL"] >= 190],
        [0, 1, 2], default=0
    )

    # --- Liver Disease (ALT, HIGH_BAD) ---
    # Normal: ALT < 40 | Mild: 40–119.99 | Chronic: >= 120
    df["Target_Liver"] = np.select(
        [df["ALT"] < 40,
         (df["ALT"] >= 40) & (df["ALT"] < 120),
         df["ALT"] >= 120],
        [0, 1, 2], default=0
    )

    # --- Thyroid Disorder (TSH, BOTH directions) ---
    # Normal: 0.4–4.0 | Mild: 0.1–0.39 or 4.01–10.0 | Chronic: <0.1 or >10.0
    df["Target_Thyroid"] = np.select(
        [(df["TSH"] >= 0.4) & (df["TSH"] <= 4.0),
         ((df["TSH"] >= 0.1) & (df["TSH"] < 0.4)) |
         ((df["TSH"] > 4.0) & (df["TSH"] <= 10.0)),
         (df["TSH"] < 0.1) | (df["TSH"] > 10.0)],
        [0, 1, 2], default=0
    )

    return df


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV, encode Gender, reclassify targets, return (df, feature_names)."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")

    # Encode Gender: M=1, F=0
    df["Gender"] = df["Gender"].map({"M": 1, "F": 0}).astype(int)

    # Reclassify binary targets → 3-class severity
    print("\n  Reclassifying targets: Binary -> 3-class (Normal/Mild/Chronic)")
    df = reclassify_targets(df)

    # Feature columns = everything except targets
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Targets: {TARGET_COLS}")

    return df, feature_cols


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------
def train_single_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    display_name: str,
) -> XGBClassifier:
    """Train one XGBClassifier for 3-class severity prediction."""
    print(f"\n{'='*60}")
    print(f"  Training: {display_name}  (target: {target_col})")
    print(f"{'='*60}")

    X = df[feature_cols]
    y = df[target_col]

    # Class distribution
    class_counts = y.value_counts().sort_index()
    print(f"  Class distribution:")
    for cls_id, count in class_counts.items():
        label = SEVERITY_LABELS[cls_id] if cls_id < len(SEVERITY_LABELS) else f"Unknown({cls_id})"
        print(f"    {label} ({cls_id}): {count}")

    # Train/test split (stratified when possible)
    min_class_count = y.value_counts().min()
    stratify_target = y if min_class_count >= 2 else None
    if stratify_target is None:
        print(f"  WARNING: Class with < 2 samples detected. Disabling stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_target
    )

    # Compute balanced sample weights for multiclass imbalance
    sample_weights = compute_sample_weight("balanced", y_train)

    # XGBClassifier — multiclass softmax with sample_weight for imbalance
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluation
    y_pred = model.predict(X_test)

    # Determine which labels are present in test set for classification_report
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [SEVERITY_LABELS[i] for i in present_labels if i < len(SEVERITY_LABELS)]
    print(classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    ))

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    df, feature_cols = load_and_preprocess(DATA_PATH)

    # Create output directory
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"\nModels will be saved to: {MODEL_DIR}")

    # Train & save each model
    for filename, target_col, display_name in DISEASE_MODELS:
        model = train_single_model(df, feature_cols, target_col, display_name)

        # Save model + metadata as a dict
        save_path = MODEL_DIR / filename
        artifact = {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "display_name": display_name,
        }
        joblib.dump(artifact, save_path)
        print(f"  [OK] Saved: {save_path}")

    # Also save the feature list separately for quick reference
    meta_path = MODEL_DIR / "feature_columns.json"
    with open(meta_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\n  [OK] Feature list saved: {meta_path}")

    print(f"\n{'='*60}")
    print(f"  All 6 models trained and saved successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
