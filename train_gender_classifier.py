#!/usr/bin/env python3
"""
Train gender classification models using selected radiomics features.

Trains Random Forest and Multi-Layer Perceptron classifiers on:
  - Global model: all organs combined
  - Organ-specific models: heart, liver, gluteus groups

Uses stratified k-fold cross-validation for robust evaluation.

Usage:
    python train_gender_classifier.py
    python train_gender_classifier.py --organ heart
    python train_gender_classifier.py --models random_forest mlp
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import config

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def load_selected_features(organ_group="global"):
    """
    Load selected features for a specific organ group.

    Args:
        organ_group: "global" (all organs) or organ name

    Returns:
        X (n_samples, n_features): Feature matrix (standardized)
        y (n_samples,): Target (gender, 1=M, 0=F)
        feature_names (list): Selected feature names
        subject_ids (list): Subject IDs for tracking
    """
    # Load selected features CSV
    features_df = pd.read_csv(config.SELECTED_FEATURES_CSV)

    # Load organ feature data
    labels_df = pd.read_csv(config.LABELS_CSV)
    if "Subject ID" in labels_df.columns:
        labels_df.rename(columns={"Subject ID": "subject_id"}, inplace=True)
    sex_map = dict(zip(labels_df["subject_id"], labels_df["sex"]))

    # Determine which organs to use
    if organ_group == "global":
        organs_to_use = features_df["organ"].unique()
        # Exclude organs with incomplete data
        exclude_organs = config.ORGAN_GROUPS.get("_exclude_organs", [])
        organs_to_use = [o for o in organs_to_use if o not in exclude_organs]
    else:
        organs_to_use = config.ORGAN_GROUPS.get(organ_group, [organ_group])

    print(f"Loading features for {organ_group}...")
    print(f"  Organs: {', '.join(organs_to_use)}")

    # Load organ CSVs and combine
    X_list = []
    feature_names_list = []
    subject_ids = None

    for organ in organs_to_use:
        organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")

        if not os.path.isfile(organ_csv):
            print(f"    ⚠ Warning: {organ} CSV not found, skipping")
            continue

        # Load organ features
        organ_df = pd.read_csv(organ_csv)
        if subject_ids is None:
            subject_ids = organ_df["subject_id"].values

        # Get selected features for this organ
        organ_features_df = features_df[features_df["organ"] == organ].copy()
        organ_features_df = organ_features_df.sort_values("feature_rank")

        selected_features = organ_features_df["feature_name"].values
        feature_names_list.extend([f"{organ}__{feat}" for feat in selected_features])

        # Extract feature columns
        organ_data = organ_df[selected_features].values.astype(np.float64)
        X_list.append(organ_data)

    # If no organs were successfully loaded, raise error
    if not X_list:
        raise ValueError(f"No features loaded for organ group '{organ_group}'. Check if the organs exist in the feature selection file.")

    # Check if all organ arrays have the same number of samples
    n_samples_per_organ = [X.shape[0] for X in X_list]
    if len(set(n_samples_per_organ)) > 1:
        # Find organs with different sample counts
        print(f"\n  ⚠ Warning: Organs have different sample counts: {dict(zip(organs_to_use, n_samples_per_organ))}")
        print(f"  Using only subjects present in all organs...")

        # Get subject IDs for each organ and find intersection
        subjects_by_organ = []
        X_list_aligned = []

        for organ in organs_to_use:
            organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")
            organ_df = pd.read_csv(organ_csv)
            subjects_by_organ.append(set(organ_df["subject_id"].values))

        # Find common subjects across all organs
        common_subjects = set.intersection(*subjects_by_organ)

        # Filter subject_ids to only include common subjects, maintaining order
        subject_ids_aligned = [sid for sid in subject_ids if sid in common_subjects]

        # Re-extract features for only common subjects
        X_list = []
        for organ in organs_to_use:
            organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")
            organ_df = pd.read_csv(organ_csv)

            # Filter to only common subjects and maintain order
            organ_df = organ_df[organ_df["subject_id"].isin(common_subjects)]
            organ_df = organ_df.set_index("subject_id").loc[subject_ids_aligned].reset_index()

            # Get selected features for this organ
            organ_features_df = features_df[features_df["organ"] == organ].copy()
            organ_features_df = organ_features_df.sort_values("feature_rank")
            selected_features = organ_features_df["feature_name"].values

            # Extract feature columns
            organ_data = organ_df[selected_features].values.astype(np.float64)
            X_list.append(organ_data)

        subject_ids = np.array(subject_ids_aligned)
        print(f"  Using {len(subject_ids)} common subjects across all organs")

    # Combine all organ features
    X = np.hstack(X_list)

    # Get target (gender)
    y = np.array([1 if sex_map.get(sid, "?") == "M" else 0 for sid in subject_ids])

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"  Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Sex distribution: {(y==1).sum()} male, {(y==0).sum()} female\n")

    return X, y, feature_names_list, subject_ids


def train_model(X, y, model_type="random_forest", cv_folds=config.CV_FOLDS):
    """
    Train and evaluate a classification model using cross-validation.

    Args:
        X: Feature matrix
        y: Target variable
        model_type: "random_forest" or "mlp"
        cv_folds: Number of CV folds

    Returns:
        cv_results (dict): Cross-validation results
        models (list): Trained models (one per fold)
        scaler (StandardScaler): Fitted scaler for deployment
    """
    if model_type == "random_forest":
        model_class = RandomForestClassifier
        model_params = config.RF_PARAMS.copy()
    elif model_type == "mlp":
        model_class = MLPClassifier
        model_params = config.MLP_PARAMS.copy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type.upper()} with {cv_folds}-fold CV...")

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    # Define scoring metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
        "roc_auc": "roc_auc",
    }

    # Run cross-validation
    cv_results = cross_validate(
        model_class(**model_params),
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Train final model on all data for deployment
    final_model = model_class(**model_params)
    final_model.fit(X, y)

    # Print results
    print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} "
          f"(+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f} "
          f"(+/- {cv_results['test_precision'].std():.4f})")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f} "
          f"(+/- {cv_results['test_recall'].std():.4f})")
    print(f"  F1 Score:  {cv_results['test_f1'].mean():.4f} "
          f"(+/- {cv_results['test_f1'].std():.4f})")
    print(f"  ROC AUC:   {cv_results['test_roc_auc'].mean():.4f} "
          f"(+/- {cv_results['test_roc_auc'].std():.4f})\n")

    return cv_results, final_model


def save_results(cv_results, model_type, organ_group):
    """Save cross-validation results to CSV."""
    results_path = config.get_results_path(model_type, organ_group)

    results_summary = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        "mean": [
            cv_results["test_accuracy"].mean(),
            cv_results["test_precision"].mean(),
            cv_results["test_recall"].mean(),
            cv_results["test_f1"].mean(),
            cv_results["test_roc_auc"].mean(),
        ],
        "std": [
            cv_results["test_accuracy"].std(),
            cv_results["test_precision"].std(),
            cv_results["test_recall"].std(),
            cv_results["test_f1"].std(),
            cv_results["test_roc_auc"].std(),
        ],
    })

    results_summary.to_csv(results_path, index=False)
    print(f"✓ Saved results to {results_path}")


def save_model(model, model_type, organ_group):
    """Save trained model to pickle file."""
    if not config.SAVE_MODELS:
        return

    model_path = config.get_model_path(model_type, organ_group)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✓ Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train gender classification models")
    parser.add_argument(
        "--organ", default="global",
        choices=["global", "heart", "liver", "gluteus_maximus"],
        help="Organ group to train on (default: global)"
    )
    parser.add_argument(
        "--models", nargs="+", default=config.MODELS_TO_TRAIN,
        choices=["random_forest", "mlp"],
        help="Models to train (default: both)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=config.CV_FOLDS,
        help="Number of CV folds (default: 5)"
    )
    args = parser.parse_args()

    config.print_config()

    # Load data
    X, y, feature_names, subject_ids = load_selected_features(args.organ)

    # Train models
    for model_type in args.models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()} | ORGAN GROUP: {args.organ.upper()}")
        print(f"{'='*70}\n")

        cv_results, final_model = train_model(X, y, model_type, args.cv_folds)

        # Save results and model
        save_results(cv_results, model_type, args.organ)
        save_model(final_model, model_type, args.organ)

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
