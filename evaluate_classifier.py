#!/usr/bin/env python3
"""
Evaluate trained gender classification models.

Provides comprehensive evaluation including:
  - Cross-validation results summary
  - Feature importance analysis (which features drive predictions)
  - Model comparison (RF vs MLP)
  - Inference on new data with confidence scores
  - Confusion matrices and detailed metrics

Usage:
    python evaluate_classifier.py
    python evaluate_classifier.py --organ heart --feature-importance
    python evaluate_classifier.py --organ global --compare-models
    python evaluate_classifier.py --organ gluteus_maximus --top-k 15
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
)

import config


def load_model(model_type, organ_group):
    """Load a trained model from disk."""
    model_path = config.get_model_path(model_type, organ_group)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_data_for_evaluation(organ_group="global"):
    """Load feature data for evaluation (same as training pipeline)."""
    # This mirrors train_gender_classifier.py logic
    features_df = pd.read_csv(config.SELECTED_FEATURES_CSV)
    labels_df = pd.read_csv(config.LABELS_CSV)

    if "Subject ID" in labels_df.columns:
        labels_df.rename(columns={"Subject ID": "subject_id"}, inplace=True)

    sex_map = dict(zip(labels_df["subject_id"], labels_df["sex"]))

    if organ_group == "global":
        organs_to_use = list(features_df["organ"].unique())
        # Exclude organs with incomplete data
        exclude_organs = config.ORGAN_GROUPS.get("_exclude_organs", [])
        organs_to_use = [o for o in organs_to_use if o not in exclude_organs]
    else:
        organs_to_use = config.ORGAN_GROUPS.get(organ_group, [organ_group])

    X_list = []
    feature_names_list = []
    subject_ids = None

    for organ in organs_to_use:
        organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")

        if not os.path.isfile(organ_csv):
            continue

        organ_df = pd.read_csv(organ_csv)
        if subject_ids is None:
            subject_ids = organ_df["subject_id"].values

        organ_features_df = features_df[features_df["organ"] == organ].copy()
        organ_features_df = organ_features_df.sort_values("feature_rank")

        selected_features = organ_features_df["feature_name"].values
        feature_names_list.extend([f"{organ}__{feat}" for feat in selected_features])
        organ_data = organ_df[selected_features].values.astype(np.float64)
        X_list.append(organ_data)

    # Handle sample count mismatches (same logic as training)
    n_samples_per_organ = [X.shape[0] for X in X_list]
    if len(set(n_samples_per_organ)) > 1:
        subjects_by_organ = []
        X_list_aligned = []

        for organ in organs_to_use:
            organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")
            organ_df = pd.read_csv(organ_csv)
            subjects_by_organ.append(set(organ_df["subject_id"].values))

        common_subjects = set.intersection(*subjects_by_organ)
        subject_ids_aligned = [sid for sid in subject_ids if sid in common_subjects]

        X_list = []
        feature_names_list = []
        for organ in organs_to_use:
            organ_csv = os.path.join(config.ORGAN_FEATURES_DIR, f"{organ}.csv")
            organ_df = pd.read_csv(organ_csv)

            organ_df = organ_df[organ_df["subject_id"].isin(common_subjects)]
            organ_df = organ_df.set_index("subject_id").loc[subject_ids_aligned].reset_index()

            organ_features_df = features_df[features_df["organ"] == organ].copy()
            organ_features_df = organ_features_df.sort_values("feature_rank")
            selected_features = organ_features_df["feature_name"].values
            feature_names_list.extend([f"{organ}__{feat}" for feat in selected_features])

            organ_data = organ_df[selected_features].values.astype(np.float64)
            X_list.append(organ_data)

        subject_ids = np.array(subject_ids_aligned)

    X = np.hstack(X_list)
    y = np.array([1 if sex_map.get(sid, "?") == "M" else 0 for sid in subject_ids])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, subject_ids, feature_names_list


def generate_evaluation_report(model, X, y, model_type, organ_group, cv_folds=config.CV_FOLDS):
    """
    Generate detailed evaluation report for a trained model.

    Returns:
        report_dict: Dictionary with evaluation metrics and insights
    """
    report_path = config.get_results_path(model_type, organ_group)
    results_csv = pd.read_csv(report_path)

    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {model_type.upper()} on {organ_group.upper()}")
    print(f"{'='*70}\n")

    print("Cross-Validation Results (from training):")
    print(results_csv.to_string(index=False))

    # Get predictions on full data
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # Overall metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\nFull-Data Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix (on full data):")
    print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
    print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")

    print(f"\n{'='*70}\n")

    return {
        "model_type": model_type,
        "organ_group": organ_group,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


def get_feature_importance(model, model_type, feature_names, top_k=20):
    """
    Extract and display feature importance from trained model.

    Args:
        model: Trained model (RF or MLP)
        model_type: "random_forest" or "mlp"
        feature_names: List of feature names
        top_k: Number of top features to return

    Returns:
        DataFrame with top K features and their importance scores
    """
    if model_type == "random_forest":
        importance = model.feature_importances_
        method = "Mean Decrease in Impurity"
    elif model_type == "mlp":
        # For MLP, use absolute weights from first hidden layer as proxy
        importance = np.abs(model.coefs_[0]).mean(axis=1)
        method = "Mean Absolute Weight (First Layer)"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create DataFrame
    importance_df = pd.DataFrame({
        "rank": range(1, len(importance) + 1),
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["percent"] = (importance_df["importance"] / importance_df["importance"].sum()) * 100
    importance_df["cumulative_percent"] = importance_df["percent"].cumsum()

    return importance_df.head(top_k), method


def compare_models(X, y, organs_groups, feature_names_dict=None):
    """
    Load and compare all trained models across organ groups.

    Args:
        X: Dictionary of feature matrices by organ group
        y: Dictionary of target labels by organ group
        organs_groups: List of organ groups to compare
        feature_names_dict: Dict of feature names by organ group

    Returns:
        comparison_df: Model comparison results
    """
    results = []

    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}\n")

    for organ_group in organs_groups:
        if organ_group not in X or organ_group not in y:
            continue

        X_data = X[organ_group]
        y_data = y[organ_group]

        for model_type in ["random_forest", "mlp"]:
            try:
                model_path = config.get_model_path(model_type, organ_group)
                if not os.path.isfile(model_path):
                    continue

                with open(model_path, "rb") as f:
                    model = pickle.load(f)

                y_pred = model.predict(X_data)
                y_pred_proba = model.predict_proba(X_data)

                metrics = {
                    "organ_group": organ_group,
                    "model": model_type,
                    "accuracy": accuracy_score(y_data, y_pred),
                    "precision": precision_score(y_data, y_pred, zero_division=0),
                    "recall": recall_score(y_data, y_pred, zero_division=0),
                    "f1": f1_score(y_data, y_pred, zero_division=0),
                }
                results.append(metrics)

            except (FileNotFoundError, Exception) as e:
                continue

    if not results:
        print("No models found for comparison.")
        return None

    comparison_df = pd.DataFrame(results)
    print(comparison_df.to_string(index=False))
    print()

    return comparison_df


def save_feature_importance(importance_df, model_type, organ_group):
    """Save feature importance results to CSV."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(
        config.RESULTS_DIR,
        f"{model_type}_{organ_group}_feature_importance.csv"
    )
    importance_df.to_csv(output_path, index=False)
    print(f"✓ Saved feature importance to {output_path}")


def save_evaluation_summary(reports):
    """Save aggregate evaluation summary to CSV."""
    summary_df = pd.DataFrame([
        {
            "model": r["model_type"],
            "organ_group": r["organ_group"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
        }
        for r in reports
    ])

    summary_path = os.path.join(config.RESULTS_DIR, "evaluation_summary.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"✓ Saved evaluation summary to {summary_path}\n")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained gender classification models")
    parser.add_argument(
        "--organ", default="global",
        choices=["global", "heart", "liver", "gluteus_maximus"],
        help="Organ group to evaluate (default: global)"
    )
    parser.add_argument(
        "--model", default="random_forest",
        choices=["random_forest", "mlp"],
        help="Model to evaluate (default: random_forest)"
    )
    parser.add_argument(
        "--feature-importance", action="store_true",
        help="Show top features driving predictions"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Show top K features (default: 20)"
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Compare all models across all organ groups"
    )
    args = parser.parse_args()

    config.print_config()

    # Load data
    print(f"\nLoading data for {args.organ}...")
    X, y, subject_ids, feature_names = load_data_for_evaluation(args.organ)
    print(f"Loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Sex distribution: {(y==1).sum()} male, {(y==0).sum()} female\n")

    # Feature importance analysis
    if args.feature_importance:
        try:
            model_path = config.get_model_path(args.model, args.organ)
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            importance_df, method = get_feature_importance(model, args.model, feature_names, top_k=args.top_k)

            print(f"\n{'='*70}")
            print(f"TOP {args.top_k} FEATURES - {args.model.upper()} ({args.organ.upper()})")
            print(f"Importance Method: {method}")
            print(f"{'='*70}\n")
            print(importance_df.to_string(index=False))
            print()

            save_feature_importance(importance_df, args.model, args.organ)

        except FileNotFoundError as e:
            print(f"✗ {e}")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Model comparison across all organ groups
    elif args.compare_models:
        organ_groups = ["global", "heart", "liver", "gluteus_maximus"]
        X_dict = {}
        y_dict = {}
        feature_names_dict = {}

        for organ in organ_groups:
            try:
                X_data, y_data, _, fn = load_data_for_evaluation(organ)
                X_dict[organ] = X_data
                y_dict[organ] = y_data
                feature_names_dict[organ] = fn
            except Exception:
                continue

        comparison_df = compare_models(X_dict, y_dict, organ_groups, feature_names_dict)
        if comparison_df is not None:
            summary_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            comparison_df.to_csv(summary_path, index=False)
            print(f"✓ Saved comparison to {summary_path}\n")

    # Default: detailed evaluation for single organ/model
    else:
        try:
            model_path = config.get_model_path(args.model, args.organ)
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            report = generate_evaluation_report(model, X, y, args.model, args.organ)
            save_evaluation_summary([report])

        except FileNotFoundError as e:
            print(f"✗ {e}")
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
