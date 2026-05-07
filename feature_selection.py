#!/usr/bin/env python3
"""
Feature selection utilities for radiomics data.

Identifies top N non-redundant features for classification tasks using:
  - Correlation analysis (remove highly correlated pairs)
  - Mutual information selection
  - Permutation importance on trained classifiers
  - Consensus ranking across methods
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORGAN_FEATURES_DIR = os.path.join(BASE_DIR, "CT", "organ_features")


def load_organ_data(organ_name, organ_features_dir=DEFAULT_ORGAN_FEATURES_DIR):
    """
    Load feature data for a specific organ.

    Args:
        organ_name: Name of organ (e.g., 'heart', 'liver')
        organ_features_dir: Path to organ_features directory

    Returns:
        X (n_samples, n_features): Feature matrix (standardized)
        y (n_samples,): Target variable (sex: M=1, F=0)
        feature_names (list): Column names of features
    """
    csv_path = os.path.join(organ_features_dir, f"{organ_name}.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Organ CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Separate labels and features
    label_cols = {"subject_id", "sex", "age", "diagnosis"}
    feature_cols = [c for c in df.columns if c not in label_cols]

    # Convert sex to binary (M=1, F=0)
    y = (df["sex"] == "M").astype(int).values

    # Extract features
    X = df[feature_cols].values.astype(np.float64)

    # Drop constant/NaN features
    valid_mask = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    X = X[:, valid_mask]
    feature_cols = [f for f, v in zip(feature_cols, valid_mask) if v]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_cols


def remove_correlated_features(X, feature_names, correlation_threshold=0.95):
    """
    Remove highly correlated features, keeping one representative from each pair.
    Greedy approach: for each high-corr pair, keep the feature with higher variance.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        correlation_threshold: Absolute correlation threshold (default: 0.95)

    Returns:
        keep_mask (bool array): Which features to keep
        removed_pairs (list): Removed (feature1, feature2, correlation) tuples
    """
    n_features = X.shape[1]
    keep_mask = np.ones(n_features, dtype=bool)
    removed_pairs = []

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Identify and remove high-correlation pairs
    for i in range(n_features):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n_features):
            if not keep_mask[j]:
                continue

            corr = abs(corr_matrix[i, j])
            if corr > correlation_threshold:
                # Remove feature with lower variance
                var_i = X[:, i].var()
                var_j = X[:, j].var()
                if var_i > var_j:
                    keep_mask[j] = False
                    removed_pairs.append((feature_names[i], feature_names[j], corr))
                else:
                    keep_mask[i] = False
                    removed_pairs.append((feature_names[j], feature_names[i], corr))

    return keep_mask, removed_pairs


def select_features_mutual_information(X, y, feature_names, k=20):
    """
    Select top K features using mutual information.

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        k: Number of features to select

    Returns:
        selected_features (list): Top K feature names
        scores (dict): Feature name → mutual information score
    """
    # Compute mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Rank by score
    top_indices = np.argsort(mi_scores)[::-1][:k]
    selected = [feature_names[i] for i in top_indices]

    scores = {feature_names[i]: mi_scores[i] for i in top_indices}

    return selected, scores


def select_features_permutation_importance(X, y, feature_names, k=20):
    """
    Select top K features using permutation importance from a Random Forest.

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        k: Number of features to select

    Returns:
        selected_features (list): Top K feature names
        scores (dict): Feature name → permutation importance score
    """
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # Compute permutation importance
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_scores = result.importances_mean

    # Rank by score
    top_indices = np.argsort(perm_scores)[::-1][:k]
    selected = [feature_names[i] for i in top_indices]

    scores = {feature_names[i]: perm_scores[i] for i in top_indices}

    return selected, scores


def consensus_feature_selection(X, y, feature_names, k=20, correlation_threshold=0.95):
    """
    Combine multiple feature selection methods for robust selection:
      1. Remove highly correlated features
      2. Mutual information ranking
      3. Permutation importance ranking
      4. Consensus: features that rank high in both methods

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        k: Target number of features to select
        correlation_threshold: Correlation threshold for removing redundancy

    Returns:
        top_features (list): Selected top K features
        importance_scores (dict): Feature → average ranking score
        removed_pairs (list): Removed correlated pairs
    """
    # Step 1: Remove highly correlated features
    keep_mask, removed_pairs = remove_correlated_features(
        X, feature_names, correlation_threshold
    )
    X_reduced = X[:, keep_mask]
    features_reduced = [f for f, m in zip(feature_names, keep_mask) if m]
    n_removed = (~keep_mask).sum()
    print(f"  Removed {n_removed} highly correlated features (r > {correlation_threshold})")

    # Step 2: Mutual information selection
    mi_selected, mi_scores = select_features_mutual_information(
        X_reduced, y, features_reduced, k=k
    )
    print(f"  Mutual Information: top {len(mi_selected)} features identified")

    # Step 3: Permutation importance selection
    perm_selected, perm_scores = select_features_permutation_importance(
        X_reduced, y, features_reduced, k=k
    )
    print(f"  Permutation Importance: top {len(perm_selected)} features identified")

    # Step 4: Consensus ranking
    # Create ranking dictionaries
    mi_ranking = {f: (k - i) for i, f in enumerate(mi_selected)}  # Higher rank = higher k
    perm_ranking = {f: (k - i) for i, f in enumerate(perm_selected)}

    # Combine rankings
    all_selected = set(mi_selected) | set(perm_selected)
    combined_scores = {}
    for feat in all_selected:
        mi_rank = mi_ranking.get(feat, 0)
        perm_rank = perm_ranking.get(feat, 0)
        combined_scores[feat] = (mi_rank + perm_rank) / 2

    # Sort by combined score and select top K
    top_features = sorted(combined_scores.keys(), key=lambda f: combined_scores[f], reverse=True)[:k]

    return top_features, combined_scores, removed_pairs


def report_feature_selection(organ_name, top_features, importance_scores, removed_pairs, k=20):
    """
    Print a formatted report of feature selection results.

    Args:
        organ_name: Name of organ
        top_features: List of selected features
        importance_scores: Dict of feature → score
        removed_pairs: List of (feat1, feat2, correlation) tuples
        k: Expected number of features
    """
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION REPORT: {organ_name.upper()}")
    print(f"{'='*70}")

    print(f"\nTop {len(top_features)} Selected Features:")
    print("-" * 70)
    for i, feat in enumerate(top_features, 1):
        score = importance_scores.get(feat, 0)
        # Shorten feature name for readability
        feat_short = feat.replace("original_", "").replace("log-sigma-", "LoG_").replace("wavelet-", "Wv_")
        print(f"  {i:2d}. {feat_short:<50s} (score: {score:.3f})")

    if removed_pairs:
        print(f"\nRemoved High-Correlation Pairs ({len(removed_pairs)} total):")
        print("-" * 70)
        for feat1, feat2, corr in removed_pairs[:5]:  # Show first 5
            feat1_short = feat1.replace("original_", "")
            feat2_short = feat2.replace("original_", "")
            print(f"  {feat1_short} <→> {feat2_short}  (r = {corr:.3f})")
        if len(removed_pairs) > 5:
            print(f"  ... and {len(removed_pairs) - 5} more")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Quick test on heart organ
    print("Testing feature selection on heart organ...")
    X, y, feature_names = load_organ_data("heart")
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    top_feats, scores, removed = consensus_feature_selection(X, y, feature_names, k=20)
    report_feature_selection("heart", top_feats, scores, removed, k=20)
