#!/usr/bin/env python3
"""
PCA + Wilcoxon Rank-Sum Analysis for Gender Differences in Radiomics Features

Loads the merged radiomics features CSV produced by batch_extract.py,
runs PCA on the standardised features, then applies the Wilcoxon rank-sum
test to each principal component to determine whether male and female
CT scans show statistically significant differences.

Usage:
    python analyze_gender.py                          # defaults
    python analyze_gender.py --input path/to/csv      # custom input
    python analyze_gender.py --n-components 10        # fixed PC count
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ranksums

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "CT", "radiomics_features.csv")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def run_analysis(input_csv, n_components=0.95, alpha=0.05, top_n=5):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} subjects from {input_csv}")

    # ---- separate labels and features ----
    label_cols = ["subject_id", "sex", "age", "diagnosis"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values.astype(np.float64)
    sex = df["sex"].values  # "M" / "F"

    # Drop any features that are constant or NaN
    valid_mask = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    X = X[:, valid_mask]
    kept_features = [f for f, v in zip(feature_cols, valid_mask) if v]
    dropped = len(feature_cols) - len(kept_features)
    if dropped:
        print(f"Dropped {dropped} constant/NaN features, {len(kept_features)} remain")

    # ---- standardise ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- PCA ----
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    n_pcs = X_pca.shape[1]
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\nPCA: {n_pcs} components explain "
          f"{cumulative[-1]*100:.1f}% of total variance\n")

    print(f"{'PC':<6} {'Var Explained':>14} {'Cumulative':>11}")
    print("-" * 33)
    for i in range(n_pcs):
        print(f"PC{i+1:<4} {explained[i]*100:>13.2f}% {cumulative[i]*100:>10.2f}%")

    # ---- Wilcoxon rank-sum test per PC ----
    male_idx = sex == "M"
    female_idx = sex == "F"
    n_male = male_idx.sum()
    n_female = female_idx.sum()
    print(f"\nGroups: Male n={n_male}, Female n={n_female}")

    print(f"\n{'PC':<6} {'M mean':>9} {'F mean':>9} {'Statistic':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 52)

    significant_pcs = []
    for i in range(n_pcs):
        scores_m = X_pca[male_idx, i]
        scores_f = X_pca[female_idx, i]
        stat, p = ranksums(scores_m, scores_f)
        sig = "*" if p < alpha else ""
        if p < alpha:
            significant_pcs.append((i + 1, p))
        print(f"PC{i+1:<4} {scores_m.mean():>9.3f} {scores_f.mean():>9.3f} "
              f"{stat:>10.3f} {p:>10.4f} {sig:>5}")

    # ---- Summary ----
    print(f"\n{'='*52}")
    if significant_pcs:
        print(f"Significant PCs (p < {alpha}):")
        for pc_num, p_val in significant_pcs:
            print(f"  PC{pc_num}: p = {p_val:.4f}")
        print(f"\nConclusion: There IS a statistically significant difference "
              f"between male and female CT scans on {len(significant_pcs)} "
              f"principal component(s).")
    else:
        print(f"No principal components showed a significant difference "
              f"(p < {alpha}) between male and female CT scans.")
        print(f"\nConclusion: No significant gender-based difference detected "
              f"in the radiomics features at alpha = {alpha}.")
    print(f"{'='*52}")

    # ---- Top feature loadings per PC ----
    # pca.components_ has shape (n_pcs, n_features). Each row is a PC,
    # each value is how much that original feature contributes to the PC.
    print(f"\n\nTop {top_n} feature loadings for PC1, PC2, PC3:")
    print("=" * 65)
    for pc_idx in range(min(3, n_pcs)):
        loadings = pca.components_[pc_idx]
        # Sort by absolute loading (strongest contributors first)
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        print(f"\nPC{pc_idx+1} ({explained[pc_idx]*100:.2f}% variance):")
        print(f"  {'Rank':<6} {'Feature':<45} {'Loading':>8}")
        print(f"  {'-'*60}")
        for rank, fi in enumerate(sorted_idx[:top_n]):
            feat_name = kept_features[fi]
            print(f"  {rank+1:<6} {feat_name:<45} {loadings[fi]:>+8.4f}")

    return X_pca, sex, pca, significant_pcs


def main():
    p = argparse.ArgumentParser(description="PCA + Wilcoxon rank-sum gender analysis")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Path to radiomics_features.csv")
    p.add_argument("--n-components", default=0.95, type=float,
                   help="Number of PCA components or variance ratio (default: 0.95)")
    p.add_argument("--alpha", default=0.05, type=float, help="Significance level")
    p.add_argument("--top-loadings", default=5, type=int,
                   help="Number of top feature loadings to show per PC (default: 5)")
    args = p.parse_args()

    run_analysis(args.input, n_components=args.n_components, alpha=args.alpha,
                 top_n=args.top_loadings)


if __name__ == "__main__":
    main()
