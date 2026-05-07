#!/usr/bin/env python3
"""
Visualize PCA + Wilcoxon rank-sum results for gender classification.

Produces:
  1. Box plots of top feature loadings per PC, split by M/F,
     with individual Wilcoxon rank-sum p-values annotated.
  2. A PCA summary bar chart (variance explained per PC + significance).
  3. A combined summary table saved as CSV.

Usage:
    python visualize_gender.py
    python visualize_gender.py --input path/to/csv --top-loadings 8
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ranksums

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "CT", "radiomics_features.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")


def shorten_feature_name(name):
    """Strip the 'original_<class>_' prefix for readable axis labels."""
    parts = name.split("_", 2)  # original, class, featureName
    if len(parts) == 3:
        return f"{parts[1]}: {parts[2]}"
    return name


def run_visualizations(input_csv, n_components=0.95, alpha=0.05, top_n=5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} subjects from {input_csv}")

    # ---- separate labels and features ----
    label_cols = ["subject_id", "sex", "age", "diagnosis"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values.astype(np.float64)
    sex = df["sex"].values

    # Drop constant / NaN features
    valid_mask = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    X = X[:, valid_mask]
    kept_features = [f for f, v in zip(feature_cols, valid_mask) if v]

    male_idx = sex == "M"
    female_idx = sex == "F"

    # ---- standardise + PCA ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    n_pcs = X_pca.shape[1]
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # ---- Wilcoxon on PCs ----
    pc_results = []
    for i in range(n_pcs):
        scores_m = X_pca[male_idx, i]
        scores_f = X_pca[female_idx, i]
        stat, p = ranksums(scores_m, scores_f)
        pc_results.append({
            "PC": f"PC{i+1}",
            "Variance Explained (%)": round(explained[i] * 100, 2),
            "Cumulative (%)": round(cumulative[i] * 100, 2),
            "Male Mean": round(scores_m.mean(), 3),
            "Female Mean": round(scores_f.mean(), 3),
            "Wilcoxon Stat": round(stat, 3),
            "p-value": round(p, 6),
            "Significant": "Yes" if p < alpha else "No",
        })

    pc_df = pd.DataFrame(pc_results)
    pc_csv_path = os.path.join(OUTPUT_DIR, "pc_wilcoxon_summary.csv")
    pc_df.to_csv(pc_csv_path, index=False)
    print(f"Saved PC summary table -> {pc_csv_path}")

    # ==================================================================
    # FIGURE 1: PCA variance explained + significance markers
    # ==================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if r["Significant"] == "Yes" else "#3498db"
              for r in pc_results]
    bars = ax1.bar(range(n_pcs), explained * 100, color=colors, edgecolor="white")
    ax1.plot(range(n_pcs), cumulative * 100, "ko-", markersize=5, label="Cumulative")
    ax1.set_xticks(range(n_pcs))
    ax1.set_xticklabels([f"PC{i+1}" for i in range(n_pcs)])
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("PCA Variance Explained by Component\n(Red = significant M/F difference, p < 0.05)")
    ax1.legend(loc="center right")

    # Annotate p-values on bars
    for i, r in enumerate(pc_results):
        p_val = r["p-value"]
        label = f"p={p_val:.4f}" if p_val >= 0.0001 else f"p={p_val:.1e}"
        ax1.text(i, explained[i] * 100 + 0.5, label,
                 ha="center", va="bottom", fontsize=7, rotation=45)

    fig1.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "pca_variance_significance.png")
    fig1.savefig(fig1_path, dpi=150)
    print(f"Saved -> {fig1_path}")
    plt.close(fig1)

    # ==================================================================
    # FIGURE 2: Top feature loadings per PC with individual Wilcoxon tests
    # ==================================================================
    # For each of the first 3 PCs (or fewer), get top_n features by loading
    n_panels = min(3, n_pcs)

    # Collect all unique top features across PCs for the individual tests
    all_top_features = []
    pc_top_info = []  # list of (pc_idx, [(feat_idx, feat_name, loading), ...])

    for pc_idx in range(n_panels):
        loadings = pca.components_[pc_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1][:top_n]
        features_info = []
        for fi in sorted_idx:
            feat_name = kept_features[fi]
            features_info.append((fi, feat_name, loadings[fi]))
            if feat_name not in all_top_features:
                all_top_features.append(feat_name)
        pc_top_info.append((pc_idx, features_info))

    # Run individual Wilcoxon on each unique top feature (raw, unstandardized)
    individual_results = []
    for feat_name in all_top_features:
        fi = kept_features.index(feat_name)
        vals_m = X[male_idx, fi]
        vals_f = X[female_idx, fi]
        stat, p = ranksums(vals_m, vals_f)
        individual_results.append({
            "Feature": feat_name,
            "Short Name": shorten_feature_name(feat_name),
            "Male Mean": round(vals_m.mean(), 4),
            "Male Std": round(vals_m.std(), 4),
            "Female Mean": round(vals_f.mean(), 4),
            "Female Std": round(vals_f.std(), 4),
            "Wilcoxon Stat": round(stat, 3),
            "p-value": round(p, 6),
            "Significant": "Yes" if p < alpha else "No",
        })

    indiv_df = pd.DataFrame(individual_results)
    indiv_csv_path = os.path.join(OUTPUT_DIR, "individual_feature_wilcoxon.csv")
    indiv_df.to_csv(indiv_csv_path, index=False)
    print(f"Saved individual feature tests -> {indiv_csv_path}")

    # Build a quick lookup: feat_name -> p-value
    feat_pval = {r["Feature"]: r["p-value"] for r in individual_results}

    # Create box plot panels — one row per PC
    fig2, axes = plt.subplots(n_panels, 1, figsize=(12, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (pc_idx, features_info) in zip(axes, pc_top_info):
        feat_names_short = []
        data_m = []
        data_f = []
        p_values = []

        for fi, feat_name, loading in features_info:
            feat_names_short.append(shorten_feature_name(feat_name))
            data_m.append(X_scaled[male_idx, fi])
            data_f.append(X_scaled[female_idx, fi])
            p_values.append(feat_pval[feat_name])

        n_feats = len(feat_names_short)
        positions_m = np.arange(n_feats) * 3
        positions_f = positions_m + 1

        bp_m = ax.boxplot(data_m, positions=positions_m, widths=0.8,
                          patch_artist=True, showfliers=False)
        bp_f = ax.boxplot(data_f, positions=positions_f, widths=0.8,
                          patch_artist=True, showfliers=False)

        for patch in bp_m["boxes"]:
            patch.set_facecolor("#5dade2")
            patch.set_alpha(0.8)
        for patch in bp_f["boxes"]:
            patch.set_facecolor("#f1948a")
            patch.set_alpha(0.8)

        # Annotate p-values above each pair
        for j in range(n_feats):
            mid_x = (positions_m[j] + positions_f[j]) / 2
            # Find the max whisker height for positioning
            all_vals = np.concatenate([data_m[j], data_f[j]])
            y_max = np.percentile(all_vals, 95) + 0.3
            p = p_values[j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            p_text = f"p={p:.4f}\n{stars}" if p >= 0.0001 else f"p={p:.1e}\n{stars}"
            ax.text(mid_x, y_max, p_text, ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color="#e74c3c" if p < alpha else "#7f8c8d")

        ax.set_xticks((positions_m + positions_f) / 2)
        ax.set_xticklabels(feat_names_short, rotation=30, ha="right", fontsize=9)

        var_pct = explained[pc_idx] * 100
        pc_p = pc_results[pc_idx]["p-value"]
        sig_label = f"p={pc_p:.4f}" if pc_p >= 0.0001 else f"p={pc_p:.1e}"
        ax.set_title(f"PC{pc_idx+1} Top {n_feats} Features — "
                     f"{var_pct:.1f}% variance — PC Wilcoxon {sig_label}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Standardized Value")

        # Legend
        ax.plot([], [], "s", color="#5dade2", markersize=10, label="Male")
        ax.plot([], [], "s", color="#f1948a", markersize=10, label="Female")
        ax.legend(loc="upper right", fontsize=9)

    fig2.suptitle("Top Feature Loadings per Principal Component\n"
                  "Male vs Female (Standardized) with Individual Wilcoxon p-values",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "feature_boxplots_by_pc.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fig2_path}")
    plt.close(fig2)

    # ==================================================================
    # FIGURE 3: Heatmap-style summary of individual feature significance
    # ==================================================================
    fig3, ax3 = plt.subplots(figsize=(10, max(4, len(all_top_features) * 0.5)))
    sorted_indiv = sorted(individual_results, key=lambda r: r["p-value"])
    feat_labels = [r["Short Name"] for r in sorted_indiv]
    p_vals = [r["p-value"] for r in sorted_indiv]
    m_means = [r["Male Mean"] for r in sorted_indiv]
    f_means = [r["Female Mean"] for r in sorted_indiv]

    y_pos = np.arange(len(feat_labels))
    bar_colors = ["#e74c3c" if p < alpha else "#95a5a6" for p in p_vals]

    ax3.barh(y_pos, [-np.log10(p) for p in p_vals], color=bar_colors, edgecolor="white")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feat_labels, fontsize=9)
    ax3.set_xlabel("-log10(p-value)")
    ax3.set_title("Individual Feature Wilcoxon Rank-Sum Tests (M vs F)\n"
                  "Red = significant at p < 0.05")

    # Draw significance threshold line
    ax3.axvline(-np.log10(alpha), color="black", linestyle="--", linewidth=1,
                label=f"p = {alpha}")
    ax3.legend(loc="lower right")

    # Annotate p-values
    for i, (p, ml, fl) in enumerate(zip(p_vals, m_means, f_means)):
        p_text = f"p={p:.4f}" if p >= 0.0001 else f"p={p:.1e}"
        ax3.text(-np.log10(p) + 0.05, i, p_text, va="center", fontsize=8)

    ax3.invert_yaxis()
    fig3.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, "individual_feature_significance.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fig3_path}")
    plt.close(fig3)

    # ---- Print summary table to console ----
    print(f"\n{'='*80}")
    print("INDIVIDUAL FEATURE WILCOXON RANK-SUM TESTS")
    print(f"{'='*80}")
    print(f"{'Feature':<50} {'M mean':>10} {'F mean':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 87)
    for r in sorted_indiv:
        sig = "*" if r["Significant"] == "Yes" else ""
        print(f"{r['Short Name']:<50} {r['Male Mean']:>10.4f} {r['Female Mean']:>10.4f} "
              f"{r['p-value']:>10.6f} {sig:>5}")
    print(f"{'='*80}")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Visualize PCA + Wilcoxon gender analysis")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Path to radiomics_features.csv")
    p.add_argument("--n-components", default=0.95, type=float,
                   help="PCA components or variance ratio (default: 0.95)")
    p.add_argument("--alpha", default=0.05, type=float, help="Significance level")
    p.add_argument("--top-loadings", default=5, type=int,
                   help="Number of top feature loadings per PC (default: 5)")
    args = p.parse_args()

    run_visualizations(args.input, n_components=args.n_components,
                       alpha=args.alpha, top_n=args.top_loadings)


if __name__ == "__main__":
    main()
