#!/usr/bin/env python3
"""
Analyze gender significance across all 117 TotalSegmentator organs.

Loads per-organ CSVs from CT/organ_features/, runs a Wilcoxon rank-sum
test on every feature for every organ (M vs F), applies Benjamini-Hochberg
FDR correction, and produces summary tables + visualizations.

Usage:
    python analyze_all_organs.py
    python analyze_all_organs.py --input-dir CT/organ_features --labels CT/First30.csv
"""
import argparse, os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(BASE_DIR, "CT", "organ_features")
DEFAULT_LABELS = os.path.join(BASE_DIR, "CT", "First30.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")


def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    if "Subject ID" in df.columns:
        df.rename(columns={"Subject ID": "subject_id"}, inplace=True)
    return df[["subject_id", "sex"]].drop_duplicates(subset="subject_id")


def run_analysis(input_dir, labels_path, alpha=0.05):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels_df = load_labels(labels_path)
    sex_map = dict(zip(labels_df["subject_id"], labels_df["sex"]))

    organ_csvs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    print(f"Found {len(organ_csvs)} organ CSVs in {input_dir}", flush=True)

    if not organ_csvs:
        print("No organ CSVs found. Run batch_extract.py first.", flush=True)
        return

    # ---- Run Wilcoxon on every feature of every organ ----
    all_results = []
    label_cols = {"subject_id", "sex", "age", "diagnosis"}

    for csv_path in organ_csvs:
        organ = os.path.basename(csv_path).replace(".csv", "")
        df = pd.read_csv(csv_path)

        # Merge sex labels if not already present
        if "sex" not in df.columns:
            df = df.merge(labels_df, on="subject_id", how="left")

        feature_cols = [c for c in df.columns if c not in label_cols]

        if "sex" not in df.columns:
            print(f"  {organ}: no sex labels, skipping", flush=True)
            continue

        male_df = df[df["sex"] == "M"]
        female_df = df[df["sex"] == "F"]

        if len(male_df) < 2 or len(female_df) < 2:
            print(f"  {organ}: insufficient subjects (M={len(male_df)}, "
                  f"F={len(female_df)}), skipping", flush=True)
            continue

        n_features = 0
        for feat in feature_cols:
            vals_m = pd.to_numeric(male_df[feat], errors="coerce").dropna().values
            vals_f = pd.to_numeric(female_df[feat], errors="coerce").dropna().values

            if len(vals_m) < 2 or len(vals_f) < 2:
                continue

            stat, p = ranksums(vals_m, vals_f)
            all_results.append({
                "organ": organ,
                "feature": feat,
                "male_mean": np.mean(vals_m),
                "female_mean": np.mean(vals_f),
                "male_std": np.std(vals_m),
                "female_std": np.std(vals_f),
                "wilcoxon_stat": stat,
                "p_value": p,
            })
            n_features += 1

        print(f"  {organ}: {n_features} features tested "
              f"(M={len(male_df)}, F={len(female_df)})", flush=True)

    if not all_results:
        print("No tests were run. Check your data.", flush=True)
        return

    results_df = pd.DataFrame(all_results)
    print(f"\nTotal tests: {len(results_df)}", flush=True)

    # ---- Benjamini-Hochberg FDR correction ----
    reject, pvals_corrected, _, _ = multipletests(
        results_df["p_value"].values, alpha=alpha, method="fdr_bh"
    )
    results_df["p_corrected"] = pvals_corrected
    results_df["significant_raw"] = results_df["p_value"] < alpha
    results_df["significant_fdr"] = reject

    n_sig_raw = results_df["significant_raw"].sum()
    n_sig_fdr = results_df["significant_fdr"].sum()
    print(f"Significant (raw p < {alpha}): {n_sig_raw}", flush=True)
    print(f"Significant (FDR-corrected):   {n_sig_fdr}", flush=True)

    # ---- Save full results ----
    results_df = results_df.sort_values("p_value")
    full_csv = os.path.join(OUTPUT_DIR, "organ_significance_summary.csv")
    results_df.to_csv(full_csv, index=False)
    print(f"\nSaved full results -> {full_csv}", flush=True)

    # ---- Organ-level summary: count significant features per organ ----
    organ_counts = (
        results_df.groupby("organ")
        .agg(
            total_features=("feature", "count"),
            sig_raw=("significant_raw", "sum"),
            sig_fdr=("significant_fdr", "sum"),
        )
        .reset_index()
        .sort_values("sig_fdr", ascending=False)
    )
    organ_counts["pct_sig_fdr"] = (
        organ_counts["sig_fdr"] / organ_counts["total_features"] * 100
    ).round(1)
    organ_csv = os.path.join(OUTPUT_DIR, "organ_significance_counts.csv")
    organ_counts.to_csv(organ_csv, index=False)
    print(f"Saved organ counts -> {organ_csv}", flush=True)

    # ---- Feature prevalence: which features are significant in the most organs ----
    # Strip organ-specific prefix: feature names like "original_shape_VoxelVolume"
    # are the same across organs, so we can count how many organs each appears in
    sig_features = results_df[results_df["significant_fdr"]].copy()

    if len(sig_features) > 0:
        feat_prevalence = (
            sig_features.groupby("feature")
            .agg(
                n_organs_significant=("organ", "count"),
                organs=("organ", lambda x: ", ".join(sorted(x))),
                median_p=("p_value", "median"),
            )
            .reset_index()
            .sort_values("n_organs_significant", ascending=False)
        )
    else:
        feat_prevalence = pd.DataFrame(
            columns=["feature", "n_organs_significant", "organs", "median_p"]
        )

    feat_csv = os.path.join(OUTPUT_DIR, "feature_prevalence.csv")
    feat_prevalence.to_csv(feat_csv, index=False)
    print(f"Saved feature prevalence -> {feat_csv}", flush=True)

    # ==================================================================
    # FIGURE 1: Top organs by number of significant features
    # ==================================================================
    top_organs = organ_counts.head(30)
    fig1, ax1 = plt.subplots(figsize=(12, max(6, len(top_organs) * 0.35)))
    colors = ["#e74c3c" if v > 0 else "#95a5a6"
              for v in top_organs["sig_fdr"].values]
    ax1.barh(range(len(top_organs)), top_organs["sig_fdr"].values,
             color=colors, edgecolor="white")
    ax1.set_yticks(range(len(top_organs)))
    ax1.set_yticklabels(top_organs["organ"].values, fontsize=9)
    ax1.set_xlabel("Number of Significant Features (FDR-corrected)")
    ax1.set_title("Organs Ranked by Number of Significant M/F Features")
    ax1.invert_yaxis()

    # Annotate counts
    for i, (sig, total) in enumerate(
        zip(top_organs["sig_fdr"].values, top_organs["total_features"].values)
    ):
        ax1.text(sig + 0.5, i, f"{int(sig)}/{int(total)}",
                 va="center", fontsize=8)

    fig1.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "top_organs_barplot.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fig1_path}", flush=True)
    plt.close(fig1)

    # ==================================================================
    # FIGURE 2: Top features by prevalence across organs
    # ==================================================================
    if len(feat_prevalence) > 0:
        top_feats = feat_prevalence.head(30)
        fig2, ax2 = plt.subplots(figsize=(12, max(6, len(top_feats) * 0.35)))
        ax2.barh(range(len(top_feats)), top_feats["n_organs_significant"].values,
                 color="#3498db", edgecolor="white")
        ax2.set_yticks(range(len(top_feats)))
        ax2.set_yticklabels(top_feats["feature"].values, fontsize=8)
        ax2.set_xlabel("Number of Organs Where Feature is Significant")
        ax2.set_title("Most Prevalent Significant Features Across Organs")
        ax2.invert_yaxis()

        for i, n in enumerate(top_feats["n_organs_significant"].values):
            ax2.text(n + 0.2, i, str(int(n)), va="center", fontsize=8)

        fig2.tight_layout()
        fig2_path = os.path.join(OUTPUT_DIR, "top_features_barplot.png")
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {fig2_path}", flush=True)
        plt.close(fig2)

    # ==================================================================
    # FIGURE 3: Heatmap — organs x feature classes
    # ==================================================================
    # Group features by their class (firstorder, shape, glcm, etc.)
    def get_feature_class(feat_name):
        # e.g. "original_shape_VoxelVolume" -> "shape"
        # e.g. "log-sigma-1-0-mm-3D_firstorder_Mean" -> "firstorder"
        # e.g. "wavelet-LLH_glcm_Contrast" -> "glcm"
        parts = feat_name.split("_")
        for cls in ["firstorder", "shape", "glcm", "glrlm",
                     "glszm", "gldm", "ngtdm"]:
            if cls in parts:
                return cls
        return "other"

    def get_image_type(feat_name):
        if feat_name.startswith("original_"):
            return "Original"
        elif feat_name.startswith("log-sigma"):
            return "LoG"
        elif feat_name.startswith("wavelet-"):
            return "Wavelet"
        return "Other"

    results_df["feature_class"] = results_df["feature"].apply(get_feature_class)
    results_df["image_type"] = results_df["feature"].apply(get_image_type)

    # Heatmap: organs (rows) x (image_type + feature_class) (columns)
    sig_counts_pivot = (
        results_df[results_df["significant_fdr"]]
        .groupby(["organ", "image_type", "feature_class"])
        .size()
        .reset_index(name="count")
    )

    if len(sig_counts_pivot) > 0:
        sig_counts_pivot["category"] = (
            sig_counts_pivot["image_type"] + "_" + sig_counts_pivot["feature_class"]
        )
        heatmap_df = sig_counts_pivot.pivot_table(
            index="organ", columns="category", values="count", fill_value=0
        )
        # Sort organs by total significant features
        heatmap_df["_total"] = heatmap_df.sum(axis=1)
        heatmap_df = heatmap_df.sort_values("_total", ascending=False).head(40)
        heatmap_df = heatmap_df.drop(columns=["_total"])

        if len(heatmap_df) > 0 and len(heatmap_df.columns) > 0:
            fig3, ax3 = plt.subplots(
                figsize=(max(10, len(heatmap_df.columns) * 0.8),
                         max(8, len(heatmap_df) * 0.35))
            )
            im = ax3.imshow(heatmap_df.values, aspect="auto", cmap="YlOrRd")
            ax3.set_xticks(range(len(heatmap_df.columns)))
            ax3.set_xticklabels(heatmap_df.columns, rotation=45, ha="right",
                                fontsize=8)
            ax3.set_yticks(range(len(heatmap_df)))
            ax3.set_yticklabels(heatmap_df.index, fontsize=8)
            ax3.set_title("Significant Features by Organ and Feature Category\n"
                          "(FDR-corrected, count per cell)")
            plt.colorbar(im, ax=ax3, label="# Significant Features")
            fig3.tight_layout()
            fig3_path = os.path.join(OUTPUT_DIR, "significance_heatmap.png")
            fig3.savefig(fig3_path, dpi=150, bbox_inches="tight")
            print(f"Saved -> {fig3_path}", flush=True)
            plt.close(fig3)

    # ---- Console summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total organs analyzed:        {len(organ_counts)}")
    print(f"Total features tested:        {len(results_df)}")
    print(f"Significant (raw p < {alpha}): {n_sig_raw} "
          f"({n_sig_raw/len(results_df)*100:.1f}%)")
    print(f"Significant (FDR-corrected):  {n_sig_fdr} "
          f"({n_sig_fdr/len(results_df)*100:.1f}%)")

    if len(organ_counts) > 0:
        print(f"\nTop 10 organs by significant features (FDR):")
        for _, row in organ_counts.head(10).iterrows():
            print(f"  {row['organ']:<35} "
                  f"{int(row['sig_fdr']):>4}/{int(row['total_features'])} "
                  f"({row['pct_sig_fdr']:.1f}%)")

    if len(feat_prevalence) > 0:
        print(f"\nTop 10 most prevalent features (significant in most organs):")
        for _, row in feat_prevalence.head(10).iterrows():
            print(f"  {row['feature']:<50} "
                  f"-> {int(row['n_organs_significant'])} organs")

    print(f"\nAll outputs in {OUTPUT_DIR}/")


def main():
    p = argparse.ArgumentParser(
        description="Wilcoxon rank-sum analysis across all organs"
    )
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                   help="Directory with per-organ CSVs")
    p.add_argument("--labels", default=DEFAULT_LABELS,
                   help="CSV with subject_id + sex columns")
    p.add_argument("--alpha", default=0.05, type=float,
                   help="Significance level (default: 0.05)")
    args = p.parse_args()

    run_analysis(args.input_dir, args.labels, alpha=args.alpha)


if __name__ == "__main__":
    main()
