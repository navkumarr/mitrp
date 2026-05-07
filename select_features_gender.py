#!/usr/bin/env python3
"""
Feature Selection for All Organs - Gender Classification

Runs consensus feature selection (correlation removal + MI + permutation importance)
on all organs and saves results to a structured output format.

Output: features/selected_features_gender.csv with columns:
  - organ
  - feature_rank
  - feature_name
  - consensus_score
  - mi_rank
  - perm_rank

Usage:
    python select_features_gender.py
    python select_features_gender.py --top-k 25 --output results.csv
"""
import argparse
import os
import glob
import pandas as pd
from feature_selection import (
    load_organ_data,
    consensus_feature_selection,
    remove_correlated_features,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORGAN_FEATURES_DIR = os.path.join(BASE_DIR, "CT", "organ_features")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "figures")


def main():
    parser = argparse.ArgumentParser(
        description="Select top K non-correlated features per organ for gender classification"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_ORGAN_FEATURES_DIR,
        help="Directory with organ feature CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to select per organ (default: 20)",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for removing redundant features (default: 0.95)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all organ CSVs
    organ_csvs = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    organ_names = [os.path.basename(c).replace(".csv", "") for c in organ_csvs]

    # Filter out non-organ files (hidden directories, etc.)
    organ_names = [o for o in organ_names if not o.startswith(".")]

    print(f"\nFound {len(organ_names)} organs in {args.input_dir}")
    print(f"Selecting top {args.top_k} features per organ\n")

    results = []

    # Process each organ
    for i, organ_name in enumerate(organ_names, 1):
        try:
            print(f"[{i}/{len(organ_names)}] {organ_name}...", end=" ", flush=True)

            # Load data
            X, y, feature_names = load_organ_data(organ_name, args.input_dir)

            # Run consensus feature selection
            top_features, importance_scores, removed_pairs = consensus_feature_selection(
                X, y, feature_names, k=args.top_k, correlation_threshold=args.correlation_threshold
            )

            # Build result rows
            for rank, feature in enumerate(top_features, 1):
                score = importance_scores.get(feature, 0)
                results.append({
                    "organ": organ_name,
                    "feature_rank": rank,
                    "feature_name": feature,
                    "consensus_score": score,
                    "n_correlated_removed": len(removed_pairs),
                })

            print(f"✓ selected {len(top_features)} features")

        except FileNotFoundError:
            print(f"⚠ skipped (CSV not found)")
        except Exception as e:
            print(f"✗ error: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(args.output_dir, "selected_features_gender.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved {len(results_df)} feature selections to {output_csv}")

    # Print summary statistics
    summary = results_df.groupby("organ").agg(
        n_features=("feature_rank", "count"),
        avg_score=("consensus_score", "mean"),
        n_correlated_removed=("n_correlated_removed", "first"),
    ).reset_index()

    print(f"\n{'ORGAN':<35} {'FEATURES':<8} {'AVG SCORE':<15} {'CORRELATED REMOVED':<18}")
    print("-" * 75)
    for _, row in summary.iterrows():
        print(f"{row['organ']:<35} {int(row['n_features']):<8.0f} {row['avg_score']:<15.4f} {int(row['n_correlated_removed']):<18.0f}")

    print(f"\nSummary:")
    print(f"  Total organs processed: {summary.shape[0]}")
    print(f"  Total feature selections: {len(results_df)}")
    print(f"  Avg features per organ: {results_df.groupby('organ').size().mean():.1f}")
    print(f"  Avg correlation redundancy removed: {summary['n_correlated_removed'].mean():.0f} pairs")

    # Create a feature summary: which features appear in which organs
    feature_summary = results_df.groupby("feature_name").agg(
        n_organs=("organ", "count"),
        organs=("organ", lambda x: ", ".join(x)),
        avg_rank=("feature_rank", "mean"),
    ).reset_index().sort_values("n_organs", ascending=False)

    feature_summary_csv = os.path.join(args.output_dir, "selected_features_prevalence.csv")
    feature_summary.to_csv(feature_summary_csv, index=False)
    print(f"\n✓ Saved feature prevalence to {feature_summary_csv}")

    # Show top 10 most prevalent features
    print(f"\nTop 10 Most Prevalent Features (across organs):")
    print("-" * 75)
    for idx, (_, row) in enumerate(feature_summary.head(10).iterrows(), 1):
        feat_short = row['feature_name'].replace("original_", "").replace("log-sigma-", "LoG_").replace("wavelet-", "Wv_")
        print(f"  {idx:2d}. {feat_short:<50s} ({int(row['n_organs']):3d} organs, avg rank: {row['avg_rank']:5.1f})")

    print(f"\n✓ Feature selection complete!\n")


if __name__ == "__main__":
    main()
