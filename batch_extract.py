#!/usr/bin/env python3
"""
Batch Radiomics Feature Extraction — All 117 TotalSegmentator Organs

Scans a folder of NIfTI CT volumes, runs TotalSegmentator to produce
all 117 organ masks, then extracts radiomics features from each organ
individually using PyRadiomics (Original + LoG + Wavelet ≈ 1,130
features per organ).

Output: CT/organ_features/<organ_name>.csv  (one CSV per organ,
        rows = subjects, columns = features)

Supports checkpointing — if interrupted, re-run the same command
and it will skip already-completed (subject, organ) pairs.

Usage:
    python batch_extract.py CT/scans
    python batch_extract.py CT/scans --labels CT/First30.csv
"""
import argparse, os, sys, glob, subprocess, tempfile, shutil, time, json, logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# Suppress verbose PyRadiomics / SimpleITK logging
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "pyradiomics", "ct_config.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "CT", "organ_features")
VENV_BIN = os.path.join(BASE_DIR, "mitrpENV", "bin")

# Import preprocessing functions from pyradiomics package
from pyradiomics.extract_radiomics import to_lps, clamp_ct_hu


# ---------------------------------------------------------------------------
# TotalSegmentator — all 117 organs
# ---------------------------------------------------------------------------
def run_totalsegmentator(ct_path, seg_dir):
    """
    Run TotalSegmentator on a CT volume with all 117 default organs.
    Uses --fast (3mm model) for speed.
    Output: seg_dir/<organ_name>.nii.gz  for each organ
    """
    ts_bin = os.path.join(VENV_BIN, "TotalSegmentator")
    cmd = [ts_bin, "-i", ct_path, "-o", seg_dir, "--fast"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed:\n{result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def progress_dir(output_dir):
    d = os.path.join(output_dir, ".progress")
    os.makedirs(d, exist_ok=True)
    return d


def is_done(output_dir, scan_id, organ):
    marker = os.path.join(progress_dir(output_dir), f"{scan_id}__{organ}.done")
    return os.path.isfile(marker)


def mark_done(output_dir, scan_id, organ):
    marker = os.path.join(progress_dir(output_dir), f"{scan_id}__{organ}.done")
    with open(marker, "w") as f:
        f.write("")


# ---------------------------------------------------------------------------
# Preprocessing (once per subject)
# ---------------------------------------------------------------------------
def preprocess_ct(ct_path, tmp_dir):
    """Orient to LPS and clamp HU. Returns path to preprocessed image."""
    img_lps = os.path.join(tmp_dir, "img_lps.nii.gz")
    to_lps(ct_path, img_lps)
    img_clamped = os.path.join(tmp_dir, "img_clamped.nii.gz")
    clamp_ct_hu(img_lps, img_clamped)
    return img_clamped


# ---------------------------------------------------------------------------
# Per-organ feature extraction
# ---------------------------------------------------------------------------
def extract_organ(img_path, mask_path, config_yaml, tmp_dir):
    """
    Extract radiomics features for a single organ mask.
    Returns dict of {feature_name: value} or None if mask is empty.
    """
    # Orient mask to LPS to match the preprocessed image
    mask_lps = os.path.join(tmp_dir, "mask_lps.nii.gz")
    to_lps(mask_path, mask_lps)

    # Check mask is not empty
    mask_img = sitk.ReadImage(mask_lps)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    if mask_arr.max() == 0:
        return None

    # Binarize (some TotalSegmentator masks may have values > 1)
    mask_bin = sitk.BinaryThreshold(mask_img, lowerThreshold=1,
                                     upperThreshold=255,
                                     insideValue=1, outsideValue=0)
    mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)
    mask_bin_path = os.path.join(tmp_dir, "mask_bin.nii.gz")
    sitk.WriteImage(mask_bin, mask_bin_path, True)

    extractor = featureextractor.RadiomicsFeatureExtractor(config_yaml)
    result = extractor.execute(img_path, mask_bin_path)

    feats = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            feats[k] = float(v)
        except (TypeError, ValueError):
            continue
    return feats


# ---------------------------------------------------------------------------
# Accumulate results into per-organ CSVs
# ---------------------------------------------------------------------------
def save_organ_row(output_dir, organ, scan_id, feats):
    """Append one subject's features to the organ's CSV file."""
    csv_path = os.path.join(output_dir, f"{organ}.csv")
    row = {"subject_id": scan_id, **feats}
    row_df = pd.DataFrame([row])

    if os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        # Avoid duplicate rows on re-run
        if scan_id in existing["subject_id"].values:
            return
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Batch 117-organ radiomics extraction (Original + LoG + Wavelet)"
    )
    p.add_argument(
        "folder",
        help="Folder containing .nii.gz CT scans (filename = subject ID)",
    )
    p.add_argument(
        "--labels", default=None,
        help="CSV with clinical labels (must have 'Subject ID' column)",
    )
    p.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"PyRadiomics YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for per-organ CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: {args.folder} is not a directory", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover scans
    nifti_files = sorted(glob.glob(os.path.join(args.folder, "*.nii.gz")))
    scans = []
    for f in nifti_files:
        scan_id = os.path.basename(f).replace(".nii.gz", "")
        scans.append((scan_id, f))

    print(f"Found {len(scans)} scans in {args.folder}", flush=True)
    if not scans:
        print("No .nii.gz files found. Nothing to do.", flush=True)
        sys.exit(0)

    # Persistent cache directories (survive Ctrl+C restarts)
    seg_cache = os.path.join(args.output_dir, ".segs")
    preproc_cache = os.path.join(args.output_dir, ".preproc")
    os.makedirs(seg_cache, exist_ok=True)
    os.makedirs(preproc_cache, exist_ok=True)

    # Process each subject
    total_t0 = time.time()
    for i, (scan_id, ct_path) in enumerate(scans):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(scans)}] Subject: {scan_id}", flush=True)
        print(f"{'='*60}", flush=True)
        subj_t0 = time.time()

        # Check if ALL organs are already done for this subject
        all_done = True
        for organ_name_check in glob.glob(os.path.join(seg_cache, scan_id, "*.nii.gz")):
            pass  # just need to check progress markers
        # Quick check: if we have 117 .done markers for this subject, skip entirely
        done_markers = glob.glob(
            os.path.join(progress_dir(args.output_dir), f"{scan_id}__*.done")
        )
        if len(done_markers) >= 117:
            print("  All organs already done, skipping.", flush=True)
            continue

        try:
            # --- Step 1: TotalSegmentator (cached) ---
            subj_seg_dir = os.path.join(seg_cache, scan_id)
            if os.path.isdir(subj_seg_dir) and len(
                glob.glob(os.path.join(subj_seg_dir, "*.nii.gz"))
            ) >= 100:
                print("  Using cached segmentation masks.", flush=True)
            else:
                os.makedirs(subj_seg_dir, exist_ok=True)
                print("  Running TotalSegmentator (all organs) ...", flush=True)
                ts_t0 = time.time()
                run_totalsegmentator(ct_path, subj_seg_dir)
                print(f"  TotalSegmentator done ({time.time()-ts_t0:.0f}s)",
                      flush=True)

            # --- Step 2: Preprocess CT (cached) ---
            img_preprocessed = os.path.join(preproc_cache, f"{scan_id}.nii.gz")
            if os.path.isfile(img_preprocessed):
                print("  Using cached preprocessed CT.", flush=True)
            else:
                print("  Preprocessing CT (LPS + HU clamp) ...", flush=True)
                preprocess_ct(ct_path, preproc_cache)
                # preprocess_ct writes to img_clamped in the dir; rename
                tmp_clamped = os.path.join(preproc_cache, "img_clamped.nii.gz")
                os.rename(tmp_clamped, img_preprocessed)
                # Clean up intermediate LPS file
                tmp_lps = os.path.join(preproc_cache, "img_lps.nii.gz")
                if os.path.isfile(tmp_lps):
                    os.remove(tmp_lps)

            # --- Step 3: Extract features for each organ ---
            mask_files = sorted(
                glob.glob(os.path.join(subj_seg_dir, "*.nii.gz"))
            )
            print(f"  Found {len(mask_files)} organ masks", flush=True)

            for j, mask_path in enumerate(mask_files):
                organ = os.path.basename(mask_path).replace(".nii.gz", "")

                # Skip if already done (checkpoint)
                if is_done(args.output_dir, scan_id, organ):
                    continue

                organ_t0 = time.time()
                print(f"    [{j+1}/{len(mask_files)}] {organ} ...",
                      end="", flush=True)

                organ_tmp = tempfile.mkdtemp(prefix=f"organ_{organ}_")
                try:
                    feats = extract_organ(
                        img_preprocessed, mask_path,
                        args.config, organ_tmp
                    )

                    if feats is None:
                        print(f" EMPTY mask, skipped ({time.time()-organ_t0:.0f}s)",
                              flush=True)
                    else:
                        save_organ_row(args.output_dir, organ, scan_id, feats)
                        print(f" {len(feats)} features ({time.time()-organ_t0:.0f}s)",
                              flush=True)

                    mark_done(args.output_dir, scan_id, organ)

                except Exception as e:
                    print(f" ERROR: {e}", flush=True)

                finally:
                    shutil.rmtree(organ_tmp, ignore_errors=True)

        except Exception as e:
            print(f"  SUBJECT ERROR: {e}", flush=True)

        elapsed = time.time() - subj_t0
        print(f"  Subject done in {elapsed/60:.1f} min", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nAll done. Total time: {total_elapsed/60:.1f} min", flush=True)

    # --- Merge labels into each organ CSV if provided ---
    if args.labels and os.path.isfile(args.labels):
        print(f"\nMerging labels from {args.labels} ...", flush=True)
        labels_df = pd.read_csv(args.labels)
        if "Subject ID" in labels_df.columns:
            labels_df.rename(columns={"Subject ID": "subject_id"}, inplace=True)
        labels_df = labels_df.drop_duplicates(subset="subject_id")
        label_cols = ["subject_id", "sex", "age", "diagnosis"]
        labels_df = labels_df[[c for c in label_cols if c in labels_df.columns]]

        organ_csvs = glob.glob(os.path.join(args.output_dir, "*.csv"))
        for csv_path in organ_csvs:
            df = pd.read_csv(csv_path)
            if "sex" in df.columns:
                continue  # already merged
            df = df.merge(labels_df, on="subject_id", how="left")
            # Move label cols to front
            lcols = [c for c in label_cols if c in df.columns]
            fcols = [c for c in df.columns if c not in lcols]
            df = df[lcols + fcols]
            df.to_csv(csv_path, index=False)

        print(f"Labels merged into {len(organ_csvs)} organ CSVs", flush=True)

    print(f"\nOutput directory: {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
