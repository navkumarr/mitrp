#!/usr/bin/env python3
"""
Batch Radiomics Feature Extraction

Scans a folder of NIfTI CT volumes, runs TotalSegmentator to produce
pelvis segmentation masks (hip_left, hip_right, sacrum), extracts
radiomics features from the combined pelvis region, and optionally
merges with labels from a CSV.

Each .nii.gz file in the folder is treated as one subject. The subject
ID is derived from the filename (e.g. PETCT_06a46414eb.nii.gz ->
subject_id = PETCT_06a46414eb).

Usage:
    python batch_extract.py CT/scans
    python batch_extract.py CT/scans --labels CT/First30.csv
    python batch_extract.py CT/scans --output results.csv --config pyradiomics/pyrads.yaml

Output: CT/radiomics_features.csv  (one row per scan, columns = features + labels)
"""
import argparse, os, sys, glob, subprocess, tempfile, shutil, time
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "pyradiomics", "ct_config.yaml")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "CT", "radiomics_features.csv")
VENV_BIN = os.path.join(BASE_DIR, "mitrpENV", "bin")

# Pelvis ROIs for gender classification (strongest skeletal sexual dimorphism)
PELVIS_ROIS = ["hip_left", "hip_right", "sacrum"]

# Import preprocessing functions from existing pipeline
sys.path.insert(0, os.path.join(BASE_DIR, "pyradiomics"))
from extract_radiomics import to_lps, clamp_ct_hu

# ---------------------------------------------------------------------------
# TotalSegmentator segmentation
# ---------------------------------------------------------------------------
def run_totalsegmentator(ct_path, seg_dir):
    """
    Run TotalSegmentator on a CT volume to produce pelvis organ masks.

    Uses --roi_subset to only segment hip_left, hip_right, and sacrum,
    which is faster than segmenting all 104 classes. Uses --fast (3mm)
    for speed since we only need the mask geometry, not sub-millimetre
    precision.

    Output: seg_dir/{hip_left,hip_right,sacrum}.nii.gz
    """
    ts_bin = os.path.join(VENV_BIN, "TotalSegmentator")
    cmd = [
        ts_bin,
        "-i", ct_path,
        "-o", seg_dir,
        "--fast",
        "--roi_subset", *PELVIS_ROIS,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed:\n{result.stderr[-500:]}"
        )


def merge_masks(seg_dir, rois, out_path):
    """
    Merge multiple binary organ masks into a single binary mask.

    Reads each ROI NIfTI from seg_dir, ORs them together, and writes
    a combined binary mask. This gives PyRadiomics one contiguous pelvic
    region to extract features from.
    """
    combined = None
    for roi in rois:
        roi_path = os.path.join(seg_dir, f"{roi}.nii.gz")
        if not os.path.isfile(roi_path):
            continue
        mask = sitk.ReadImage(roi_path)
        binary = sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255,
                                      insideValue=1, outsideValue=0)
        binary = sitk.Cast(binary, sitk.sitkUInt8)
        if combined is None:
            combined = binary
        else:
            combined = sitk.Or(combined, binary)

    if combined is None:
        raise RuntimeError(f"No pelvis ROI masks found in {seg_dir}")

    sitk.WriteImage(combined, out_path, True)
    return out_path


# ---------------------------------------------------------------------------
# Per-scan extraction
# ---------------------------------------------------------------------------
def extract_scan(scan_id, ct_path, config_yaml, tmp_dir):
    """
    Run the full pipeline on one CT scan:
      1. Run TotalSegmentator for pelvis segmentation
      2. Merge pelvis masks into one binary mask
      3. Orient CT to LPS
      4. Clamp HU
      5. Extract radiomics features (PyRadiomics handles resampling internally)
    Returns a dict of {feature_name: value}.
    """
    seg_dir = os.path.join(tmp_dir, "segs")
    os.makedirs(seg_dir, exist_ok=True)

    # Step 1 – TotalSegmentator pelvis segmentation
    run_totalsegmentator(ct_path, seg_dir)

    # Step 2 – merge hip_left + hip_right + sacrum into one mask
    mask_path = os.path.join(tmp_dir, "pelvis_mask.nii.gz")
    merge_masks(seg_dir, PELVIS_ROIS, mask_path)

    # Step 3 – orient to LPS
    img_lps = os.path.join(tmp_dir, "img_lps.nii.gz")
    mask_lps = os.path.join(tmp_dir, "mask_lps.nii.gz")
    to_lps(ct_path, img_lps)
    to_lps(mask_path, mask_lps)

    # Step 4 – HU clamping
    img_clamped = os.path.join(tmp_dir, "img_clamped.nii.gz")
    clamp_ct_hu(img_lps, img_clamped)

    # Step 5 – extract features
    extractor = featureextractor.RadiomicsFeatureExtractor(config_yaml)
    result = extractor.execute(img_clamped, mask_lps)

    feats = {"subject_id": scan_id}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            feats[k] = float(v)
        except (TypeError, ValueError):
            continue
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Batch TotalSegmentator + PyRadiomics feature extraction"
    )
    p.add_argument(
        "folder",
        help="Folder containing .nii.gz CT scans (one file per subject, "
             "filename used as subject ID)",
    )
    p.add_argument(
        "--labels", default=None,
        help="CSV with clinical labels (must have 'Subject ID' column). "
             "If provided, features are merged with labels in the output.",
    )
    p.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to PyRadiomics YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: {args.folder} is not a directory", flush=True)
        sys.exit(1)

    # Discover scans: every .nii.gz in the folder is one subject
    nifti_files = sorted(glob.glob(os.path.join(args.folder, "*.nii.gz")))
    scans = []
    for f in nifti_files:
        basename = os.path.basename(f)
        # Strip .nii.gz to get subject ID
        scan_id = basename.replace(".nii.gz", "")
        scans.append((scan_id, f))

    print(f"Found {len(scans)} scans in {args.folder}", flush=True)
    print(f"Pelvis ROIs: {', '.join(PELVIS_ROIS)}", flush=True)

    if not scans:
        print("No .nii.gz files found. Nothing to do.", flush=True)
        sys.exit(0)

    # Load labels if provided
    labels_df = None
    if args.labels and os.path.isfile(args.labels):
        labels_df = pd.read_csv(args.labels)
        if "Subject ID" in labels_df.columns:
            labels_df.rename(columns={"Subject ID": "subject_id"}, inplace=True)
        labels_df = labels_df.drop_duplicates(subset="subject_id")
        print(f"Loaded {len(labels_df)} labels from {args.labels}", flush=True)

    # Process each scan
    all_features = []
    for i, (scan_id, ct_path) in enumerate(scans):
        print(f"  [{i+1}/{len(scans)}] Processing {scan_id} ...", flush=True)
        t0 = time.time()

        tmp_dir = tempfile.mkdtemp(prefix=f"rads_{scan_id}_")
        try:
            feats = extract_scan(scan_id, ct_path, args.config, tmp_dir)
            all_features.append(feats)
            elapsed = time.time() - t0
            print(f"    -> {len(feats) - 1} features extracted ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f"    -> ERROR: {e}", flush=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not all_features:
        print("No features extracted. Exiting.", flush=True)
        sys.exit(1)

    # Build dataframe
    feat_df = pd.DataFrame(all_features)

    # Merge with labels if available
    if labels_df is not None:
        feat_df = feat_df.merge(labels_df, on="subject_id", how="left")
        # Move label columns to the front
        label_cols = [c for c in labels_df.columns if c in feat_df.columns]
        feature_cols = [c for c in feat_df.columns if c not in label_cols]
        feat_df = feat_df[label_cols + feature_cols]
        n_features = len(feature_cols)
    else:
        n_features = len(feat_df.columns) - 1  # minus subject_id

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    feat_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(feat_df)} rows x {n_features} features -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
