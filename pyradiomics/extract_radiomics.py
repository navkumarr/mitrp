#!/usr/bin/env python3
"""
Radiomics Feature Extraction Pipeline with Standardization

This pipeline implements 7 key standardization steps to ensure reproducible,
cross-scanner radiomics feature extraction:
1. Orientation Standardization (RAS)
2. Bias Field Correction (MRI only)
3. Intensity Normalization (z-score for MRI, HU range for CT)
4. Spatial Resampling (isotropic 1x1x1 mm)
5. Consistent Mask and Cropping Policy
6. Gray-Level Discretization (fixed bin width for CT, fixed bin count for MRI)
7. Version and Configuration Control (locked software versions and YAML config)
"""
import os, json, argparse
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor

# =============================================================================
# STEP 1: ORIENTATION STANDARDIZATION
# =============================================================================
def to_ras(nifti_in, nifti_out):
    """
    Convert NIfTI image to Right-Anterior-Superior (RAS) coordinate system.
    
    Purpose: Different scanners and DICOM headers may encode axes inconsistently,
    leading to mirrored or transposed volumes. Standardizing to RAS ensures that:
    - Spatial features (e.g., GLCM directions) align identically across images
    - Masks correspond precisely to their anatomical regions
    - Left-right, anterior-posterior orientations are consistent
    
    This is a baseline preprocessing step applied to all images and masks.
    """
    img = nib.load(nifti_in)
    ras = nib.as_closest_canonical(img)
    nib.save(ras, nifti_out)
    return nifti_out

# =============================================================================
# STEP 2: BIAS FIELD CORRECTION (MRI ONLY)
# =============================================================================
def n4_bias_correct(nifti_in, nifti_out, shrink=4, bspline=(200,)):
    """
    Apply N4 bias field correction to remove intensity non-uniformities in MRI.
    
    Purpose: Magnetic resonance images suffer from low-frequency intensity
    gradients caused by:
    - Coil sensitivity variations
    - Magnetic field inhomogeneity (B1 field)
    
    These artifacts create artificial intensity gradients across the image that
    do not reflect true tissue properties. N4 correction:
    - Preserves true tissue contrast
    - Removes spatially-varying intensity drift
    - Prevents distortion of first-order and texture features
    
    Note: This step should ONLY be applied to MRI. Skip for CT images, which
    already have standardized Hounsfield Units (HU).
    
    Implementation details:
    - Images are cast to float32 (N4 requires floating-point data)
    - Otsu thresholding creates a foreground mask automatically
    - Optional shrinking (default 4x) speeds up computation
    - Bias field is computed on downsampled image, then applied at full resolution
    """
    img = sitk.ReadImage(nifti_in)
    
    # Cast to float32 as N4 doesn't support integer types
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    
    # Create mask using Otsu thresholding
    # If you have a dedicated brain/organ mask, use that instead
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    # Optional: shrink image for faster processing
    if shrink > 1:
        img_shrunk = sitk.Shrink(img, [shrink] * img.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [shrink] * img.GetDimension())
    else:
        img_shrunk = img
        mask_shrunk = mask
    
    # Configure N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(0.001)
    
    # Execute on shrunk image
    corrected_shrunk = corrector.Execute(img_shrunk, mask_shrunk)
    
    # If we shrunk the image, get the bias field and apply to full resolution
    if shrink > 1:
        log_bias_field = corrector.GetLogBiasFieldAsImage(img)
        corrected = img / sitk.Exp(log_bias_field)
    else:
        corrected = corrected_shrunk
    
    sitk.WriteImage(corrected, nifti_out, True)
    return nifti_out

# =============================================================================
# STEP 4: SPATIAL RESAMPLING TO ISOTROPIC VOXEL SPACING
# =============================================================================
def resample_pair(image_path, mask_path, spacing=(1.0,1.0,1.0)):
    """
    Resample both image and mask to isotropic 1x1x1 mm voxel spacing.
    
    Purpose: Texture matrices (GLCM, GLRLM, GLSZM) depend on voxel adjacency.
    Variable voxel sizes across scanners introduce scale-related bias because:
    - A "neighbor" in one image might span 0.5mm, in another 2mm
    - Run lengths and zone sizes become incomparable
    - Directional features (e.g., GLCM angles) have inconsistent meaning
    
    Standardizing to 1x1x1 mm ensures:
    - All radiomic features are computed over equivalent physical distances
    - Texture patterns are scale-invariant across datasets
    - Multi-scanner studies have comparable feature values
    
    Implementation:
    - Linear interpolation for images (preserves intensity gradients)
    - Nearest-neighbor interpolation for masks (preserves label integrity)
    - Both are resampled to the same grid to maintain geometric correspondence
    
    This corresponds to STEP 5 (mask policy) as well: the mask is transformed
    with identical spatial parameters to maintain precise ROI alignment.
    """
    img = sitk.ReadImage(image_path)
    msk = sitk.ReadImage(mask_path)

    def _resample(im, spacing, interp):
        """
        Internal resampling function.
        Calculates new grid size based on original spacing and desired spacing.
        """
        original_spacing = im.GetSpacing()
        original_size = im.GetSize()
        # Calculate new size to preserve physical extent
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interp)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(im.GetOrigin())
        resampler.SetOutputDirection(im.GetDirection())
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(im)

    # Resample image with linear interpolation (smooth intensity transitions)
    img_r = _resample(img, spacing, sitk.sitkLinear)
    # Resample mask with nearest-neighbor (preserve discrete labels)
    msk_r = _resample(msk, spacing, sitk.sitkNearestNeighbor)

    # Overwrite with resampled versions
    sitk.WriteImage(img_r, image_path, True)
    sitk.WriteImage(msk_r, mask_path, True)

# =============================================================================
# STEP 3, 5, 6: FEATURE EXTRACTION WITH INTENSITY NORMALIZATION,
# MASK POLICY, AND GRAY-LEVEL DISCRETIZATION
# =============================================================================
def extract_features(image_path, mask_path, config_yaml):
    """
    Extract radiomic features using PyRadiomics with locked configuration.
    
    This function integrates three critical standardization steps through the
    YAML configuration file:
    
    STEP 3 - INTENSITY NORMALIZATION:
    - For MRI: z-score normalization within ROI (mean=0, std=1)
      Controlled by: normalize=true in YAML
      Purpose: Inter-scanner and inter-subject MRI intensities are arbitrary.
               Z-scoring creates consistent statistical distributions.
    - For CT: Intensities left in Hounsfield Units (HU), clipped to [-1000, 3000]
      Purpose: HU are already physically standardized. Clipping removes outliers
               and metal artifacts without destroying the physical meaning.
    
    STEP 5 - CONSISTENT MASK AND CROPPING POLICY:
    - Controlled by: correctMask=true, preCrop=true in YAML
    - The mask is applied with identical spatial parameters as the image
    - ROI is cropped to bounding box to eliminate contextual variation
    - Ensures all texture statistics are computed within precisely defined regions
    
    STEP 6 - GRAY-LEVEL DISCRETIZATION:
    - For CT: Fixed bin width of 25 HU (preserves physical HU interpretability)
      Controlled by: binWidth=25 in YAML
    - For MRI: Fixed bin count of 64 gray levels (after z-score normalization)
      Controlled by: binCount=64 in YAML
    - Critical for reproducible GLCM, GLRLM, GLSZM texture matrices
    - Inconsistent discretization leads to non-comparable texture features
    
    Returns:
    - Dictionary of radiomic features (excluding diagnostic metadata)
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(config_yaml)
    result = extractor.execute(image_path, mask_path)
    # Keep scalar features only (exclude diagnostics_ metadata)
    return {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}

# =============================================================================
# MAIN PIPELINE: ORCHESTRATES ALL 7 STANDARDIZATION STEPS
# =============================================================================
def main():
    """
    Deterministic radiomics feature extraction pipeline.
    
    STEP 7 - VERSION AND CONFIGURATION CONTROL:
    All standardization parameters are locked in the YAML configuration file:
    - Software versions: PyRadiomics 3.1.0, SimpleITK 2.3.1, NumPy 1.26.4
    - Processing parameters: resampling method, interpolation, normalization
    - Discretization: bin width/count, masking policy
    
    Even minor version changes (e.g., rounding behavior, interpolation algorithms)
    can alter numerical outputs. Version locking ensures:
    - Cross-platform reproducibility
    - Deterministic feature values
    - Auditability for regulatory compliance
    
    Pipeline workflow:
    1. Parse command-line arguments
    2. Convert image and mask to RAS orientation (STEP 1)
    3. Apply N4 bias correction if requested (STEP 2, MRI only)
    4. Resample to isotropic 1x1x1 mm spacing (STEP 4)
    5. Extract features with normalization, masking, and discretization (STEPS 3,5,6)
    6. Save features as JSON and CSV
    """
    p = argparse.ArgumentParser(description="Deterministic PyRadiomics extraction")
    p.add_argument("--image", required=True, help="Path to NIfTI image (.nii/.nii.gz)")
    p.add_argument("--mask", required=True, help="Path to NIfTI mask (.nii/.nii.gz)")
    p.add_argument("--config", required=True, help="Path to PyRadiomics YAML")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--spacing", nargs=3, type=float, default=[1.0,1.0,1.0],
                   help="Resampled voxel spacing, e.g. --spacing 1 1 1")
    p.add_argument("--bias-correct", action="store_true",
                   help="Apply N4 bias correction (MRI). Skip for CT.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: ORIENTATION STANDARDIZATION
    # -------------------------------------------------------------------------
    # Make RAS-oriented copies so originals stay untouched
    img_ras = os.path.join(args.outdir, "img_ras.nii.gz")
    msk_ras = os.path.join(args.outdir, "mask_ras.nii.gz")
    to_ras(args.image, img_ras)
    to_ras(args.mask, msk_ras)

    # -------------------------------------------------------------------------
    # STEP 2: BIAS FIELD CORRECTION (MRI ONLY)
    # -------------------------------------------------------------------------
    # Optional N4 (use for MRI, skip for CT which already has standardized HU)
    img_proc = img_ras
    if args.bias_correct:
        img_proc = os.path.join(args.outdir, "img_n4.nii.gz")
        n4_bias_correct(img_ras, img_proc)

    # -------------------------------------------------------------------------
    # STEP 4: SPATIAL RESAMPLING TO ISOTROPIC VOXELS
    # -------------------------------------------------------------------------
    # Resample both image & mask to fixed spacing (default 1x1x1 mm)
    # This also enforces STEP 5 (consistent mask alignment)
    resample_pair(img_proc, msk_ras, spacing=tuple(args.spacing))

    # -------------------------------------------------------------------------
    # STEPS 3, 5, 6: FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    # Extract features using the locked YAML configuration
    # - STEP 3: Intensity normalization (z-score for MRI, HU clipping for CT)
    # - STEP 5: Mask correction and cropping policy
    # - STEP 6: Gray-level discretization (binWidth for CT, binCount for MRI)
    feats = extract_features(img_proc, msk_ras, args.config)

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    # Save features in both JSON and CSV formats for downstream analysis
    json_path = os.path.join(args.outdir, "features.json")
    with open(json_path, "w") as f:
        json.dump(feats, f, indent=2)

    # Also write CSV (one row per feature)
    import csv
    csv_path = os.path.join(args.outdir, "features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "value"])
        for k, v in feats.items():
            w.writerow([k, v])

    print(f"Done.\nJSON: {json_path}\nCSV:  {csv_path}")

if __name__ == "__main__":
    main()
