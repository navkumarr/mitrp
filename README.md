# MIT Radiomics Playground

Deterministic pipeline for going from a single abdominal CT volume to multi-organ segmentations and PyRadiomics feature sets. Only lightweight demo volumes (`example.nii` / `example.nii.gz`) are tracked in git; drop your own scans locally to explore the full workflow.

## Repository Layout
- `example.nii(.gz)`: toy CT volume that can be segmented and featurized end-to-end.
- `segmentations/`: organ masks exported by [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). Each `.nii.gz` corresponds to one anatomical label (ignored by git by default).
- `pyradiomics/`: reproducible feature extraction scripts, configs (`params.yaml`, `pyrads.yaml`), and sample outputs (`out/`).
- `mitrpENV/`: optional local virtual environment with all clinical imaging dependencies pre-installed (ignored by git).

## Prerequisites
1. Python 3.11+
2. GPU optional, CPU mode works for the included sample
3. Core packages:
   ```bash
   python3 -m venv mitrpENV
   source mitrpENV/bin/activate
   pip install --upgrade pip
   pip install totalsegmentator SimpleITK nibabel pyradiomics numpy pandas
   ```
   - `TotalSegmentator` pulls in `nnUNetv2` and will prompt to download pretrained weights on first run.
   - Radiomics scripts were validated with PyRadiomics 3.1.0, SimpleITK 2.3.1, NumPy 1.26.4.

## End-to-End Workflow
1. **Run TotalSegmentator** on any CT volume (sample uses `example.nii.gz`).
   ```bash
   TotalSegmentator \
     -i example.nii.gz \
     -o segmentations \
     --roi_subset lung_left lung_right heart aorta
   ```
   - Omit `--roi_subset` to export every available label (>100 masks) into `segmentations/`.
   - Add `--fast` for a faster (slightly lower fidelity) pass, or `--preview` to sanity-check orientation.

2. **Extract radiomics features** for a specific organ mask.
   ```bash
   python pyradiomics/extract_radiomics.py \
     --image example.nii.gz \
     --mask segmentations/lung_left.nii.gz \
     --config pyradiomics/pyrads.yaml \
     --outdir outputs/lung_left \
     --bias-correct
   ```
   - Pipeline enforces RAS orientation, optional N4 bias correction, isotropic resampling, intensity normalization, and locked discretization before invoking PyRadiomics.
   - Results land in `outputs/lung_left/features.(json|csv)` plus the intermediate RAS / N4 volumes.

3. **Minimal reproducible sample**: `python pyradiomics/main.py` runs PyRadiomics with `params.yaml` against `example.nii.gz` and `segmentations/lung_left.nii.gz` to produce a quick feature dictionary.

4. **Inspect / analyze**: `pyradiomics/lung_left_data.csv` contains a single-run dump that mirrors what the JSON/CSV exporter emits for the curated sample.

## Working With Your Own Data
- Drop new CT volumes anywhere (commonly alongside `example.nii.gz`). All `.nii.gz` files except the example volume are ignored by git, so you can safely keep PHI data local.
- Store TotalSegmentator outputs in `segmentations/` to align with the provided commands, or point the `--mask` argument to another folder.
- Update `pyradiomics/pyrads.yaml` if you need different voxel spacing, bin widths, or feature families; the docstrings inside `extract_radiomics.py` describe every standardized step.

## Troubleshooting
- **Weights download**: If `TotalSegmentator` cannot find weights, run `TotalSegmentator --download_dataset` or follow the CLI prompt to fetch them.
- **Memory**: Reduce label set with `--roi_subset`, or downsample via `--fast` to stay within 8 GB GPUs / 32 GB RAM.
- **PyRadiomics config drift**: Always tie runs to a specific YAML (commit the config, not the derived data) to keep experiments reproducible.

## Next Steps
- Benchmark multiple organs (e.g., liver + pancreas) and aggregate CSVs for ML modeling.
- Replace the sample CT with institution-specific scans; copy any non-PHI slices into the repo only if they are cleared for release.
- Extend `extract_radiomics.py` with cohort looping or Pandas export if you want to process batches automatically.
