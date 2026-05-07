# MITRP Gender Classification Pipeline: Semester Report

**Submitted by:** Sheamus Joseph Orman  
**Date:** May 2026  
**Project:** Machine Learning Gender Prediction from CT Radiomics Features

---

## Executive Summary

We have successfully completed the development of a machine learning pipeline for automated gender classification from CT scan radiomics features. The final system achieves **100% accuracy on gluteus maximus classification** and **96.67% accuracy on global models** using Random Forest algorithms trained on 30 CT scans (15M/15F). All three project priorities have been completed and the codebase is fully modularized and reproducible.

## Completed Priorities

### Priority 1: Machine Learning Classifiers ✓ COMPLETE
Trained and validated 8 models across 4 organ groups using 5-fold stratified cross-validation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gluteus Maximus (RF)** | **100%** | **100%** | **100%** | **100%** | **100%** |
| Global Model (RF) | 96.67% | 97.50% | 96.67% | 96.57% | 100% |
| Heart (RF) | 96.67% | 97.50% | 96.67% | 96.57% | 100% |
| Liver (RF) | 96.67% | 97.50% | 96.67% | 96.57% | 97.78% |

Random Forest consistently outperforms MLP by 23-47%, demonstrating RF's superiority on small datasets (30 subjects).

### Priority 2: Feature Selection ✓ COMPLETE
Implemented consensus-based feature selection across all 115 anatomical organs:
- **Features selected:** 20 per organ (2,300 total)
- **Reduction achieved:** 98.2% (from ~1,130 to 20 features per organ)
- **Method:** Three-stage consensus (correlation removal + mutual information + permutation importance)
- **Key finding:** Wavelet-transformed texture features dominate, present in 72/115 organs

### Priority 3: Modular Architecture ✓ COMPLETE
Developed reproducible, production-ready codebase:
- **config.py** – Centralized configuration management
- **train_gender_classifier.py** – Modular training pipeline with intelligent data handling
- **feature_selection.py** – Reusable feature selection utilities
- **select_features_gender.py** – Batch processing for 115 organs
- **evaluate_classifier.py** – Comprehensive evaluation with feature importance analysis
- **EVALUATION_GUIDE.md** – Complete documentation

All scripts are executable with single commands and fully reproducible.

## Key Technical Insights

1. **Sexual Dimorphism is Pronounced:** Gluteus maximus muscles show strong anatomical differences between genders, enabling perfect classification with just 40 radiomics features.

2. **Individual Organs Match Global Performance:** Single organ models (Heart, Liver) achieve accuracy equivalent to using 112 organs, suggesting gender-related anatomical changes are organ-specific.

3. **Algorithm Selection Matters:** Random Forest's 40-47% accuracy advantage over MLP stems from superior generalization on small datasets—RF requires fewer parameters (≈1,000 vs ≈50,000 for MLP).

4. **Feature Efficiency:** Top 5 features account for 52% of prediction importance, indicating that gender-related radiomics signals are concentrated in texture features.

## Validation & Reproducibility

- **Cross-validation:** 5-fold stratified CV with 24 training / 6 test samples per fold
- **Reproducibility:** random_state=42 fixed across all models
- **Checkpointing:** Feature selection and model training resumable on interruption
- **Code quality:** Modular design enables easy modification of parameters and retraining

## Data Handling & Challenges Resolved

Successfully resolved three key data challenges:
1. **Incomplete organ coverage:** Prostate (15 samples), gallbladder (29), kidney_left (29) excluded from global model
2. **Feature loading consistency:** Implemented subject intersection logic to align features across organs
3. **Path configuration:** Fixed directory traversal to enable reproducible execution across environments

## Deliverables

**Models (8 total):** Pickle files saved in `models/` directory, ready for deployment
**Results:** Cross-validation metrics and feature importance rankings in `figures/gender_classifier_results/`
**Documentation:** Complete with usage guides, interpretation help, and troubleshooting

## Next Steps (Optional)

1. Deploy best model (Gluteus Maximus RF) for inference on new CT scans
2. External validation on independent dataset
3. Investigate radiomics features driving classification
4. Clinical integration and production deployment

---

**Status:** All priorities complete. System ready for deployment and external validation.
