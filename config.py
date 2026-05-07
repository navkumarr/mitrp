#!/usr/bin/env python3
"""
Centralized configuration for the mitrp gender classification pipeline.

Removes hardcoding of paths, model parameters, and settings.
All modules should import from this file.
"""
import os

# ============================================================================
# DIRECTORIES
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # mitrp/ directory
DATA_DIR = os.path.join(BASE_DIR, "CT")
ORGAN_FEATURES_DIR = os.path.join(DATA_DIR, "organ_features")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for d in [FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
LABELS_CSV = os.path.join(DATA_DIR, "First30.csv")
SELECTED_FEATURES_CSV = os.path.join(FIGURES_DIR, "selected_features_gender.csv")
RESULTS_DIR = os.path.join(FIGURES_DIR, "gender_classifier_results")

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}

# Multi-Layer Perceptron
MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "max_iter": 1000,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
    "solver": "adam",
    "activation": "relu",
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
}

# Cross-validation
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
TOP_K_FEATURES = 20  # Features per organ to use from feature selection
FEATURE_SELECTION_METHOD = "consensus"  # How features were selected
MODELS_TO_TRAIN = ["random_forest", "mlp"]  # Which models to train

# ============================================================================
# ORGAN GROUPS FOR ANALYSIS
# ============================================================================
# Global model uses ALL organs
# Organ-specific models use individual organs

ORGAN_GROUPS = {
    "global": "ALL_ORGANS",  # Placeholder - will be determined by selected_features CSV
    "heart": ["heart"],
    "liver": ["liver"],
    "gluteus_maximus": ["gluteus_maximus_left", "gluteus_maximus_right"],
    "_exclude_organs": ["gallbladder", "kidney_left", "prostate"],  # Organs with incomplete data across subjects
}

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================
VERBOSE = True
SAVE_MODELS = True
SAVE_PREDICTIONS = True

# Model output format
MODEL_FILENAME_FORMAT = "{model_type}_{organ_group}_cv{cv_folds}.pkl"
RESULTS_FILENAME_FORMAT = "{model_type}_{organ_group}_results.csv"
PREDICTIONS_FILENAME_FORMAT = "{model_type}_{organ_group}_predictions.csv"

# ============================================================================
# VALIDATION METRICS
# ============================================================================
# Which metrics to compute for each model
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "confusion_matrix"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_path(model_type, organ_group, cv_folds=CV_FOLDS):
    """Get full path for saving a trained model."""
    filename = MODEL_FILENAME_FORMAT.format(
        model_type=model_type,
        organ_group=organ_group,
        cv_folds=cv_folds
    )
    return os.path.join(MODELS_DIR, filename)


def get_results_path(model_type, organ_group):
    """Get full path for saving results CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = RESULTS_FILENAME_FORMAT.format(
        model_type=model_type,
        organ_group=organ_group
    )
    return os.path.join(RESULTS_DIR, filename)


def get_predictions_path(model_type, organ_group):
    """Get full path for saving predictions CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = PREDICTIONS_FILENAME_FORMAT.format(
        model_type=model_type,
        organ_group=organ_group
    )
    return os.path.join(RESULTS_DIR, filename)


def print_config():
    """Print current configuration for verification."""
    print("\n" + "="*70)
    print("MITRP GENDER CLASSIFIER CONFIGURATION")
    print("="*70)
    print(f"\nDIRECTORIES:")
    print(f"  Base:             {BASE_DIR}")
    print(f"  Data:             {DATA_DIR}")
    print(f"  Organ Features:   {ORGAN_FEATURES_DIR}")
    print(f"  Figures:          {FIGURES_DIR}")
    print(f"  Models:           {MODELS_DIR}")
    print(f"\nDATA:")
    print(f"  Labels CSV:       {LABELS_CSV}")
    print(f"  Selected Features:{SELECTED_FEATURES_CSV}")
    print(f"\nMODELS TO TRAIN:")
    for m in MODELS_TO_TRAIN:
        print(f"  - {m}")
    print(f"\nCROSS-VALIDATION:")
    print(f"  Folds:            {CV_FOLDS}")
    print(f"  Test Size:        {TEST_SIZE}")
    print(f"\nOPTIMIZATION:")
    print(f"  Features/organ:   {TOP_K_FEATURES}")
    print(f"  Random state:     {RANDOM_STATE}")
    print(f"\nOUTPUT:")
    print(f"  Results dir:      {RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
