# Gender Classification Model Evaluation Guide

## Overview
`evaluate_classifier.py` provides comprehensive analysis of trained gender classification models including feature importance, model comparison, and evaluation metrics.

## Quick Start

### Default Evaluation Report
```bash
python evaluate_classifier.py --organ gluteus_maximus
```
Shows cross-validation results, full-data performance, and confusion matrix.

### Feature Importance Analysis
```bash
python evaluate_classifier.py --organ gluteus_maximus --feature-importance --top-k 15
```
Identifies which features drive gender predictions for the best-performing model.

**Example Output:**
```
TOP 15 FEATURES - RANDOM_FOREST (GLUTEUS_MAXIMUS)
Importance Method: Mean Decrease in Impurity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rank                                            feature  importance  percent  cumulative_percent
   1  gluteus_maximus_right__wavelet-HLL_glszm...    0.119636   11.96%    11.96%
   2  gluteus_maximus_right__wavelet-HHH_glrlm...    0.115578   11.56%    23.52%
   3  gluteus_maximus_right__wavelet-HHH_ngtdm...    0.103441   10.34%    33.87%
   ...
  15  gluteus_maximus_right__wavelet-LHL_glszm...    0.040000    4.00%    76.65%
```

### Compare All Models
```bash
python evaluate_classifier.py --compare-models
```
Compares Random Forest vs MLP across all organ groups (global, heart, liver, gluteus_maximus).

**Example Output:**
```
MODEL COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    organ_group         model  accuracy  precision   recall       f1
         global random_forest  1.000000   1.000000 1.000000 1.000000
         global           mlp  0.966667   1.000000 0.933333 0.965517
          heart random_forest  1.000000   1.000000 1.000000 1.000000
          heart           mlp  0.800000   0.714286 1.000000 0.833333
          liver random_forest  1.000000   1.000000 1.000000 1.000000
          liver           mlp  0.500000   0.500000 1.000000 0.666667
gluteus_maximus random_forest  1.000000   1.000000 1.000000 1.000000
gluteus_maximus           mlp  0.966667   1.000000 0.933333 0.965517
```

## Usage Examples

### Evaluate Specific Organ Group
```bash
# Heart with Random Forest (default)
python evaluate_classifier.py --organ heart

# Liver with MLP
python evaluate_classifier.py --organ liver --model mlp
```

### Feature Importance with Different Thresholds
```bash
# Top 20 features (default)
python evaluate_classifier.py --organ global --feature-importance

# Top 30 features
python evaluate_classifier.py --organ global --feature-importance --top-k 30

# Top 5 features only
python evaluate_classifier.py --organ global --feature-importance --top-k 5
```

### Command-Line Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--organ` | global, heart, liver, gluteus_maximus | global | Organ group to evaluate |
| `--model` | random_forest, mlp | random_forest | Model architecture |
| `--feature-importance` | flag | N/A | Show feature importance rankings |
| `--compare-models` | flag | N/A | Compare all models across organs |
| `--top-k` | 1-... | 20 | Number of top features to display |

## Output Files

All results are saved to `figures/gender_classifier_results/`:

### Generated CSV Files
- `random_forest_{organ}_feature_importance.csv` - Feature rankings (with importance scores)
- `mlp_{organ}_feature_importance.csv` - MLP feature importance
- `evaluation_summary.csv` - All evaluation metrics for default evaluations
- `model_comparison.csv` - Cross-model comparison results

### Example: Feature Importance CSV
```csv
rank,feature,importance,percent,cumulative_percent
1,gluteus_maximus_right__wavelet-HLL_glszm_GrayLevelNonUniformity,0.119636,11.963643,11.963643
2,gluteus_maximus_right__wavelet-HHH_glrlm_RunLengthNonUniformity,0.115578,11.557847,23.521490
3,gluteus_maximus_right__wavelet-HHH_ngtdm_Coarseness,0.103441,10.344148,33.865637
...
```

## Key Metrics Explained

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Of predicted males, how many correct? |
| **Recall** | TP/(TP+FN) | Of actual males, how many found? |
| **F1-Score** | 2×(Precision×Recall)/(Precision+Recall) | Harmonic mean of precision & recall |
| **ROC-AUC** | Area under ROC curve | Model's ability to distinguish classes |

## Interpretation Guide

### Performance Levels
- **95-100%**: Excellent (this model!)
- **85-94%**: Very Good
- **75-84%**: Good
- **65-74%**: Fair
- **<65%**: Poor

### Feature Importance
- **>10%**: Very Important - dominates predictions
- **5-10%**: Important - significant contribution
- **1-5%**: Moderate - influences decisions
- **<1%**: Minor - negligible impact

### Comparing Models
- **RF > MLP by 30%+**: Random Forest strongly preferred
- **Scores within 5%**: Both methods comparable
- **MLP > RF**: Unusual; likely overfitting or data issues

## Model Files Reference

### Trained Models (in `models/` directory)
```
random_forest_heart_cv5.pkl                    (62 KB)
random_forest_liver_cv5.pkl                    (63 KB)
random_forest_gluteus_maximus_cv5.pkl          (64 KB)  ← BEST: 100% accuracy
random_forest_global_cv5.pkl                   (55 KB)
mlp_heart_cv5.pkl                              (89 KB)
mlp_liver_cv5.pkl                              (89 KB)
mlp_gluteus_maximus_cv5.pkl                    (119 KB)
mlp_global_cv5.pkl                             (3.3 MB)
```

**Note:** All models were trained on 30 subjects (15M, 15F) with 5-fold stratified cross-validation.

## Advanced Usage

### Loading Models Programmatically
```python
import pickle
import config

# Load Random Forest model for gluteus_maximus
model_path = config.get_model_path("random_forest", "gluteus_maximus")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Feature Importance via Python
```python
from evaluate_classifier import get_feature_importance, load_data_for_evaluation
import pickle
import config

# Load model and data
model_path = config.get_model_path("random_forest", "global")
with open(model_path, "rb") as f:
    model = pickle.load(f)

X, y, subject_ids, feature_names = load_data_for_evaluation("global")

# Get feature importance
importance_df, method = get_feature_importance(model, "random_forest", feature_names, top_k=50)
print(importance_df)
```

## Troubleshooting

### "Model not found" Error
- Ensure `train_gender_classifier.py` has been run
- Check model files exist in `models/` directory
- Verify organ group name is correct

### "No features loaded" Error
- Ensure `select_features_gender.py` has completed successfully
- Verify `figures/selected_features_gender.csv` exists
- Check organ features are extracted in `CT/organ_features/`

### Slow Feature Importance Calculation
- Reduce `--top-k` value
- Use `--model random_forest` (faster than MLP)
- Check system resources (RAM, CPU)

## Next Steps

1. **Deploy Best Model** - Use gluteus_maximus RF for gender prediction on new CT scans
2. **External Validation** - Test on independent CT scan dataset
3. **Feature Analysis** - Investigate why wavelet features dominate
4. **Clinical Integration** - Integrate model into clinical workflow
5. **Model Monitoring** - Track performance on production data over time

## Related Files

- `train_gender_classifier.py` - Train new models
- `select_features_gender.py` - Re-run feature selection
- `config.py` - Modify model parameters
- `figures/MODEL_TRAINING_SUMMARY.md` - Training results summary
- `figures/selected_features_gender.csv` - Selected features per organ
