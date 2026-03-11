# Classification Modeling Skill

A comprehensive skill set for building, evaluating, and deploying classification models using machine learning best practices.

## Overview

This skill provides a complete workflow for classification modeling tasks, including:

- **Binary Classification** (yes/no, spam/not spam)
- **Multiclass Classification** (category A, B, C...)
- **Alert Correlation/Grouping**
- **Predictive Modeling for Categorical Outcomes**

## Contents

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill definition with complete workflow, patterns, and checklist |
| `classification_agent.py` | Python script for automated classification pipeline |
| `Classification_Modeling.ipynb` | Jupyter notebook template with interactive workflow |
| `README.md` | This documentation file |

## 12-Step Workflow

1. **Data Preparation** - Load, inspect, validate target variable
2. **Feature Engineering** - Encode categorical, scale numerical features
3. **Train/Test Split** - Stratified split to preserve class ratios
4. **Handle Class Imbalance** - SMOTE, class weights, undersampling
5. **Model Selection & Training** - Train multiple classifiers
6. **Cross-Validation** - K-fold CV for robust evaluation
7. **Hyperparameter Tuning** - GridSearchCV, RandomizedSearchCV
8. **Model Evaluation** - Comprehensive metrics and visualizations
9. **Feature Importance** - Understand key predictors
10. **Model Comparison** - Compare and rank models
11. **Model Persistence** - Save models and preprocessing pipeline
12. **Summary Report** - Document findings and recommendations

## Quick Start

### Using the Jupyter Notebook

1. Open `Classification_Modeling.ipynb`
2. Update the configuration cell:
   ```python
   DATA_FILE = "your_data.csv"
   TARGET_COLUMN = "your_target_column"
   ```
3. Run all cells sequentially

### Using the Python Agent

```bash
# Basic usage
python classification_agent.py "data.csv" --target "label"

# With hyperparameter tuning
python classification_agent.py "data.csv" --target "label" --tune

# Specify output directory
python classification_agent.py "data.csv" --target "label" --output-dir ./models

# Select specific models
python classification_agent.py "data.csv" --target "label" --models random_forest xgboost
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target` | Target column name | Required |
| `--output-dir` | Output directory | `./classification_results` |
| `--test-size` | Test set proportion | 0.2 |
| `--models` | Models to train | All available |
| `--tune` | Enable hyperparameter tuning | False |
| `--cv-folds` | Cross-validation folds | 5 |
| `--no-imbalance` | Disable imbalance handling | False |
| `--no-plots` | Disable plot generation | False |

## Supported Classifiers

| Model | Key |
|-------|-----|
| Logistic Regression | `logistic_regression` |
| Random Forest | `random_forest` |
| Gradient Boosting | `gradient_boosting` |
| XGBoost | `xgboost` |
| LightGBM | `lightgbm` |
| Decision Tree | `decision_tree` |
| SVM | `svm` |
| KNN | `knn` |
| Naive Bayes | `naive_bayes` |
| AdaBoost | `adaboost` |

## Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Accuracy | ≥80% | Baseline metric |
| Precision | ≥85% | When false positives are costly |
| Recall | ≥75% | When false negatives are costly |
| F1-Score | ≥80% | Balanced measure |
| ROC-AUC | ≥0.85 | Overall discriminative ability |

## Output Files

The skill generates the following outputs:

- `best_model_*.joblib` - Trained best model
- `all_models_*.joblib` - All trained models
- `scaler_*.joblib` - Fitted StandardScaler
- `label_encoder_*.joblib` - Fitted LabelEncoder (if used)
- `feature_names_*.txt` - List of feature names
- `classification_report_*.json` - Complete analysis report
- `model_comparison_*.csv` - Model metrics comparison
- `confusion_matrix_*.png` - Confusion matrix visualization
- `roc_curves_*.png` - ROC curve comparison
- `feature_importance_*.png` - Top feature importances
- `model_comparison_*.png` - Model metrics bar chart

## Requirements

### Required
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

### Optional (Enhanced Features)
```
xgboost          # For XGBoost classifier
lightgbm         # For LightGBM classifier
imbalanced-learn # For SMOTE and other resampling
shap             # For SHAP feature importance
```

### Install All
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm imbalanced-learn
```

## Best Practices Enforced

### ✅ Required Patterns
- Stratified train/test split
- Scale features after split (no data leakage)
- Cross-validation for robust evaluation
- Multiple metrics beyond accuracy
- Model comparison before selection
- Save preprocessing pipeline with model

### ❌ Forbidden Patterns
- Training and testing on same data
- Applying SMOTE before train/test split
- Fitting scaler on entire dataset
- Only reporting accuracy
- No cross-validation
- Not saving trained models

## Example Usage in Code

```python
from classification_agent import ClassificationAgent

# Initialize agent
agent = ClassificationAgent(
    filepath="customer_churn.csv",
    target_column="churned",
    output_dir="./churn_model"
)

# Run full pipeline
results = agent.run(
    models=['random_forest', 'xgboost', 'logistic_regression'],
    handle_imbalance=True,
    tune_hyperparameters=True,
    cv_folds=5
)

# Make predictions on new data
predictions = agent.predict(new_customer_data)
```

## License

MIT License

## Version

1.0.0 - Initial Release
