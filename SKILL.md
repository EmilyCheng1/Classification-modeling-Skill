---
name: Classification Modeling Skill
description: Use when building ANY classification model, binary or multiclass classifier, or predictive model for categorical targets. MANDATORY for supervised learning with discrete outcomes. FORBIDDEN to skip train/test split, ignore class imbalance, or deploy without evaluation metrics. REQUIRED patterns include cross-validation, feature importance analysis, and model comparison.
---

# Classification Modeling Skill

## THE MANDATE

When building classification models for ML projects, you MUST follow this comprehensive workflow to ensure robust, production-ready classifiers with proper evaluation and interpretability.

## SCOPE & APPLICABILITY

| Use Case | Applies |
|----------|---------|
| Binary classification (yes/no, spam/not spam) | ✅ YES |
| Multiclass classification (category A, B, C...) | ✅ YES |
| Alert correlation/grouping | ✅ YES |
| Anomaly detection (as binary) | ✅ YES |
| Regression (continuous output) | ❌ NO |
| Clustering (unsupervised) | ❌ NO |

## SUCCESS CRITERIA

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Accuracy | ≥80% | Baseline, less reliable for imbalanced data |
| Precision | ≥85% | Critical when false positives are costly |
| Recall | ≥75% | Critical when false negatives are costly |
| F1-Score | ≥80% | Balanced measure |
| ROC-AUC | ≥0.85 | Overall discriminative ability |
| False Positive Rate | ≤10% | Business requirement dependent |

## CLASSIFICATION WORKFLOW

### Step 1: Data Preparation
- Load and inspect dataset
- Identify target variable (must be categorical)
- Check class distribution (detect imbalance)
- Handle missing values (imputation or removal)
- Remove or handle outliers
- Feature selection (remove irrelevant columns)

### Step 2: Feature Engineering
- Encode categorical features (One-Hot, Label, Target encoding)
- Scale numerical features (StandardScaler, MinMaxScaler)
- Create derived features if domain appropriate
- Handle datetime features (extract components)
- Apply dimensionality reduction if needed (PCA)

### Step 3: Train/Test Split
- Split data with stratification (preserve class ratios)
- Use 70-30 or 80-20 split typically
- Set random_state for reproducibility
- Consider time-based split for temporal data

### Step 4: Handle Class Imbalance
- Identify imbalance ratio (minority/majority)
- Apply appropriate technique:
  - SMOTE (Synthetic Minority Oversampling)
  - Random undersampling
  - Class weights in model
  - ADASYN for adaptive synthetic sampling
- ONLY apply to training data, never test data

### Step 5: Model Selection & Training
Train multiple classifiers for comparison:

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| Logistic Regression | Baseline, interpretable | Fast, probabilistic | Linear boundaries only |
| Random Forest | General purpose | Handles non-linear, feature importance | Can overfit |
| XGBoost/LightGBM | High performance | State-of-the-art accuracy | Requires tuning |
| SVM | High-dimensional data | Effective in high dimensions | Slower on large datasets |
| Neural Network | Complex patterns | Flexible | Needs lots of data |
| Naive Bayes | Text classification | Fast, works with small data | Independence assumption |

### Step 6: Cross-Validation
- Use k-fold cross-validation (k=5 or k=10)
- Apply stratified k-fold for imbalanced data
- Report mean and std of metrics across folds
- Use cross_val_score or cross_validate

### Step 7: Hyperparameter Tuning
- Define parameter grid for each model
- Use GridSearchCV or RandomizedSearchCV
- Optimize for appropriate metric (F1 for imbalanced)
- Use early stopping for gradient boosting

### Step 8: Model Evaluation
Generate comprehensive evaluation:
- Confusion matrix (visualize with heatmap)
- Classification report (precision, recall, F1 per class)
- ROC curve and AUC score
- Precision-Recall curve (for imbalanced data)
- Learning curves (detect overfitting)

### Step 9: Feature Importance Analysis
- Extract feature importances from tree models
- Use SHAP values for interpretation
- Apply permutation importance for any model
- Document top contributing features

### Step 10: Model Comparison & Selection
- Compare all models on test set
- Select best model based on business criteria
- Document trade-offs (speed vs accuracy)
- Consider ensemble of top models

### Step 11: Model Persistence
- Save best model using joblib or pickle
- Save preprocessing pipeline (scaler, encoder)
- Export feature names and importance
- Version the model with timestamp
- Create inference function

### Step 12: Summary Report
- Document final model selection
- Report all evaluation metrics
- Include feature importance visualization
- Provide recommendations for deployment
- Generate PDF report

## REQUIRED OUTPUTS

1. **Trained model file** (.joblib or .pkl)
2. **Preprocessing pipeline** (scaler, encoders)
3. **Evaluation report (PDF)** with all metrics and visualizations
4. **Model comparison table**
5. **Feature importance visualization**
6. **Confusion matrix heatmap**
7. **ROC/PR curves**

## FORBIDDEN PATTERNS

### ❌ No Train/Test Split - BANNED
```python
# ❌ FORBIDDEN - Training and evaluating on same data
model.fit(X, y)
accuracy = model.score(X, y)  # Overly optimistic!

# ✅ REQUIRED - Proper split with stratification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
```

### ❌ Ignoring Class Imbalance - BANNED
```python
# ❌ FORBIDDEN - Training on imbalanced data without handling
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ REQUIRED - Handle imbalance appropriately
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train_balanced, y_train_balanced)
```

### ❌ Only Reporting Accuracy - BANNED
```python
# ❌ FORBIDDEN - Single metric evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# ✅ REQUIRED - Comprehensive evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")
print(confusion_matrix(y_test, y_pred))
```

### ❌ No Cross-Validation - BANNED
```python
# ❌ FORBIDDEN - Single train/test evaluation
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# ✅ REQUIRED - Cross-validation for robust estimate
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
print(f"CV F1: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### ❌ Data Leakage from Test Set - BANNED
```python
# ❌ FORBIDDEN - Fitting scaler on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leaks test statistics!
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ REQUIRED - Fit only on training data
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform only, no fit!
```

### ❌ Applying SMOTE to Entire Dataset - BANNED
```python
# ❌ FORBIDDEN - SMOTE before split
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_resampled, ...)  # Synthetic data in test!

# ✅ REQUIRED - SMOTE only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Test set remains untouched with original distribution
```

### ❌ No Model Persistence - BANNED
```python
# ❌ FORBIDDEN - Training without saving
model.fit(X_train, y_train)
# Model lost when script ends!

# ✅ REQUIRED - Save model for deployment
import joblib
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(model, f'best_classifier_{timestamp}.joblib')
joblib.dump(scaler, f'scaler_{timestamp}.joblib')
```

## REQUIRED PATTERNS

### ✅ Complete Classification Pipeline
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load and prepare data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Encode categorical features
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 4. Scale features (fit only on train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Handle imbalance on training data only
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

# 8. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
print(f"CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# 9. Save model
joblib.dump(model, 'classifier_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(le, 'label_encoder.joblib')
```

### ✅ Multi-Model Comparison
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'SVM': SVC(probability=True)
}

results = []
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

comparison_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(comparison_df)
```

### ✅ Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_balanced, y_train_balanced)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
best_model = grid_search.best_estimator_
```

### ✅ Feature Importance Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# For tree-based models
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
```

### ✅ Confusion Matrix Visualization
```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    display_labels=le.classes_,
    cmap='Blues',
    ax=ax
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
```

### ✅ ROC Curve Visualization
```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
```

## CHECKLIST

- [ ] Step 1: Data loaded and inspected, target variable identified
- [ ] Step 2: Features engineered, encoded, and scaled
- [ ] Step 3: Train/test split with stratification applied
- [ ] Step 4: Class imbalance handled (if applicable)
- [ ] Step 5: Multiple models trained
- [ ] Step 6: Cross-validation performed
- [ ] Step 7: Hyperparameter tuning completed
- [ ] Step 8: Comprehensive evaluation metrics generated
- [ ] Step 9: Feature importance analyzed
- [ ] Step 10: Best model selected with justification
- [ ] Step 11: Model and pipeline saved
- [ ] Step 12: Summary report generated with visualizations

## MULTICLASS CONSIDERATIONS

For multiclass classification (3+ classes):

```python
# Use appropriate averaging for metrics
from sklearn.metrics import f1_score, precision_score, recall_score

# Macro: treats all classes equally
f1_macro = f1_score(y_test, y_pred, average='macro')

# Weighted: accounts for class imbalance
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Micro: aggregates contributions of all classes
f1_micro = f1_score(y_test, y_pred, average='micro')

# For ROC-AUC with multiclass
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_pred_proba = model.predict_proba(X_test_scaled)
roc_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr')
```

## IMBALANCED DATA STRATEGIES

| Imbalance Ratio | Recommended Approach |
|-----------------|---------------------|
| < 1:5 | Class weights |
| 1:5 to 1:20 | SMOTE + Class weights |
| > 1:20 | Combine undersampling + SMOTE |
| Extreme | Consider anomaly detection approach |

---

**Created:** Auto-generated by Classification Modeling Skill
**Version:** 1.0
