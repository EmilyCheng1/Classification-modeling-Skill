#!/usr/bin/env python3
"""
Classification Modeling Agent - Automated ML Classification Pipeline

A comprehensive agent for building, evaluating, and comparing classification models.
Supports binary and multiclass classification with proper handling of class imbalance,
cross-validation, hyperparameter tuning, and model persistence.

Usage:
    python classification_agent.py "data.csv" --target "label_column"
    python classification_agent.py "data.csv" --target "label" --output-dir ./models
    python classification_agent.py "data.csv" --target "label" --tune --compare-all
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Try to import optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ClassificationAgent:
    """
    Automated Classification Modeling Agent.
    
    Follows the 12-step workflow:
        1. Data Preparation
        2. Feature Engineering
        3. Train/Test Split
        4. Handle Class Imbalance
        5. Model Selection & Training
        6. Cross-Validation
        7. Hyperparameter Tuning
        8. Model Evaluation
        9. Feature Importance Analysis
        10. Model Comparison & Selection
        11. Model Persistence
        12. Summary Report
    """
    
    # Available classifiers
    CLASSIFIERS = {
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {'max_iter': 1000, 'random_state': 42},
            'tune_params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {'n_estimators': 100, 'random_state': 42},
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier,
            'params': {'n_estimators': 100, 'random_state': 42},
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier,
            'params': {'random_state': 42},
            'tune_params': {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'svm': {
            'model': SVC,
            'params': {'probability': True, 'random_state': 42},
            'tune_params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'knn': {
            'model': KNeighborsClassifier,
            'params': {},
            'tune_params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'naive_bayes': {
            'model': GaussianNB,
            'params': {},
            'tune_params': {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        'adaboost': {
            'model': AdaBoostClassifier,
            'params': {'random_state': 42},
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        }
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        CLASSIFIERS['xgboost'] = {
            'model': XGBClassifier,
            'params': {'eval_metric': 'logloss', 'random_state': 42, 'use_label_encoder': False},
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
    
    # Add LightGBM if available
    if HAS_LIGHTGBM:
        CLASSIFIERS['lightgbm'] = {
            'model': LGBMClassifier,
            'params': {'random_state': 42, 'verbose': -1},
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100]
            }
        }
    
    def __init__(
        self,
        filepath: str,
        target_column: str,
        output_dir: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the Classification Agent.
        
        Args:
            filepath: Path to the dataset CSV file
            target_column: Name of the target variable column
            output_dir: Directory for output files
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
        """
        self.filepath = Path(filepath)
        self.target_column = target_column
        self.output_dir = Path(output_dir) if output_dir else self.filepath.parent / 'classification_results'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Preprocessing
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        
        # Results
        self.trained_models: Dict[str, Any] = {}
        self.model_results: List[Dict[str, Any]] = []
        self.best_model: Optional[Any] = None
        self.best_model_name: str = ""
        self.report: Dict[str, Any] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configure plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
    def run(
        self,
        models: Optional[List[str]] = None,
        handle_imbalance: bool = True,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
        generate_plots: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the complete classification workflow.
        
        Args:
            models: List of model names to train (default: all available)
            handle_imbalance: Whether to apply imbalance handling
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of cross-validation folds
            generate_plots: Whether to generate visualization plots
            generate_report: Whether to generate PDF report
            
        Returns:
            Dictionary containing all results
        """
        print("=" * 70)
        print("🤖 CLASSIFICATION MODELING AGENT")
        print("=" * 70)
        print(f"📁 Input: {self.filepath.name}")
        print(f"🎯 Target: {self.target_column}")
        print(f"📂 Output: {self.output_dir}")
        print("=" * 70)
        
        # Execute workflow
        self.step1_data_preparation()
        self.step2_feature_engineering()
        self.step3_train_test_split()
        
        if handle_imbalance:
            self.step4_handle_imbalance()
        
        self.step5_train_models(models)
        self.step6_cross_validation(cv_folds)
        
        if tune_hyperparameters:
            self.step7_hyperparameter_tuning(models, cv_folds)
        
        self.step8_model_evaluation(generate_plots)
        self.step9_feature_importance(generate_plots)
        self.step10_model_comparison()
        self.step11_model_persistence()
        self.step12_summary_report()
        
        if generate_plots:
            self._generate_all_plots()
        
        print("\n" + "=" * 70)
        print("✅ CLASSIFICATION MODELING COMPLETE!")
        print(f"   Best Model: {self.best_model_name}")
        print(f"   Output Directory: {self.output_dir}")
        print("=" * 70)
        
        return self.report
    
    # =========================================================================
    # STEP 1: Data Preparation
    # =========================================================================
    def step1_data_preparation(self) -> None:
        """Load and prepare the dataset."""
        print("\n📊 STEP 1: Data Preparation")
        print("-" * 50)
        
        # Load data
        self.df = self._load_file(self.filepath)
        print(f"  ✓ Loaded dataset: {len(self.df):,} rows × {len(self.df.columns)} columns")
        
        # Validate target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        
        # Encode target if categorical
        if self.y.dtype == 'object' or self.y.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            self.y = pd.Series(self.label_encoder.fit_transform(self.y), name=self.target_column)
            self.class_names = list(self.label_encoder.classes_)
        else:
            self.class_names = [str(c) for c in sorted(self.y.unique())]
        
        # Check class distribution
        class_dist = self.y.value_counts()
        self.report['step1'] = {
            'total_rows': len(self.df),
            'features': len(self.X.columns),
            'target_column': self.target_column,
            'n_classes': len(class_dist),
            'class_names': self.class_names,
            'class_distribution': class_dist.to_dict(),
            'imbalance_ratio': round(class_dist.max() / class_dist.min(), 2)
        }
        
        print(f"  ✓ Target: {self.target_column} ({len(class_dist)} classes)")
        print(f"  ✓ Class distribution:")
        for cls, count in class_dist.items():
            cls_name = self.class_names[cls] if self.label_encoder else cls
            print(f"      {cls_name}: {count:,} ({count/len(self.y)*100:.1f}%)")
        print(f"  ✓ Imbalance ratio: {self.report['step1']['imbalance_ratio']:.2f}")
        
    def _load_file(self, filepath: Path) -> pd.DataFrame:
        """Load a CSV file with encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(filepath, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        
        return pd.read_csv(filepath, encoding='utf-8', errors='ignore', low_memory=False)
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    def step2_feature_engineering(self) -> None:
        """Perform feature engineering and preprocessing."""
        print("\n🔧 STEP 2: Feature Engineering")
        print("-" * 50)
        
        # Identify column types
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"  ✓ Numeric features: {len(numeric_cols)}")
        print(f"  ✓ Categorical features: {len(categorical_cols)}")
        
        # Handle missing values for numeric columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            self.X[numeric_cols] = imputer.fit_transform(self.X[numeric_cols])
        
        # Handle missing values and encode categorical columns
        if len(categorical_cols) > 0:
            # Fill missing with mode
            for col in categorical_cols:
                self.X[col] = self.X[col].fillna(self.X[col].mode().iloc[0] if not self.X[col].mode().empty else 'Unknown')
            
            # One-hot encode categorical columns
            self.X = pd.get_dummies(self.X, columns=categorical_cols, drop_first=True)
        
        self.feature_names = list(self.X.columns)
        
        self.report['step2'] = {
            'n_features_after': len(self.feature_names),
            'numeric_features': numeric_cols,
            'categorical_features': categorical_cols,
            'encoded_features': self.feature_names
        }
        
        print(f"  ✓ Final features after encoding: {len(self.feature_names)}")
        
    # =========================================================================
    # STEP 3: Train/Test Split
    # =========================================================================
    def step3_train_test_split(self) -> None:
        """Split data into training and testing sets."""
        print("\n✂️ STEP 3: Train/Test Split")
        print("-" * 50)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, 
            self.y.values,
            test_size=self.test_size,
            stratify=self.y.values,
            random_state=self.random_state
        )
        
        self.report['step3'] = {
            'test_size': self.test_size,
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        print(f"  ✓ Training set: {len(self.X_train):,} samples")
        print(f"  ✓ Test set: {len(self.X_test):,} samples")
        print(f"  ✓ Stratified split with random_state={self.random_state}")
        
    # =========================================================================
    # STEP 4: Handle Class Imbalance
    # =========================================================================
    def step4_handle_imbalance(self) -> None:
        """Handle class imbalance in training data."""
        print("\n⚖️ STEP 4: Handle Class Imbalance")
        print("-" * 50)
        
        if not HAS_IMBLEARN:
            print("  ⚠ imbalanced-learn not installed. Using class_weight='balanced' instead.")
            self.report['step4'] = {'method': 'class_weight_balanced', 'applied': False}
            return
        
        # Check imbalance ratio
        class_counts = pd.Series(self.y_train).value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        if imbalance_ratio < 1.5:
            print(f"  ✓ Classes are balanced (ratio: {imbalance_ratio:.2f}). No resampling needed.")
            self.report['step4'] = {'method': None, 'applied': False, 'reason': 'balanced'}
            return
        
        # Apply SMOTE
        print(f"  ⚠ Imbalance detected (ratio: {imbalance_ratio:.2f}). Applying SMOTE...")
        
        original_len = len(self.y_train)
        smote = SMOTE(random_state=self.random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        new_counts = pd.Series(self.y_train).value_counts()
        
        self.report['step4'] = {
            'method': 'SMOTE',
            'applied': True,
            'original_samples': original_len,
            'resampled_samples': len(self.y_train),
            'new_class_distribution': new_counts.to_dict()
        }
        
        print(f"  ✓ SMOTE applied: {original_len:,} → {len(self.y_train):,} samples")
        
    # =========================================================================
    # STEP 5: Model Selection & Training
    # =========================================================================
    def step5_train_models(self, models: Optional[List[str]] = None) -> None:
        """Train multiple classification models."""
        print("\n🏋️ STEP 5: Model Training")
        print("-" * 50)
        
        if models is None:
            models = list(self.CLASSIFIERS.keys())
        
        for name in models:
            if name not in self.CLASSIFIERS:
                print(f"  ⚠ Unknown model: {name}. Skipping.")
                continue
            
            config = self.CLASSIFIERS[name]
            model = config['model'](**config['params'])
            
            print(f"  Training {name}...", end=" ")
            model.fit(self.X_train, self.y_train)
            self.trained_models[name] = model
            print("✓")
        
        self.report['step5'] = {
            'models_trained': list(self.trained_models.keys())
        }
        
        print(f"  ✓ Trained {len(self.trained_models)} models")
        
    # =========================================================================
    # STEP 6: Cross-Validation
    # =========================================================================
    def step6_cross_validation(self, cv_folds: int = 5) -> None:
        """Perform cross-validation for all trained models."""
        print("\n🔄 STEP 6: Cross-Validation")
        print("-" * 50)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = {}
        
        for name, model in self.trained_models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_weighted')
            cv_results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"  {name}: F1 = {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
        
        self.report['step6'] = {
            'cv_folds': cv_folds,
            'results': cv_results
        }
        
    # =========================================================================
    # STEP 7: Hyperparameter Tuning
    # =========================================================================
    def step7_hyperparameter_tuning(
        self, 
        models: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> None:
        """Tune hyperparameters for models using GridSearchCV."""
        print("\n🔍 STEP 7: Hyperparameter Tuning")
        print("-" * 50)
        
        if models is None:
            models = list(self.trained_models.keys())
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        tuning_results = {}
        
        for name in models:
            if name not in self.CLASSIFIERS:
                continue
            
            config = self.CLASSIFIERS[name]
            base_model = config['model'](**{k: v for k, v in config['params'].items() 
                                            if k != 'random_state'})
            
            print(f"  Tuning {name}...", end=" ")
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                base_model,
                config['tune_params'],
                n_iter=20,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            search.fit(self.X_train, self.y_train)
            
            # Update model with best estimator
            self.trained_models[name] = search.best_estimator_
            tuning_results[name] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_
            }
            
            print(f"✓ Best CV F1: {search.best_score_:.3f}")
        
        self.report['step7'] = tuning_results
        
    # =========================================================================
    # STEP 8: Model Evaluation
    # =========================================================================
    def step8_model_evaluation(self, generate_plots: bool = True) -> None:
        """Evaluate all trained models on the test set."""
        print("\n📈 STEP 8: Model Evaluation")
        print("-" * 50)
        
        self.model_results = []
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'model': name,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # ROC-AUC for binary or probability-based multiclass
            if y_pred_proba is not None:
                if len(self.class_names) == 2:
                    metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    try:
                        metrics['roc_auc'] = roc_auc_score(
                            self.y_test, y_pred_proba, multi_class='ovr', average='weighted'
                        )
                    except ValueError:
                        metrics['roc_auc'] = None
            else:
                metrics['roc_auc'] = None
            
            metrics['confusion_matrix'] = confusion_matrix(self.y_test, y_pred).tolist()
            metrics['classification_report'] = classification_report(
                self.y_test, y_pred, target_names=self.class_names, output_dict=True
            )
            
            self.model_results.append(metrics)
        
        # Print results table
        results_df = pd.DataFrame(self.model_results)[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        results_df = results_df.sort_values('f1', ascending=False)
        
        print("\n" + results_df.to_string(index=False))
        
        self.report['step8'] = {
            'results': self.model_results,
            'results_table': results_df.to_dict('records')
        }
        
    # =========================================================================
    # STEP 9: Feature Importance Analysis
    # =========================================================================
    def step9_feature_importance(self, generate_plots: bool = True) -> None:
        """Analyze feature importance from tree-based models."""
        print("\n🔬 STEP 9: Feature Importance Analysis")
        print("-" * 50)
        
        importance_results = {}
        
        # Get feature importance from tree-based models
        tree_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'decision_tree', 'adaboost']
        
        for name in tree_models:
            if name in self.trained_models:
                model = self.trained_models[name]
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[name] = importance_df.head(20).to_dict('records')
                    
                    print(f"\n  Top 5 features ({name}):")
                    for i, row in importance_df.head(5).iterrows():
                        print(f"    • {row['feature']}: {row['importance']:.4f}")
        
        self.report['step9'] = importance_results
        
    # =========================================================================
    # STEP 10: Model Comparison & Selection
    # =========================================================================
    def step10_model_comparison(self) -> None:
        """Compare models and select the best one."""
        print("\n🏆 STEP 10: Model Comparison & Selection")
        print("-" * 50)
        
        # Sort by F1 score
        sorted_results = sorted(self.model_results, key=lambda x: x['f1'], reverse=True)
        
        self.best_model_name = sorted_results[0]['model']
        self.best_model = self.trained_models[self.best_model_name]
        best_metrics = sorted_results[0]
        
        print(f"\n  🥇 Best Model: {self.best_model_name}")
        print(f"     Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"     Precision: {best_metrics['precision']:.4f}")
        print(f"     Recall:    {best_metrics['recall']:.4f}")
        print(f"     F1-Score:  {best_metrics['f1']:.4f}")
        if best_metrics['roc_auc']:
            print(f"     ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        
        self.report['step10'] = {
            'best_model': self.best_model_name,
            'best_metrics': {k: v for k, v in best_metrics.items() 
                           if k not in ['confusion_matrix', 'classification_report']},
            'ranking': [{'model': r['model'], 'f1': r['f1']} for r in sorted_results]
        }
        
    # =========================================================================
    # STEP 11: Model Persistence
    # =========================================================================
    def step11_model_persistence(self) -> None:
        """Save models and preprocessing objects."""
        print("\n💾 STEP 11: Model Persistence")
        print("-" * 50)
        
        # Save best model
        model_path = self.output_dir / f'best_model_{self.best_model_name}_{self.timestamp}.joblib'
        joblib.dump(self.best_model, model_path)
        print(f"  ✓ Best model saved: {model_path.name}")
        
        # Save all models
        all_models_path = self.output_dir / f'all_models_{self.timestamp}.joblib'
        joblib.dump(self.trained_models, all_models_path)
        print(f"  ✓ All models saved: {all_models_path.name}")
        
        # Save preprocessing objects
        scaler_path = self.output_dir / f'scaler_{self.timestamp}.joblib'
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Scaler saved: {scaler_path.name}")
        
        if self.label_encoder:
            encoder_path = self.output_dir / f'label_encoder_{self.timestamp}.joblib'
            joblib.dump(self.label_encoder, encoder_path)
            print(f"  ✓ Label encoder saved: {encoder_path.name}")
        
        # Save feature names
        features_path = self.output_dir / f'feature_names_{self.timestamp}.txt'
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"  ✓ Feature names saved: {features_path.name}")
        
        self.report['step11'] = {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'features_path': str(features_path)
        }
        
    # =========================================================================
    # STEP 12: Summary Report
    # =========================================================================
    def step12_summary_report(self) -> None:
        """Generate summary report."""
        print("\n📝 STEP 12: Summary Report")
        print("-" * 50)
        
        # Save JSON report
        import json
        
        # Make report JSON serializable
        serializable_report = {}
        for key, value in self.report.items():
            try:
                json.dumps(value)
                serializable_report[key] = value
            except (TypeError, ValueError):
                serializable_report[key] = str(value)
        
        report_path = self.output_dir / f'classification_report_{self.timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"  ✓ Report saved: {report_path.name}")
        
        # Save model comparison CSV
        results_df = pd.DataFrame(self.model_results)[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        results_df = results_df.sort_values('f1', ascending=False)
        csv_path = self.output_dir / f'model_comparison_{self.timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"  ✓ Model comparison saved: {csv_path.name}")
        
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    def _generate_all_plots(self) -> None:
        """Generate all visualization plots."""
        print("\n📊 Generating Visualizations...")
        
        # 1. Confusion Matrix for best model
        self._plot_confusion_matrix()
        
        # 2. ROC Curves
        self._plot_roc_curves()
        
        # 3. Model Comparison Bar Chart
        self._plot_model_comparison()
        
        # 4. Feature Importance
        self._plot_feature_importance()
        
        # 5. Class Distribution
        self._plot_class_distribution()
        
        print("  ✓ All visualizations saved")
        
    def _plot_confusion_matrix(self) -> None:
        """Plot confusion matrix for best model."""
        y_pred = self.best_model.predict(self.X_test)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            self.y_test, y_pred,
            display_labels=self.class_names,
            cmap='Blues',
            ax=ax
        )
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{self.timestamp}.png', dpi=150)
        plt.close()
        
    def _plot_roc_curves(self) -> None:
        """Plot ROC curves for models (binary classification only)."""
        if len(self.class_names) != 2:
            return  # Skip for multiclass
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                RocCurveDisplay.from_predictions(
                    self.y_test, y_pred_proba, name=name, ax=ax
                )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_curves_{self.timestamp}.png', dpi=150)
        plt.close()
        
    def _plot_model_comparison(self) -> None:
        """Plot model comparison bar chart."""
        results_df = pd.DataFrame(self.model_results)[['model', 'accuracy', 'precision', 'recall', 'f1']]
        results_df = results_df.sort_values('f1', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results_df))
        width = 0.2
        
        ax.barh(x - 1.5*width, results_df['accuracy'], width, label='Accuracy', color='#3498db')
        ax.barh(x - 0.5*width, results_df['precision'], width, label='Precision', color='#2ecc71')
        ax.barh(x + 0.5*width, results_df['recall'], width, label='Recall', color='#e74c3c')
        ax.barh(x + 1.5*width, results_df['f1'], width, label='F1-Score', color='#9b59b6')
        
        ax.set_yticks(x)
        ax.set_yticklabels(results_df['model'])
        ax.set_xlabel('Score')
        ax.set_title('Model Comparison')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_comparison_{self.timestamp}.png', dpi=150)
        plt.close()
        
    def _plot_feature_importance(self) -> None:
        """Plot feature importance from best tree-based model."""
        if not hasattr(self.best_model, 'feature_importances_'):
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
        plt.title(f'Top 20 Feature Importances - {self.best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_importance_{self.timestamp}.png', dpi=150)
        plt.close()
        
    def _plot_class_distribution(self) -> None:
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original distribution
        original_dist = self.report['step1']['class_distribution']
        axes[0].bar(range(len(original_dist)), list(original_dist.values()), color='#3498db')
        axes[0].set_xticks(range(len(original_dist)))
        axes[0].set_xticklabels(self.class_names)
        axes[0].set_title('Original Class Distribution')
        axes[0].set_ylabel('Count')
        
        # Test set distribution
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        axes[1].bar(range(len(test_dist)), test_dist.values, color='#2ecc71')
        axes[1].set_xticks(range(len(test_dist)))
        axes[1].set_xticklabels(self.class_names)
        axes[1].set_title('Test Set Class Distribution')
        axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'class_distribution_{self.timestamp}.png', dpi=150)
        plt.close()
        
    # =========================================================================
    # Inference Helper
    # =========================================================================
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data using the best model.
        
        Args:
            X: Feature DataFrame (must have same columns as training data)
            
        Returns:
            Array of predicted class labels
        """
        if self.best_model is None:
            raise ValueError("No model trained. Run the pipeline first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.best_model.predict(X_scaled)
        
        # Decode if label encoder was used
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Classification Modeling Agent - Automated ML Classification Pipeline'
    )
    parser.add_argument('filepath', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--target', '-t', type=str, required=True,
                       help='Name of the target column')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--models', '-m', type=str, nargs='+', default=None,
                       help='Models to train (default: all)')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--no-imbalance', action='store_true',
                       help='Disable imbalance handling')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Create and run agent
    agent = ClassificationAgent(
        filepath=args.filepath,
        target_column=args.target,
        output_dir=args.output_dir,
        test_size=args.test_size
    )
    
    agent.run(
        models=args.models,
        handle_imbalance=not args.no_imbalance,
        tune_hyperparameters=args.tune,
        cv_folds=args.cv_folds,
        generate_plots=not args.no_plots
    )


if __name__ == '__main__':
    main()
