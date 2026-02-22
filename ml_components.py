#!/usr/bin/env python3
"""
================================================================================
ML COMPONENTS - Shared Custom Transformers and Utilities
================================================================================

This module contains all custom sklearn-compatible transformers used in the
Student Performance Prediction pipeline. Both training_pipeline.py and 
server.py import from this module to ensure proper pickling/unpickling.

Components:
    - FeatureEngineer: Creates derived features from raw inputs
    - AdaptiveFeatureSelector: MI + RFE combined feature selection
    - ConditionalScaler: Optional scaling (disabled for RF, enabled for SVM)
    - SafeSMOTEENN: SMOTE-ENN with automatic fallback

================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Check for imbalanced-learn
try:
    from imblearn.base import BaseSampler
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    BaseSampler = BaseEstimator  # Fallback


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GRADE_LABELS = ["Fail", "Pass", "Distinction", "Excellent", "Exceptional"]
GRADE_MAP = {label: idx for idx, label in enumerate(GRADE_LABELS)}

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERIC_FIELDS = [
    "G1", "G2", "age", "Medu", "Fedu", "traveltime",
    "studytime", "failures", "famrel", "freetime",
    "goout", "Dalc", "Walc", "health", "absences"
]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature engineering.
    Creates derived features from existing ones.
    
    New Features:
        - avg_internal: Average of G1 and G2
        - grade_trend: G2 - G1 (improvement/decline)
        - total_alcohol: Dalc + Walc
        - study_free_ratio: studytime / (freetime + 1)
        - parent_edu_avg: (Medu + Fedu) / 2
    
    Parameters
    ----------
    feature_names : list of str, optional
        Names of input features. If None, uses generic names.
    """
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names
        self.feature_idx_ = None
        self.output_feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer by mapping feature names to indices."""
        X = np.asarray(X)
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_idx_ = {name: i for i, name in enumerate(self.feature_names)}
        return self
    
    def transform(self, X):
        """Transform X by adding engineered features."""
        X = np.asarray(X, dtype=np.float64)
        
        if self.feature_idx_ is None:
            self.fit(X)
        
        new_features = []
        new_names = list(self.feature_names)
        
        # Helper to safely get column
        def get_col(name):
            idx = self.feature_idx_.get(name)
            return X[:, idx] if idx is not None else None
        
        # avg_internal = (G1 + G2) / 2
        g1, g2 = get_col('G1'), get_col('G2')
        if g1 is not None and g2 is not None:
            avg_internal = (g1 + g2) / 2.0
            new_features.append(avg_internal.reshape(-1, 1))
            new_names.append('avg_internal')
        
        # grade_trend = G2 - G1
        if g1 is not None and g2 is not None:
            grade_trend = g2 - g1
            new_features.append(grade_trend.reshape(-1, 1))
            new_names.append('grade_trend')
        
        # total_alcohol = Dalc + Walc
        dalc, walc = get_col('Dalc'), get_col('Walc')
        if dalc is not None and walc is not None:
            total_alcohol = dalc + walc
            new_features.append(total_alcohol.reshape(-1, 1))
            new_names.append('total_alcohol')
        
        # study_free_ratio = studytime / (freetime + 1)
        studytime, freetime = get_col('studytime'), get_col('freetime')
        if studytime is not None and freetime is not None:
            study_free_ratio = studytime / (freetime + 1)
            new_features.append(study_free_ratio.reshape(-1, 1))
            new_names.append('study_free_ratio')
        
        # parent_edu_avg = (Medu + Fedu) / 2
        medu, fedu = get_col('Medu'), get_col('Fedu')
        if medu is not None and fedu is not None:
            parent_edu_avg = (medu + fedu) / 2.0
            new_features.append(parent_edu_avg.reshape(-1, 1))
            new_names.append('parent_edu_avg')
        
        # Combine original and new features
        if new_features:
            X_out = np.hstack([X] + new_features)
        else:
            X_out = X
        
        self.output_feature_names_ = new_names
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        return self.output_feature_names_ if self.output_feature_names_ else self.feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE FEATURE SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Adaptive feature selection combining Mutual Information and RFE.
    
    Methods:
        - 'mutual_info': Select top-k features by MI score
        - 'rfe': Use Recursive Feature Elimination
        - 'combined': Union of MI and RFE selections (default)
    
    Parameters
    ----------
    n_features : int, default=15
        Target number of features to select
    method : str, default='combined'
        Selection method ('mutual_info', 'rfe', 'combined')
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, n_features: int = 15, method: str = 'combined', 
                 random_state: int = 42):
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        
        # Fitted attributes
        self.selected_indices_ = None
        self.mi_scores_ = None
        self.rfe_ranking_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y):
        """Fit the feature selector."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_in_ = X.shape[1]
        n_select = min(self.n_features, X.shape[1])
        
        if self.method == 'mutual_info':
            self._fit_mi(X, y, n_select)
        elif self.method == 'rfe':
            self._fit_rfe(X, y, n_select)
        else:  # combined
            self._fit_combined(X, y, n_select)
        
        return self
    
    def _fit_mi(self, X, y, k):
        """Select top-k features by mutual information."""
        self.mi_scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        self.selected_indices_ = np.argsort(self.mi_scores_)[::-1][:k]
        self.selected_indices_ = np.sort(self.selected_indices_)
    
    def _fit_rfe(self, X, y, k):
        """Select features using RFE."""
        estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1
        )
        rfe = RFE(estimator, n_features_to_select=k, step=1)
        rfe.fit(X, y)
        self.rfe_ranking_ = rfe.ranking_
        self.selected_indices_ = np.where(rfe.support_)[0]
    
    def _fit_combined(self, X, y, k):
        """Union of MI and RFE selections."""
        # MI selection
        self.mi_scores_ = mutual_info_classif(X, y, random_state=self.random_state)
        mi_indices = set(np.argsort(self.mi_scores_)[::-1][:k])
        
        # RFE selection
        estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1
        )
        rfe = RFE(estimator, n_features_to_select=k, step=1)
        rfe.fit(X, y)
        self.rfe_ranking_ = rfe.ranking_
        rfe_indices = set(np.where(rfe.support_)[0])
        
        # Union
        self.selected_indices_ = np.array(sorted(mi_indices | rfe_indices))
    
    def transform(self, X):
        """Transform X by selecting features."""
        X = np.asarray(X)
        
        if self.selected_indices_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[:, self.selected_indices_]
    
    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        if input_features is not None and self.selected_indices_ is not None:
            return [input_features[i] for i in self.selected_indices_]
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL SCALER
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that can be conditionally enabled/disabled.
    
    Useful for pipelines where scaling is optional:
        - Random Forest: No scaling needed
        - SVM: Scaling recommended
    
    Parameters
    ----------
    enabled : bool, default=True
        Whether to apply scaling
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.scaler_ = None
    
    def fit(self, X, y=None):
        """Fit the scaler if enabled."""
        if self.enabled:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        return self
    
    def transform(self, X):
        """Transform X if scaling is enabled."""
        if self.enabled and self.scaler_ is not None:
            return self.scaler_.transform(X)
        return np.asarray(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE SMOTE-ENN RESAMPLER
# ═══════════════════════════════════════════════════════════════════════════════

class SafeSMOTEENN(BaseEstimator):
    """
    SMOTE-ENN with automatic fallback for edge cases.
    
    Handles:
        - Classes with very few samples
        - Automatic k_neighbors adjustment
        - Fallback to SMOTE-only or passthrough
    
    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.sampler_ = None
    
    def fit_resample(self, X, y):
        """Fit and resample the data."""
        if not HAS_IMBLEARN:
            return X, y
        
        from collections import Counter
        
        counts = Counter(y)
        min_count = min(counts.values())
        
        # Need at least 2 samples per class
        if min_count < 2:
            return X, y
        
        # Adjust k_neighbors
        k = max(1, min(5, min_count - 1))
        
        try:
            # Try SMOTE-ENN first
            from imblearn.under_sampling import EditedNearestNeighbours
            smote = SMOTE(k_neighbors=k, random_state=self.random_state)
            enn = EditedNearestNeighbours(n_neighbors=min(3, k + 1))
            self.sampler_ = SMOTEENN(smote=smote, enn=enn, random_state=self.random_state)
            return self.sampler_.fit_resample(X, y)
        except Exception:
            try:
                # Fallback to SMOTE only
                self.sampler_ = SMOTE(k_neighbors=k, random_state=self.random_state)
                return self.sampler_.fit_resample(X, y)
            except Exception:
                # Final fallback: return original
                return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def categorize_grade(g3_score: int) -> str:
    """
    Convert numeric grade (0-20) to category.
    
    Thresholds (Portuguese grading system):
        0-9:   Fail
        10-11: Pass
        12-14: Distinction
        15-17: Excellent
        18-20: Exceptional
    """
    if g3_score >= 18:
        return "Exceptional"
    elif g3_score >= 15:
        return "Excellent"
    elif g3_score >= 12:
        return "Distinction"
    elif g3_score >= 10:
        return "Pass"
    return "Fail"


def get_grade_color(grade: str) -> str:
    """Get color for grade visualization."""
    colors = {
        'Fail': '#EF4444',
        'Pass': '#F59E0B',
        'Distinction': '#3B82F6',
        'Excellent': '#10B981',
        'Exceptional': '#8B5CF6'
    }
    return colors.get(grade, '#6B7280')


def get_grade_icon(grade: str) -> str:
    """Get emoji icon for grade."""
    icons = {
        'Fail': '⚠️',
        'Pass': '✅',
        'Distinction': '🎯',
        'Excellent': '🌟',
        'Exceptional': '🏆'
    }
    return icons.get(grade, '📊')