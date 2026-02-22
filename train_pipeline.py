#!/usr/bin/env python3
"""
================================================================================
STUDENT PERFORMANCE PREDICTION — RESEARCH-GRADE TRAINING PIPELINE
================================================================================

This pipeline implements:
    - Nested Cross-Validation for unbiased evaluation
    - Data leakage prevention via unified pipelines
    - Statistical model comparison (paired t-test)
    - Proper feature selection within CV folds

Run: python training_pipeline.py
================================================================================
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
import hashlib

# Scikit-learn
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    train_test_split
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

# Statistical testing
from scipy import stats

# Import custom components from shared module
from ml_components import (
    FeatureEngineer,
    AdaptiveFeatureSelector,
    ConditionalScaler,
    SafeSMOTEENN,
    categorize_grade,
    GRADE_LABELS,
    GRADE_MAP,
    CATEGORICAL_COLS
)

# Check for imbalanced-learn
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""
    random_state: int = 42
    outer_cv_folds: int = 5
    inner_cv_folds: int = 3
    n_features_to_select: int = 15
    test_holdout_size: float = 0.15
    artifacts_dir: str = "model_artifacts"
    data_dir: str = "data"
    feature_selection_method: str = "combined"
    scale_for_rf: bool = False
    scale_for_svm: bool = True


@dataclass
class CVResults:
    """Results from nested cross-validation."""
    model_name: str
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float
    mean_kappa: float
    fold_scores: List[Dict] = field(default_factory=list)
    best_params: Dict = field(default_factory=dict)
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    training_time: float = 0.0


@dataclass
class ModelComparison:
    """Statistical comparison between models."""
    model_a: str
    model_b: str
    winner: str
    t_statistic: float
    p_value: float
    effect_size: float
    is_significant: bool
    confidence_level: float
    interpretation: str


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """Load the UCI Student Performance dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Try loading existing file
    for fname in ["student-mat.csv", "student.csv"]:
        fpath = data_path / fname
        if fpath.exists():
            for sep in [";", ","]:
                try:
                    df = pd.read_csv(str(fpath), sep=sep)
                    if len(df.columns) > 5:
                        log.info(f"✓ Loaded {fname}: {df.shape[0]} rows × {df.shape[1]} cols")
                        return df
                except:
                    continue
    
    # Try downloading
    log.info("↓ Downloading UCI Student Performance dataset...")
    try:
        import urllib.request
        import zipfile
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
        zip_path = data_path / "student.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(data_path))
        zip_path.unlink(missing_ok=True)
        
        fpath = data_path / "student-mat.csv"
        if fpath.exists():
            df = pd.read_csv(str(fpath), sep=";")
            log.info(f"✓ Downloaded: {df.shape[0]} rows")
            return df
    except Exception as e:
        log.warning(f"Download failed: {e}")
    
    # Generate synthetic data
    log.warning("⚠ Generating synthetic data...")
    return _generate_synthetic_data(data_path)


def _generate_synthetic_data(data_path: Path, n: int = 500) -> pd.DataFrame:
    """Generate synthetic data matching UCI schema."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        "school": np.random.choice(["GP", "MS"], n, p=[0.7, 0.3]),
        "sex": np.random.choice(["F", "M"], n),
        "age": np.random.choice(range(15, 23), n),
        "address": np.random.choice(["U", "R"], n, p=[0.78, 0.22]),
        "famsize": np.random.choice(["GT3", "LE3"], n, p=[0.68, 0.32]),
        "Pstatus": np.random.choice(["T", "A"], n, p=[0.87, 0.13]),
        "Medu": np.random.choice(range(5), n),
        "Fedu": np.random.choice(range(5), n),
        "Mjob": np.random.choice(["at_home", "health", "other", "services", "teacher"], n),
        "Fjob": np.random.choice(["at_home", "health", "other", "services", "teacher"], n),
        "reason": np.random.choice(["course", "home", "other", "reputation"], n),
        "guardian": np.random.choice(["mother", "father", "other"], n),
        "traveltime": np.random.choice(range(1, 5), n),
        "studytime": np.random.choice(range(1, 5), n),
        "failures": np.random.choice(range(5), n, p=[0.65, 0.2, 0.1, 0.03, 0.02]),
        "schoolsup": np.random.choice(["yes", "no"], n, p=[0.1, 0.9]),
        "famsup": np.random.choice(["yes", "no"], n, p=[0.4, 0.6]),
        "paid": np.random.choice(["yes", "no"], n),
        "activities": np.random.choice(["yes", "no"], n),
        "nursery": np.random.choice(["yes", "no"], n, p=[0.8, 0.2]),
        "higher": np.random.choice(["yes", "no"], n, p=[0.85, 0.15]),
        "internet": np.random.choice(["yes", "no"], n, p=[0.7, 0.3]),
        "romantic": np.random.choice(["yes", "no"], n, p=[0.35, 0.65]),
        "famrel": np.random.choice(range(1, 6), n),
        "freetime": np.random.choice(range(1, 6), n),
        "goout": np.random.choice(range(1, 6), n),
        "Dalc": np.random.choice(range(1, 6), n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        "Walc": np.random.choice(range(1, 6), n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        "health": np.random.choice(range(1, 6), n),
        "absences": np.clip(np.random.exponential(5, n).astype(int), 0, 75),
    })
    
    # Generate correlated grades
    base = (df["studytime"]*2 + df["Medu"]*0.8 + df["Fedu"]*0.5 
            - df["failures"]*3 - df["goout"]*0.5 - df["Dalc"]*0.8 
            + np.random.normal(8, 3, n))
    df["G1"] = np.clip(base.astype(int), 0, 20)
    df["G2"] = np.clip((df["G1"] + np.random.normal(0.5, 2, n)).astype(int), 0, 20)
    df["G3"] = np.clip((df["G2"]*0.6 + df["G1"]*0.2 + np.random.normal(2, 2.5, n)).astype(int), 0, 20)
    
    fpath = data_path / "student-mat.csv"
    df.to_csv(str(fpath), sep=";", index=False)
    log.info(f"✓ Synthetic dataset saved: {n} students")
    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Encode categorical columns to numeric values."""
    df = df.copy()
    label_encoders = {}
    cat_options = {}
    
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        cat_options[col] = list(le.classes_)
    
    return df, label_encoders, cat_options


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_rf_pipeline(config: PipelineConfig, feature_names: List[str]) -> Pipeline:
    """Build Random Forest pipeline."""
    steps = [
        ('engineer', FeatureEngineer(feature_names=feature_names)),
        ('selector', AdaptiveFeatureSelector(
            n_features=config.n_features_to_select,
            method=config.feature_selection_method,
            random_state=config.random_state
        )),
        ('scaler', ConditionalScaler(enabled=config.scale_for_rf)),
        ('classifier', RandomForestClassifier(
            random_state=config.random_state,
            n_jobs=-1
        ))
    ]
    
    return Pipeline(steps)


def build_svm_pipeline(config: PipelineConfig, feature_names: List[str]) -> Pipeline:
    """Build SVM pipeline."""
    steps = [
        ('engineer', FeatureEngineer(feature_names=feature_names)),
        ('selector', AdaptiveFeatureSelector(
            n_features=config.n_features_to_select,
            method=config.feature_selection_method,
            random_state=config.random_state
        )),
        ('scaler', ConditionalScaler(enabled=config.scale_for_svm)),
        ('classifier', SVC(
            random_state=config.random_state,
            probability=True
        ))
    ]
    
    return Pipeline(steps)


# ═══════════════════════════════════════════════════════════════════════════════
# NESTED CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def nested_cross_validation(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict,
    config: PipelineConfig,
    model_name: str
) -> CVResults:
    """
    Perform nested cross-validation for unbiased evaluation.
    
    Outer loop: Performance estimation
    Inner loop: Hyperparameter tuning
    """
    log.info(f"\n{'═'*60}")
    log.info(f"  NESTED CV: {model_name}")
    log.info(f"{'═'*60}")
    
    start_time = datetime.now()
    
    outer_cv = StratifiedKFold(
        n_splits=config.outer_cv_folds,
        shuffle=True,
        random_state=config.random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=config.inner_cv_folds,
        shuffle=True,
        random_state=config.random_state
    )
    
    fold_results = []
    all_best_params = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        log.info(f"  Fold {fold_idx + 1}/{config.outer_cv_folds}: "
                 f"Train={len(train_idx)}, Test={len(test_idx)}")
        
        # Apply SMOTE-ENN to training data only
        resampler = SafeSMOTEENN(random_state=config.random_state)
        X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            clone(pipeline),
            param_grid,
            cv=inner_cv,
            scoring='f1_weighted',
            n_jobs=-1,
            refit=True
        )
        
        try:
            grid_search.fit(X_train_res, y_train_res)
            best_model = grid_search.best_estimator_
            all_best_params.append(grid_search.best_params_)
            
            # Evaluate on outer test fold (original, non-resampled)
            y_pred = best_model.predict(X_test)
            
            fold_metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'kappa': float(cohen_kappa_score(y_test, y_pred)),
                'best_params': grid_search.best_params_
            }
            
            fold_results.append(fold_metrics)
            log.info(f"    → Accuracy: {fold_metrics['accuracy']:.4f}, "
                     f"F1: {fold_metrics['f1']:.4f}")
            
        except Exception as e:
            log.warning(f"    ⚠ Fold {fold_idx + 1} failed: {e}")
            continue
    
    if not fold_results:
        raise ValueError("All folds failed during nested CV")
    
    # Aggregate results
    accuracies = [r['accuracy'] for r in fold_results]
    f1_scores = [r['f1'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    recalls = [r['recall'] for r in fold_results]
    kappas = [r['kappa'] for r in fold_results]
    
    # Calculate 95% CI for F1
    n = len(f1_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores, ddof=1) if n > 1 else 0
    
    if n > 1 and std_f1 > 0:
        ci_95 = stats.t.interval(0.95, df=n-1, loc=mean_f1, scale=std_f1/np.sqrt(n))
    else:
        ci_95 = (mean_f1, mean_f1)
    
    # Most common best params
    if all_best_params:
        param_strs = [str(sorted(p.items())) for p in all_best_params]
        most_common = Counter(param_strs).most_common(1)[0][0]
        best_params = dict(eval(most_common))
    else:
        best_params = {}
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    results = CVResults(
        model_name=model_name,
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies)),
        mean_f1=float(mean_f1),
        std_f1=float(std_f1),
        mean_precision=float(np.mean(precisions)),
        mean_recall=float(np.mean(recalls)),
        mean_kappa=float(np.mean(kappas)),
        fold_scores=fold_results,
        best_params=best_params,
        confidence_interval_95=(float(ci_95[0]), float(ci_95[1])),
        training_time=training_time
    )
    
    log.info(f"\n  Summary:")
    log.info(f"    Accuracy:  {results.mean_accuracy:.4f} ± {results.std_accuracy:.4f}")
    log.info(f"    F1-Score:  {results.mean_f1:.4f} ± {results.std_f1:.4f}")
    log.info(f"    95% CI:    [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    log.info(f"    Time:      {training_time:.1f}s")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models_statistically(
    results_a: CVResults,
    results_b: CVResults,
    alpha: float = 0.05
) -> ModelComparison:
    """Perform paired t-test to compare models."""
    log.info(f"\n{'═'*60}")
    log.info(f"  STATISTICAL COMPARISON")
    log.info(f"{'═'*60}")
    
    scores_a = [r['f1'] for r in results_a.fold_scores]
    scores_b = [r['f1'] for r in results_b.fold_scores]
    
    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Cohen's d
    diff = np.array(scores_a) - np.array(scores_b)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    
    is_significant = p_value < alpha
    
    if is_significant:
        winner = results_a.model_name if np.mean(scores_a) > np.mean(scores_b) else results_b.model_name
    else:
        winner = "No significant difference"
    
    if not is_significant:
        interpretation = (
            f"No statistically significant difference between models "
            f"(p={p_value:.4f} > α={alpha}). Either model is acceptable."
        )
    else:
        effect_desc = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        interpretation = (
            f"{winner} significantly outperforms the other model "
            f"(p={p_value:.4f}, Cohen's d={cohens_d:.3f} [{effect_desc} effect])."
        )
    
    comparison = ModelComparison(
        model_a=results_a.model_name,
        model_b=results_b.model_name,
        winner=winner,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        effect_size=float(cohens_d),
        is_significant=is_significant,
        confidence_level=1 - alpha,
        interpretation=interpretation
    )
    
    log.info(f"  {results_a.model_name} F1: {results_a.mean_f1:.4f} ± {results_a.std_f1:.4f}")
    log.info(f"  {results_b.model_name} F1: {results_b.mean_f1:.4f} ± {results_b.std_f1:.4f}")
    log.info(f"  t-statistic: {t_stat:.4f}")
    log.info(f"  p-value: {p_value:.4f}")
    log.info(f"  Cohen's d: {cohens_d:.4f}")
    log.info(f"  → {interpretation}")
    
    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_model(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict,
    config: PipelineConfig
) -> Pipeline:
    """Train final model with best parameters."""
    # Apply SMOTE-ENN to full training data
    resampler = SafeSMOTEENN(random_state=config.random_state)
    X_res, y_res = resampler.fit_resample(X, y)
    
    final_pipeline = clone(pipeline)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_res, y_res)
    
    return final_pipeline


def evaluate_on_holdout(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict:
    """Evaluate model on holdout set."""
    y_pred = model.predict(X_test)
    
    labels = list(range(len(GRADE_LABELS)))
    
    results = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'cohen_kappa': float(cohen_kappa_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        'per_class': {
            'precision': {GRADE_LABELS[i]: float(v) for i, v in enumerate(
                precision_score(y_test, y_pred, average=None, zero_division=0, labels=labels))},
            'recall': {GRADE_LABELS[i]: float(v) for i, v in enumerate(
                recall_score(y_test, y_pred, average=None, zero_division=0, labels=labels))},
            'f1_score': {GRADE_LABELS[i]: float(v) for i, v in enumerate(
                f1_score(y_test, y_pred, average=None, zero_division=0, labels=labels))},
        }
    }
    
    log.info(f"\n  Holdout: {model_name}")
    log.info(f"    Accuracy: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_version() -> str:
    """Generate unique version ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:6]
    return f"v{timestamp}_{hash_suffix}"


def save_artifacts(
    config: PipelineConfig,
    rf_pipeline: Pipeline,
    svm_pipeline: Pipeline,
    rf_cv: CVResults,
    svm_cv: CVResults,
    rf_holdout: Dict,
    svm_holdout: Dict,
    comparison: ModelComparison,
    feature_names: List[str],
    label_encoders: Dict,
    cat_options: Dict
) -> str:
    """Save all artifacts."""
    artifact_dir = Path(config.artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    version = generate_version()
    log.info(f"\n{'═'*60}")
    log.info(f"  SAVING ARTIFACTS: {version}")
    log.info(f"{'═'*60}")
    
    # Save pipelines
    joblib.dump(rf_pipeline, artifact_dir / "random_forest_pipeline.pkl")
    joblib.dump(svm_pipeline, artifact_dir / "svm_pipeline.pkl")
    log.info("  ✓ Pipelines saved")
    
    # Save label encoders
    joblib.dump(label_encoders, artifact_dir / "label_encoders.pkl")
    log.info("  ✓ Label encoders saved")
    
    # Extract feature info from fitted pipeline
    mi_scores = {}
    rfe_ranking = {}
    selected_features = []
    
    try:
        selector = rf_pipeline.named_steps.get('selector')
        engineer = rf_pipeline.named_steps.get('engineer')
        
        if selector and engineer:
            eng_names = engineer.output_feature_names_ or []
            
            if selector.mi_scores_ is not None:
                mi_scores = {eng_names[i]: float(v) 
                            for i, v in enumerate(selector.mi_scores_) 
                            if i < len(eng_names)}
            
            if selector.rfe_ranking_ is not None:
                rfe_ranking = {eng_names[i]: int(v) 
                              for i, v in enumerate(selector.rfe_ranking_) 
                              if i < len(eng_names)}
            
            if selector.selected_indices_ is not None:
                selected_features = [eng_names[i] 
                                    for i in selector.selected_indices_ 
                                    if i < len(eng_names)]
    except Exception as e:
        log.warning(f"  Could not extract feature info: {e}")
    
    # Build metadata
    metadata = {
        "version": version,
        "training_timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "all_features": feature_names,
        "selected_features": selected_features,
        "mi_scores": mi_scores,
        "rfe_ranking": rfe_ranking,
        "grade_labels": GRADE_LABELS,
        "grade_map": GRADE_MAP,
        "categorical_cols": CATEGORICAL_COLS,
        "cat_options": cat_options,
        "cv_results": {
            "Random Forest": asdict(rf_cv),
            "SVM": asdict(svm_cv)
        },
        "holdout_results": {
            "Random Forest": rf_holdout,
            "SVM": svm_holdout
        },
        "model_comparison": asdict(comparison),
        "recommended_model": comparison.winner if comparison.is_significant else "Random Forest",
        "recommendation_reason": comparison.interpretation
    }
    
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info("  ✓ Metadata saved")
    
    return version


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(config: PipelineConfig = None) -> Dict:
    """Execute the complete training pipeline."""
    if config is None:
        config = PipelineConfig()
    
    start_time = datetime.now()
    
    log.info("╔" + "═"*58 + "╗")
    log.info("║  STUDENT PERFORMANCE PREDICTION — TRAINING PIPELINE     ║")
    log.info("╚" + "═"*58 + "╝")
    
    # Load data
    log.info("\n─── STEP 1: Loading Data ───")
    df = load_dataset(config.data_dir)
    df, label_encoders, cat_options = encode_categoricals(df)
    
    df["grade_category"] = df["G3"].apply(categorize_grade)
    df["grade_label"] = df["grade_category"].map(GRADE_MAP)
    
    log.info("Grade distribution:")
    for grade in GRADE_LABELS:
        count = (df["grade_category"] == grade).sum()
        log.info(f"  {grade:14s} {count:4d} ({100*count/len(df):.1f}%)")
    
    # Prepare features
    exclude = {"G3", "grade_category", "grade_label"}
    feature_names = [c for c in df.columns if c not in exclude]
    
    X = df[feature_names].values.astype(np.float64)
    y = df["grade_label"].values.astype(int)
    
    # Holdout split
    log.info("\n─── STEP 2: Train/Holdout Split ───")
    X_main, X_holdout, y_main, y_holdout = train_test_split(
        X, y, test_size=config.test_holdout_size,
        random_state=config.random_state, stratify=y
    )
    log.info(f"  Main: {len(y_main)}, Holdout: {len(y_holdout)}")
    
    # Build pipelines
    log.info("\n─── STEP 3: Building Pipelines ───")
    rf_pipeline = build_rf_pipeline(config, feature_names)
    svm_pipeline = build_svm_pipeline(config, feature_names)
    
    # Parameter grids
    rf_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced']
    }
    
    svm_params = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': ['scale', 'auto'],
        'classifier__class_weight': ['balanced']
    }
    
    # Nested CV
    log.info("\n─── STEP 4: Nested Cross-Validation ───")
    rf_cv = nested_cross_validation(rf_pipeline, X_main, y_main, rf_params, config, "Random Forest")
    svm_cv = nested_cross_validation(svm_pipeline, X_main, y_main, svm_params, config, "SVM")
    
    # Statistical comparison
    comparison = compare_models_statistically(rf_cv, svm_cv)
    
    # Train final models
    log.info("\n─── STEP 5: Training Final Models ───")
    rf_final = train_final_model(rf_pipeline, X_main, y_main, rf_cv.best_params, config)
    svm_final = train_final_model(svm_pipeline, X_main, y_main, svm_cv.best_params, config)
    
    # Holdout evaluation
    log.info("\n─── STEP 6: Holdout Evaluation ───")
    rf_holdout = evaluate_on_holdout(rf_final, X_holdout, y_holdout, "Random Forest")
    svm_holdout = evaluate_on_holdout(svm_final, X_holdout, y_holdout, "SVM")
    
    # Save artifacts
    version = save_artifacts(
        config, rf_final, svm_final,
        rf_cv, svm_cv, rf_holdout, svm_holdout,
        comparison, feature_names, label_encoders, cat_options
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log.info("\n" + "═"*60)
    log.info(f"  COMPLETE | Version: {version} | Time: {elapsed:.1f}s")
    log.info(f"  Winner: {comparison.winner}")
    log.info("═"*60)
    log.info("  → Run: python server.py")
    log.info("  → Open: http://localhost:5000")
    
    return {"version": version, "elapsed": elapsed}


if __name__ == "__main__":
    run_pipeline()