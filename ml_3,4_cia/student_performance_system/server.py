#!/usr/bin/env python3
"""
================================================================================
STUDENT PERFORMANCE PREDICTION — FLASK BACKEND
================================================================================

Dynamic API endpoints for predictions, metrics, and model management.
================================================================================
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib
import logging
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Dict, Optional

# Import custom components (CRITICAL - must match training)
from ml_components import (
    FeatureEngineer,
    AdaptiveFeatureSelector,
    ConditionalScaler,
    GRADE_LABELS,
    CATEGORICAL_COLS,
    NUMERIC_FIELDS
)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("model_artifacts")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Thread-safe model manager."""
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self._lock = threading.Lock()
        self._models = {}
        self._metadata = None
        self._label_encoders = None
        self._load_error = None
        
        self.load_models()
    
    @property
    def is_loaded(self) -> bool:
        return bool(self._models) and self._metadata is not None
    
    @property
    def load_error(self) -> Optional[str]:
        return self._load_error
    
    def load_models(self) -> bool:
        """Load all model artifacts."""
        with self._lock:
            try:
                log.info("Loading models...")
                
                # Load pipelines
                rf_path = self.artifacts_dir / "random_forest_pipeline.pkl"
                svm_path = self.artifacts_dir / "svm_pipeline.pkl"
                
                if rf_path.exists():
                    self._models['rf'] = joblib.load(rf_path)
                    log.info("  ✓ RF pipeline loaded")
                
                if svm_path.exists():
                    self._models['svm'] = joblib.load(svm_path)
                    log.info("  ✓ SVM pipeline loaded")
                
                # Load encoders
                le_path = self.artifacts_dir / "label_encoders.pkl"
                if le_path.exists():
                    self._label_encoders = joblib.load(le_path)
                    log.info("  ✓ Label encoders loaded")
                
                # Load metadata
                meta_path = self.artifacts_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        self._metadata = json.load(f)
                    log.info("  ✓ Metadata loaded")
                
                self._load_error = None
                log.info("Models loaded successfully!")
                return True
                
            except Exception as e:
                self._load_error = str(e)
                log.error(f"Load failed: {e}")
                return False
    
    def preprocess_input(self, raw: Dict) -> np.ndarray:
        """Preprocess input for prediction."""
        if not self._metadata:
            raise ValueError("Metadata not loaded")
        
        feature_names = self._metadata.get('all_features', [])
        df = pd.DataFrame([{f: raw.get(f, 0) for f in feature_names}])
        
        # Encode categoricals
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self._label_encoders:
                le = self._label_encoders[col]
                val = str(df[col].iloc[0])
                df[col] = le.transform([val])[0] if val in le.classes_ else 0
        
        return df.values.astype(np.float64)
    
    def predict(self, raw: Dict) -> Dict:
        """Generate predictions from both models."""
        if not self.is_loaded:
            raise ValueError(f"Models not loaded: {self._load_error}")
        
        X = self.preprocess_input(raw)
        results = {}
        
        for name, model in self._models.items():
            try:
                proba = model.predict_proba(X)[0]
                pred_idx = int(np.argmax(proba))
                
                results[name] = {
                    'grade': GRADE_LABELS[pred_idx],
                    'confidence': round(float(proba[pred_idx]), 4),
                    'probabilities': {
                        g: round(float(p), 4)
                        for g, p in zip(GRADE_LABELS, proba)
                    }
                }
            except Exception as e:
                log.error(f"Prediction error ({name}): {e}")
                results[name] = {
                    'grade': 'Error',
                    'confidence': 0.0,
                    'probabilities': {g: 0.0 for g in GRADE_LABELS},
                    'error': str(e)
                }
        
        rf_grade = results.get('rf', {}).get('grade')
        svm_grade = results.get('svm', {}).get('grade')
        
        results['agree'] = rf_grade == svm_grade and rf_grade != 'Error'
        results['recommended_model'] = self._metadata.get('recommended_model', 'Random Forest')
        
        return results
    
    def get_metadata(self) -> Dict:
        return self._metadata or {}


# Initialize
model_manager = ModelManager(ARTIFACTS_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def handle_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log.error(f"Error in {f.__name__}: {e}")
            return jsonify({'error': str(e)}), 500
    return decorated


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if not model_manager.is_loaded:
        return render_template(
            "error.html",
            message="Models not loaded. Please run training first.",
            error=model_manager.load_error
        ), 500
    return render_template("index.html")


@app.route("/api/health")
@handle_errors
def health():
    return jsonify({
        'status': 'healthy' if model_manager.is_loaded else 'degraded',
        'models_loaded': model_manager.is_loaded,
        'version': model_manager.get_metadata().get('version', 'unknown')
    })


@app.route("/api/predict", methods=["POST"])
@handle_errors
def predict():
    raw = request.json
    
    # Convert numeric fields
    for field in NUMERIC_FIELDS:
        if field in raw:
            try:
                raw[field] = int(raw[field])
            except (ValueError, TypeError):
                raw[field] = 0
    
    return jsonify(model_manager.predict(raw))


@app.route("/api/model-info")
@handle_errors
def model_info():
    meta = model_manager.get_metadata()
    
    cv_results = meta.get('cv_results', {})
    holdout_results = meta.get('holdout_results', {})
    
    results = {}
    for model_name in ['Random Forest', 'SVM']:
        holdout = holdout_results.get(model_name, {})
        results[model_name] = {
            'accuracy': holdout.get('accuracy', 0),
            'precision': holdout.get('precision', 0),
            'recall': holdout.get('recall', 0),
            'f1_score': holdout.get('f1_score', 0),
            'cohen_kappa': holdout.get('cohen_kappa', 0),
            'confusion_matrix': holdout.get('confusion_matrix', []),
            'per_class': holdout.get('per_class', {})
        }
    
    return jsonify({
        'results': results,
        'cv': cv_results,
        'mi_scores': meta.get('mi_scores', {}),
        'rfe_ranking': meta.get('rfe_ranking', {}),
        'selected_features': meta.get('selected_features', []),
        'grade_labels': GRADE_LABELS,
        'version': meta.get('version', 'unknown'),
        'training_timestamp': meta.get('training_timestamp', 'unknown'),
        'model_comparison': meta.get('model_comparison', {}),
        'recommended_model': meta.get('recommended_model', 'Random Forest')
    })


@app.route("/api/model-metrics")
@handle_errors
def model_metrics():
    meta = model_manager.get_metadata()
    return jsonify({
        'cv_results': meta.get('cv_results', {}),
        'holdout_results': meta.get('holdout_results', {}),
        'model_comparison': meta.get('model_comparison', {}),
        'version': meta.get('version', 'unknown')
    })


@app.route("/api/feature-importance")
@handle_errors
def feature_importance():
    meta = model_manager.get_metadata()
    return jsonify({
        'mi_scores': meta.get('mi_scores', {}),
        'rfe_ranking': meta.get('rfe_ranking', {}),
        'selected_features': meta.get('selected_features', [])
    })


@app.route("/api/reload", methods=["POST"])
@handle_errors
def reload_models():
    success = model_manager.load_models()
    return jsonify({
        'success': success,
        'error': model_manager.load_error
    })


if __name__ == "__main__":
    log.info("╔════════════════════════════════════════════╗")
    log.info("║  Student Performance Prediction Server     ║")
    log.info("╚════════════════════════════════════════════╝")
    
    if not model_manager.is_loaded:
        log.warning("⚠ Models not loaded. Run: python training_pipeline.py")
    
    app.run(debug=True, port=5000, threaded=True)