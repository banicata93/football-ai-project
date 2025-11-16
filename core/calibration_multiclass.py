#!/usr/bin/env python3
"""
Multiclass Calibration for Football Predictions

Implements calibration methods for 3-class (1X2) predictions:
- Temperature Scaling for softmax outputs
- Vector Scaling for multi-class probability vectors
- Platt Scaling for binary components
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemperatureScaling:
    """
    Temperature Scaling calibration for multiclass predictions
    
    Applies a single temperature parameter to scale logits before softmax:
    calibrated_probs = softmax(logits / temperature)
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'TemperatureScaling':
        """
        Fit temperature parameter using validation data
        
        Args:
            logits: Raw model outputs (N, 3) for 1X2 predictions
            y_true: True class labels (0=home, 1=draw, 2=away)
            
        Returns:
            Self for method chaining
        """
        logger.info("🌡️ Fitting temperature scaling...")
        
        def temperature_loss(temp):
            """Negative log-likelihood loss for temperature optimization"""
            temp = max(temp[0], 0.01)  # Ensure positive temperature
            scaled_logits = logits / temp
            
            # Apply softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Calculate negative log-likelihood
            log_probs = np.log(np.clip(probs, 1e-15, 1.0))
            nll = -np.mean(log_probs[np.arange(len(y_true)), y_true])
            
            return nll
        
        # Optimize temperature
        result = minimize(temperature_loss, [1.0], method='L-BFGS-B', bounds=[(0.01, 10.0)])
        
        self.temperature = max(result.x[0], 0.01)
        self.is_fitted = True
        
        logger.info(f"✅ Optimal temperature: {self.temperature:.4f}")
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to get calibrated probabilities
        
        Args:
            logits: Raw model outputs (N, 3)
            
        Returns:
            Calibrated probabilities (N, 3)
        """
        if not self.is_fitted:
            raise ValueError("Temperature scaling not fitted. Call fit() first.")
        
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs


class VectorScaling:
    """
    Vector Scaling calibration for multiclass predictions
    
    Applies separate scaling parameters to each class:
    calibrated_probs = softmax(W * logits + b)
    """
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'VectorScaling':
        """
        Fit vector scaling parameters
        
        Args:
            logits: Raw model outputs (N, 3)
            y_true: True class labels (0=home, 1=draw, 2=away)
            
        Returns:
            Self for method chaining
        """
        logger.info("📊 Fitting vector scaling...")
        
        n_classes = logits.shape[1]
        
        def vector_loss(params):
            """Negative log-likelihood loss for vector scaling"""
            weights = params[:n_classes]
            bias = params[n_classes:]
            
            # Apply scaling: W * logits + b
            scaled_logits = logits * weights + bias
            
            # Apply softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Calculate negative log-likelihood
            log_probs = np.log(np.clip(probs, 1e-15, 1.0))
            nll = -np.mean(log_probs[np.arange(len(y_true)), y_true])
            
            return nll
        
        # Initialize parameters
        initial_params = np.concatenate([np.ones(n_classes), np.zeros(n_classes)])
        
        # Optimize parameters
        result = minimize(vector_loss, initial_params, method='L-BFGS-B')
        
        self.weights = result.x[:n_classes]
        self.bias = result.x[n_classes:]
        self.is_fitted = True
        
        logger.info(f"✅ Vector scaling weights: {self.weights}")
        logger.info(f"✅ Vector scaling bias: {self.bias}")
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply vector scaling to get calibrated probabilities
        
        Args:
            logits: Raw model outputs (N, 3)
            
        Returns:
            Calibrated probabilities (N, 3)
        """
        if not self.is_fitted:
            raise ValueError("Vector scaling not fitted. Call fit() first.")
        
        # Apply scaling
        scaled_logits = logits * self.weights + self.bias
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs


class MulticlassCalibrator:
    """
    Comprehensive multiclass calibration for 1X2 predictions
    
    Supports multiple calibration methods:
    - Temperature scaling
    - Vector scaling  
    - Binary calibration for each class
    """
    
    def __init__(self, method: str = 'temperature', binary_method: str = 'platt'):
        """
        Initialize multiclass calibrator
        
        Args:
            method: Calibration method ('temperature', 'vector', 'binary')
            binary_method: Binary calibration method ('platt', 'isotonic')
        """
        self.method = method
        self.binary_method = binary_method
        self.is_fitted = False
        
        # Calibration models
        self.temp_calibrator = None
        self.vector_calibrator = None
        self.binary_calibrators = {}
        
        logger.info(f"🎯 Initialized multiclass calibrator (method={method})")
    
    def fit(self, predictions: np.ndarray, y_true: np.ndarray, 
            logits: Optional[np.ndarray] = None) -> 'MulticlassCalibrator':
        """
        Fit calibration model
        
        Args:
            predictions: Model probability predictions (N, 3) for 1X2
            y_true: True class labels (0=home, 1=draw, 2=away)
            logits: Raw model logits (optional, needed for temp/vector scaling)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"🔧 Fitting {self.method} calibration on {len(predictions)} samples...")
        
        if self.method == 'temperature':
            if logits is None:
                # Convert probabilities to logits
                logits = np.log(np.clip(predictions, 1e-15, 1.0))
            
            self.temp_calibrator = TemperatureScaling()
            self.temp_calibrator.fit(logits, y_true)
            
        elif self.method == 'vector':
            if logits is None:
                # Convert probabilities to logits
                logits = np.log(np.clip(predictions, 1e-15, 1.0))
            
            self.vector_calibrator = VectorScaling()
            self.vector_calibrator.fit(logits, y_true)
            
        elif self.method == 'binary':
            # Fit separate binary calibrators for each class
            for class_idx in range(3):
                class_name = ['home', 'draw', 'away'][class_idx]
                
                # Create binary labels (1 if this class, 0 otherwise)
                binary_labels = (y_true == class_idx).astype(int)
                class_probs = predictions[:, class_idx]
                
                if self.binary_method == 'platt':
                    # Use logistic regression for Platt scaling
                    calibrator = LogisticRegression()
                    calibrator.fit(class_probs.reshape(-1, 1), binary_labels)
                elif self.binary_method == 'isotonic':
                    # Use isotonic regression
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(class_probs, binary_labels)
                
                self.binary_calibrators[class_idx] = calibrator
                logger.info(f"✅ Fitted {self.binary_method} calibrator for {class_name}")
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, predictions: np.ndarray, 
                     logits: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply calibration to get calibrated probabilities
        
        Args:
            predictions: Model probability predictions (N, 3)
            logits: Raw model logits (optional)
            
        Returns:
            Calibrated probabilities (N, 3)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'temperature':
            if logits is None:
                logits = np.log(np.clip(predictions, 1e-15, 1.0))
            return self.temp_calibrator.predict_proba(logits)
            
        elif self.method == 'vector':
            if logits is None:
                logits = np.log(np.clip(predictions, 1e-15, 1.0))
            return self.vector_calibrator.predict_proba(logits)
            
        elif self.method == 'binary':
            calibrated_probs = np.zeros_like(predictions)
            
            for class_idx in range(3):
                class_probs = predictions[:, class_idx]
                calibrator = self.binary_calibrators[class_idx]
                
                if self.binary_method == 'platt':
                    calibrated_class_probs = calibrator.predict_proba(class_probs.reshape(-1, 1))[:, 1]
                elif self.binary_method == 'isotonic':
                    calibrated_class_probs = calibrator.predict(class_probs)
                
                calibrated_probs[:, class_idx] = calibrated_class_probs
            
            # Normalize to ensure probabilities sum to 1
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            calibrated_probs = calibrated_probs / np.maximum(row_sums, 1e-15)
            
            return calibrated_probs
    
    def save_calibrator(self, filepath: str) -> None:
        """
        Save calibrator to file
        
        Args:
            filepath: Path to save the calibrator
        """
        calibrator_data = {
            'method': self.method,
            'binary_method': self.binary_method,
            'is_fitted': self.is_fitted,
            'temp_calibrator': self.temp_calibrator,
            'vector_calibrator': self.vector_calibrator,
            'binary_calibrators': self.binary_calibrators
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
        
        logger.info(f"💾 Saved calibrator to {filepath}")
    
    @classmethod
    def load_calibrator(cls, filepath: str) -> 'MulticlassCalibrator':
        """
        Load calibrator from file
        
        Args:
            filepath: Path to the saved calibrator
            
        Returns:
            Loaded MulticlassCalibrator instance
        """
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        # Create new instance
        calibrator = cls(
            method=calibrator_data['method'],
            binary_method=calibrator_data['binary_method']
        )
        
        # Restore state
        calibrator.is_fitted = calibrator_data['is_fitted']
        calibrator.temp_calibrator = calibrator_data['temp_calibrator']
        calibrator.vector_calibrator = calibrator_data['vector_calibrator']
        calibrator.binary_calibrators = calibrator_data['binary_calibrators']
        
        logger.info(f"📂 Loaded {calibrator.method} calibrator from {filepath}")
        return calibrator


def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate calibration metrics for multiclass predictions
    
    Args:
        y_true: True class labels (0, 1, 2)
        y_prob: Predicted probabilities (N, 3)
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with calibration metrics
    """
    metrics = {}
    
    # Expected Calibration Error (ECE)
    ece_total = 0.0
    
    for class_idx in range(3):
        class_name = ['home', 'draw', 'away'][class_idx]
        
        # Binary labels for this class
        binary_labels = (y_true == class_idx).astype(int)
        class_probs = y_prob[:, class_idx]
        
        # Calculate ECE for this class
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        class_ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = binary_labels[in_bin].mean()
                avg_confidence_in_bin = class_probs[in_bin].mean()
                class_ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        metrics[f'ece_{class_name}'] = class_ece
        ece_total += class_ece
    
    metrics['ece_overall'] = ece_total / 3
    
    # Brier Score
    brier_scores = []
    for class_idx in range(3):
        binary_labels = (y_true == class_idx).astype(int)
        class_probs = y_prob[:, class_idx]
        brier_score = np.mean((class_probs - binary_labels) ** 2)
        brier_scores.append(brier_score)
        metrics[f'brier_{["home", "draw", "away"][class_idx]}'] = brier_score
    
    metrics['brier_overall'] = np.mean(brier_scores)
    
    # Log Loss
    log_probs = np.log(np.clip(y_prob, 1e-15, 1.0))
    log_loss = -np.mean(log_probs[np.arange(len(y_true)), y_true])
    metrics['log_loss'] = log_loss
    
    return metrics


def main():
    """
    Example usage and testing
    """
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate uncalibrated predictions (overconfident)
    raw_logits = np.random.randn(n_samples, 3) * 2
    raw_probs = np.exp(raw_logits) / np.sum(np.exp(raw_logits), axis=1, keepdims=True)
    
    # Create true labels
    y_true = np.random.choice(3, n_samples, p=[0.45, 0.25, 0.30])
    
    print("🧪 Testing Multiclass Calibration")
    print("=" * 40)
    
    # Test different calibration methods
    methods = ['temperature', 'vector', 'binary']
    
    for method in methods:
        print(f"\n📊 Testing {method} calibration...")
        
        # Fit calibrator
        calibrator = MulticlassCalibrator(method=method)
        calibrator.fit(raw_probs, y_true, raw_logits)
        
        # Get calibrated predictions
        calibrated_probs = calibrator.predict_proba(raw_probs, raw_logits)
        
        # Calculate metrics
        raw_metrics = calculate_calibration_metrics(y_true, raw_probs)
        calibrated_metrics = calculate_calibration_metrics(y_true, calibrated_probs)
        
        print(f"Raw ECE: {raw_metrics['ece_overall']:.4f}")
        print(f"Calibrated ECE: {calibrated_metrics['ece_overall']:.4f}")
        print(f"Raw Brier: {raw_metrics['brier_overall']:.4f}")
        print(f"Calibrated Brier: {calibrated_metrics['brier_overall']:.4f}")
        
        # Save and load test
        calibrator.save_calibrator(f'test_{method}_calibrator.pkl')
        loaded_calibrator = MulticlassCalibrator.load_calibrator(f'test_{method}_calibrator.pkl')
        
        print(f"✅ {method.capitalize()} calibration test completed!")
    
    print("\n✅ All multiclass calibration tests completed successfully!")


if __name__ == "__main__":
    main()
