#!/usr/bin/env python3
"""
1X2 Ensemble Optimizer

Standalone optimizer for calculating optimal ensemble weights for 1X2 predictions.
Combines ML 1X2 v2, Poisson v2, and Elo predictions using historical performance data.

IMPORTANT: This module is ADDITIVE ONLY - does not modify existing code.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging


class Ensemble1X2Optimizer:
    """
    Standalone optimizer for 1X2 ensemble weights
    
    Optimizes weights for:
    - ML 1X2 v2 (three-binary-model system)
    - Poisson v2 1X2 probabilities  
    - Elo 1X2 probabilities
    
    Target metrics:
    - Minimize multi-class log loss
    - Minimize Brier score
    - Improve calibration (ECE)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize ensemble optimizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Optimization parameters
        self.weights_history = []
        self.optimization_results = {}
        self.validation_scores = {}
        
        # Data storage
        self.historical_data = None
        self.current_weights = None
        
        self.logger.info("🎯 Initialized 1X2 Ensemble Optimizer")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'lookback_days': 45,
            'min_predictions': 100,
            'improvement_threshold': 0.02,  # 2% improvement required
            'cv_folds': 5,
            'optimization_method': 'SLSQP',
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'weights_bounds': (0.0, 1.0),
            'prediction_logs_path': 'logs/prediction_history',
            'output_config_path': 'config/ensemble_1x2_weights.yaml',
            'backup_path': 'models/ensemble_1x2_v1',
            'log_file': 'logs/ensemble_1x2_optimizer.log'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated logging for optimizer"""
        # Create logs directory
        log_dir = Path(self.config['log_file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('Ensemble1X2Optimizer')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.config['log_file'])
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_historical_predictions(self) -> pd.DataFrame:
        """
        Load historical predictions from prediction logs
        
        Returns:
            DataFrame with historical prediction data
        """
        self.logger.info("📂 Loading historical prediction data...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['lookback_days'])
        
        prediction_data = []
        logs_path = Path(self.config['prediction_logs_path'])
        
        if not logs_path.exists():
            self.logger.warning(f"⚠️ Prediction logs path not found: {logs_path}")
            # Create sample data for testing
            return self._create_sample_data()
        
        # Load prediction log files
        for log_file in logs_path.glob('*.json'):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Parse timestamp
                            timestamp = datetime.fromisoformat(entry.get('timestamp', '').replace('Z', '+00:00'))
                            
                            if start_date <= timestamp <= end_date:
                                # Extract prediction data
                                if self._is_valid_prediction_entry(entry):
                                    prediction_data.append(self._parse_prediction_entry(entry))
                                    
                        except (json.JSONDecodeError, ValueError) as e:
                            continue
                            
            except Exception as e:
                self.logger.warning(f"⚠️ Error reading {log_file}: {e}")
                continue
        
        if not prediction_data:
            self.logger.warning("⚠️ No historical prediction data found, creating sample data")
            return self._create_sample_data()
        
        df = pd.DataFrame(prediction_data)
        self.logger.info(f"✅ Loaded {len(df)} historical predictions from {len(prediction_data)} entries")
        
        return df
    
    def _is_valid_prediction_entry(self, entry: Dict) -> bool:
        """Check if prediction entry has required fields"""
        required_fields = ['prediction_1x2', 'actual_result']
        return all(field in entry for field in required_fields)
    
    def _parse_prediction_entry(self, entry: Dict) -> Dict:
        """Parse prediction entry into standardized format"""
        pred_1x2 = entry['prediction_1x2']
        
        # Extract ML probabilities (1X2 v2 if available, fallback to v1)
        p_ml = [
            pred_1x2.get('prob_home_win', 0.33),
            pred_1x2.get('prob_draw', 0.33),
            pred_1x2.get('prob_away_win', 0.33)
        ]
        
        # Extract Poisson probabilities
        p_poisson = [
            pred_1x2.get('poisson_p_home', 0.33),
            pred_1x2.get('poisson_p_draw', 0.33), 
            pred_1x2.get('poisson_p_away', 0.33)
        ]
        
        # Extract Elo probabilities (if available)
        p_elo = [
            pred_1x2.get('elo_p_home', 0.33),
            pred_1x2.get('elo_p_draw', 0.33),
            pred_1x2.get('elo_p_away', 0.33)
        ]
        
        # Parse actual result
        actual_result = entry.get('actual_result', 'X')
        if actual_result == '1':
            y_true = 0  # Home win
        elif actual_result == 'X':
            y_true = 1  # Draw
        elif actual_result == '2':
            y_true = 2  # Away win
        else:
            y_true = 1  # Default to draw
        
        return {
            'timestamp': entry.get('timestamp'),
            'home_team': entry.get('home_team', 'Unknown'),
            'away_team': entry.get('away_team', 'Unknown'),
            'p_ml_home': p_ml[0],
            'p_ml_draw': p_ml[1], 
            'p_ml_away': p_ml[2],
            'p_poisson_home': p_poisson[0],
            'p_poisson_draw': p_poisson[1],
            'p_poisson_away': p_poisson[2],
            'p_elo_home': p_elo[0],
            'p_elo_draw': p_elo[1],
            'p_elo_away': p_elo[2],
            'y_true': y_true
        }
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing when no historical data available"""
        self.logger.info("🧪 Creating sample data for testing...")
        
        np.random.seed(42)
        n_samples = 500
        
        sample_data = []
        for i in range(n_samples):
            # Simulate realistic probabilities
            p_ml = np.random.dirichlet([2, 1, 2])  # Slight home bias
            p_poisson = np.random.dirichlet([2.2, 0.8, 1.8])  # More home bias
            p_elo = np.random.dirichlet([1.8, 1.2, 2.0])  # Balanced
            
            # Simulate actual result based on average probabilities
            avg_probs = (p_ml + p_poisson + p_elo) / 3
            y_true = np.random.choice(3, p=avg_probs)
            
            sample_data.append({
                'timestamp': (datetime.now() - timedelta(days=np.random.randint(1, 45))).isoformat(),
                'home_team': f'Team_{i%20}',
                'away_team': f'Team_{(i+10)%20}',
                'p_ml_home': p_ml[0],
                'p_ml_draw': p_ml[1],
                'p_ml_away': p_ml[2],
                'p_poisson_home': p_poisson[0],
                'p_poisson_draw': p_poisson[1],
                'p_poisson_away': p_poisson[2],
                'p_elo_home': p_elo[0],
                'p_elo_draw': p_elo[1],
                'p_elo_away': p_elo[2],
                'y_true': y_true
            })
        
        return pd.DataFrame(sample_data)
    
    def calculate_ensemble_probabilities(self, weights: np.ndarray, 
                                       p_ml: np.ndarray, p_poisson: np.ndarray, 
                                       p_elo: np.ndarray) -> np.ndarray:
        """
        Calculate ensemble probabilities using given weights
        
        Args:
            weights: [w_ml, w_poisson, w_elo]
            p_ml: ML probabilities (N, 3)
            p_poisson: Poisson probabilities (N, 3)
            p_elo: Elo probabilities (N, 3)
            
        Returns:
            Ensemble probabilities (N, 3)
        """
        w_ml, w_poisson, w_elo = weights
        
        ensemble_probs = (w_ml * p_ml + 
                         w_poisson * p_poisson + 
                         w_elo * p_elo)
        
        # Normalize to ensure probabilities sum to 1
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        ensemble_probs = ensemble_probs / np.maximum(row_sums, 1e-15)
        
        return ensemble_probs
    
    def objective_function(self, weights: np.ndarray, 
                          p_ml: np.ndarray, p_poisson: np.ndarray, 
                          p_elo: np.ndarray, y_true: np.ndarray) -> float:
        """
        Objective function to minimize (multi-class log loss)
        
        Args:
            weights: [w_ml, w_poisson, w_elo]
            p_ml: ML probabilities (N, 3)
            p_poisson: Poisson probabilities (N, 3)
            p_elo: Elo probabilities (N, 3)
            y_true: True labels (N,)
            
        Returns:
            Log loss value
        """
        ensemble_probs = self.calculate_ensemble_probabilities(weights, p_ml, p_poisson, p_elo)
        
        # Calculate log loss
        try:
            loss = log_loss(y_true, ensemble_probs, labels=[0, 1, 2])
            return loss
        except Exception as e:
            # Return high penalty if calculation fails
            return 10.0
    
    def calculate_metrics(self, weights: np.ndarray, 
                         p_ml: np.ndarray, p_poisson: np.ndarray, 
                         p_elo: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for given weights
        
        Args:
            weights: [w_ml, w_poisson, w_elo]
            p_ml: ML probabilities (N, 3)
            p_poisson: Poisson probabilities (N, 3)
            p_elo: Elo probabilities (N, 3)
            y_true: True labels (N,)
            
        Returns:
            Dictionary with metrics
        """
        ensemble_probs = self.calculate_ensemble_probabilities(weights, p_ml, p_poisson, p_elo)
        
        metrics = {}
        
        # Log loss
        try:
            metrics['log_loss'] = log_loss(y_true, ensemble_probs, labels=[0, 1, 2])
        except:
            metrics['log_loss'] = 10.0
        
        # Brier score (average across classes)
        brier_scores = []
        for class_idx in range(3):
            y_binary = (y_true == class_idx).astype(int)
            try:
                brier = brier_score_loss(y_binary, ensemble_probs[:, class_idx])
                brier_scores.append(brier)
            except:
                brier_scores.append(1.0)
        
        metrics['brier_score'] = np.mean(brier_scores)
        
        # Accuracy
        predictions = np.argmax(ensemble_probs, axis=1)
        metrics['accuracy'] = np.mean(predictions == y_true)
        
        # Expected Calibration Error (simplified)
        metrics['ece'] = self._calculate_ece(ensemble_probs, y_true)
        
        return metrics
    
    def _calculate_ece(self, probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        ece = 0.0
        
        for class_idx in range(3):
            class_probs = probs[:, class_idx]
            binary_labels = (y_true == class_idx).astype(int)
            
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = binary_labels[in_bin].mean()
                    avg_confidence_in_bin = class_probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece / 3  # Average across classes
    
    def optimize_weights(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Optimize ensemble weights using historical data
        
        Args:
            df: Historical prediction data
            
        Returns:
            Tuple of (optimal_weights, optimization_results)
        """
        self.logger.info("🔧 Starting ensemble weight optimization...")
        
        # Prepare data
        p_ml = df[['p_ml_home', 'p_ml_draw', 'p_ml_away']].values
        p_poisson = df[['p_poisson_home', 'p_poisson_draw', 'p_poisson_away']].values
        p_elo = df[['p_elo_home', 'p_elo_draw', 'p_elo_away']].values
        y_true = df['y_true'].values
        
        # Initial weights (equal weighting)
        initial_weights = np.array([0.33, 0.33, 0.34])
        
        # Constraints: weights >= 0, sum(weights) = 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        bounds = [self.config['weights_bounds']] * 3
        
        # Optimize
        self.logger.info("🎯 Running optimization...")
        result = minimize(
            fun=self.objective_function,
            x0=initial_weights,
            args=(p_ml, p_poisson, p_elo, y_true),
            method=self.config['optimization_method'],
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config['max_iterations'],
                'ftol': self.config['tolerance']
            }
        )
        
        if result.success:
            optimal_weights = result.x
            self.logger.info(f"✅ Optimization successful!")
            self.logger.info(f"📊 Optimal weights: ML={optimal_weights[0]:.3f}, Poisson={optimal_weights[1]:.3f}, Elo={optimal_weights[2]:.3f}")
        else:
            self.logger.warning(f"⚠️ Optimization failed: {result.message}")
            optimal_weights = initial_weights
        
        # Calculate metrics for optimal weights
        metrics = self.calculate_metrics(optimal_weights, p_ml, p_poisson, p_elo, y_true)
        
        optimization_results = {
            'success': result.success,
            'weights': optimal_weights.tolist(),
            'metrics': metrics,
            'optimization_info': {
                'method': self.config['optimization_method'],
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'function_value': result.fun,
                'message': result.message
            }
        }
        
        return optimal_weights, optimization_results
    
    def cross_validate_weights(self, df: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """
        Cross-validate weights using k-fold CV
        
        Args:
            df: Historical prediction data
            weights: Weights to validate
            
        Returns:
            Cross-validation metrics
        """
        self.logger.info(f"🔄 Cross-validating weights with {self.config['cv_folds']} folds...")
        
        # Prepare data
        p_ml = df[['p_ml_home', 'p_ml_draw', 'p_ml_away']].values
        p_poisson = df[['p_poisson_home', 'p_poisson_draw', 'p_poisson_away']].values
        p_elo = df[['p_elo_home', 'p_elo_draw', 'p_elo_away']].values
        y_true = df['y_true'].values
        
        kf = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=42)
        
        cv_metrics = {
            'log_loss': [],
            'brier_score': [],
            'accuracy': [],
            'ece': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            # Use validation set for evaluation
            p_ml_val = p_ml[val_idx]
            p_poisson_val = p_poisson[val_idx]
            p_elo_val = p_elo[val_idx]
            y_true_val = y_true[val_idx]
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(weights, p_ml_val, p_poisson_val, p_elo_val, y_true_val)
            
            for metric_name, value in fold_metrics.items():
                cv_metrics[metric_name].append(value)
        
        # Calculate mean and std
        cv_results = {}
        for metric_name, values in cv_metrics.items():
            cv_results[f'{metric_name}_mean'] = np.mean(values)
            cv_results[f'{metric_name}_std'] = np.std(values)
        
        self.logger.info(f"✅ CV Log Loss: {cv_results['log_loss_mean']:.4f} ± {cv_results['log_loss_std']:.4f}")
        self.logger.info(f"✅ CV Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        
        return cv_results
    
    def load_current_weights(self) -> Optional[np.ndarray]:
        """Load current ensemble weights from config file"""
        config_path = Path(self.config['output_config_path'])
        
        if not config_path.exists():
            self.logger.info("📄 No existing weights config found")
            return None
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            weights = np.array([
                config_data['weights']['ml'],
                config_data['weights']['poisson'],
                config_data['weights']['elo']
            ])
            
            self.logger.info(f"📂 Loaded current weights: ML={weights[0]:.3f}, Poisson={weights[1]:.3f}, Elo={weights[2]:.3f}")
            return weights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading current weights: {e}")
            return None
    
    def save_weights(self, weights: np.ndarray, metrics: Dict, backup_old: bool = True) -> None:
        """
        Save optimized weights to config file
        
        Args:
            weights: Optimal weights [w_ml, w_poisson, w_elo]
            metrics: Performance metrics
            backup_old: Whether to backup existing weights
        """
        config_path = Path(self.config['output_config_path'])
        backup_dir = Path(self.config['backup_path'])
        
        # Create directories
        config_path.parent.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup existing weights
        if backup_old and config_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"ensemble_1x2_weights_backup_{timestamp}.yaml"
            
            try:
                import shutil
                shutil.copy2(config_path, backup_file)
                self.logger.info(f"💾 Backed up old weights to {backup_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to backup old weights: {e}")
        
        # Create new config
        config_data = {
            'ensemble_1x2_weights': {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'optimization_method': self.config['optimization_method'],
                'weights': {
                    'ml': float(weights[0]),
                    'poisson': float(weights[1]),
                    'elo': float(weights[2])
                },
                'performance_metrics': {
                    'log_loss': float(metrics.get('log_loss', 0)),
                    'brier_score': float(metrics.get('brier_score', 0)),
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'ece': float(metrics.get('ece', 0))
                },
                'data_info': {
                    'lookback_days': self.config['lookback_days'],
                    'min_predictions': self.config['min_predictions'],
                    'improvement_threshold': self.config['improvement_threshold']
                }
            }
        }
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"💾 Saved optimized weights to {config_path}")
        self.logger.info(f"📊 Performance - Log Loss: {metrics.get('log_loss', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    def run_optimization(self) -> Dict:
        """
        Run complete ensemble optimization workflow
        
        Returns:
            Optimization results
        """
        self.logger.info("🚀 Starting 1X2 ensemble optimization workflow...")
        
        try:
            # Load historical data
            df = self.load_historical_predictions()
            
            if len(df) < self.config['min_predictions']:
                self.logger.warning(f"⚠️ Insufficient data: {len(df)} < {self.config['min_predictions']}")
                return {'success': False, 'error': 'Insufficient historical data'}
            
            # Load current weights for comparison
            current_weights = self.load_current_weights()
            current_metrics = None
            
            if current_weights is not None:
                # Prepare data for metrics calculation
                p_ml = df[['p_ml_home', 'p_ml_draw', 'p_ml_away']].values
                p_poisson = df[['p_poisson_home', 'p_poisson_draw', 'p_poisson_away']].values
                p_elo = df[['p_elo_home', 'p_elo_draw', 'p_elo_away']].values
                y_true = df['y_true'].values
                
                current_metrics = self.calculate_metrics(current_weights, p_ml, p_poisson, p_elo, y_true)
                self.logger.info(f"📊 Current performance - Log Loss: {current_metrics['log_loss']:.4f}")
            
            # Optimize weights
            optimal_weights, optimization_results = self.optimize_weights(df)
            
            # Cross-validate
            cv_results = self.cross_validate_weights(df, optimal_weights)
            
            # Check improvement
            new_log_loss = optimization_results['metrics']['log_loss']
            improvement = 0.0
            
            if current_metrics is not None:
                current_log_loss = current_metrics['log_loss']
                improvement = (current_log_loss - new_log_loss) / current_log_loss
                self.logger.info(f"📈 Log loss improvement: {improvement:.1%}")
            
            # Save weights if improvement is significant
            if improvement >= self.config['improvement_threshold'] or current_weights is None:
                self.save_weights(optimal_weights, optimization_results['metrics'])
                self.logger.info("✅ Weights updated due to significant improvement!")
                weights_updated = True
            else:
                self.logger.info(f"📊 No significant improvement ({improvement:.1%} < {self.config['improvement_threshold']:.1%})")
                weights_updated = False
            
            # Compile results
            results = {
                'success': True,
                'weights_updated': weights_updated,
                'optimal_weights': optimal_weights.tolist(),
                'improvement': improvement,
                'optimization_results': optimization_results,
                'cv_results': cv_results,
                'data_info': {
                    'total_predictions': len(df),
                    'lookback_days': self.config['lookback_days']
                }
            }
            
            self.logger.info("🎉 Ensemble optimization completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """
    Main execution function
    """
    print("🎯 1X2 Ensemble Optimizer")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = Ensemble1X2Optimizer()
    
    # Run optimization
    results = optimizer.run_optimization()
    
    if results['success']:
        print("\n✅ OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"📊 Weights Updated: {results['weights_updated']}")
        
        if 'optimal_weights' in results:
            weights = results['optimal_weights']
            print(f"🎯 Optimal Weights:")
            print(f"   ML (1X2 v2): {weights[0]:.3f}")
            print(f"   Poisson v2:  {weights[1]:.3f}")
            print(f"   Elo:         {weights[2]:.3f}")
        
        if 'improvement' in results:
            print(f"📈 Improvement: {results['improvement']:.1%}")
        
        if 'optimization_results' in results:
            metrics = results['optimization_results']['metrics']
            print(f"📊 Performance:")
            print(f"   Log Loss: {metrics['log_loss']:.4f}")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   ECE:      {metrics['ece']:.4f}")
    else:
        print(f"\n❌ OPTIMIZATION FAILED: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
