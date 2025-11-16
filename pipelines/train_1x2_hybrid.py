#!/usr/bin/env python3
"""
Train 1X2 Hybrid Pipeline

Trains calibration layer and generates hybrid weight optimization
based on validation sets. Saves results to models/1x2_hybrid_v1/

Does NOT modify existing ML model training - only consumes existing
predictions and computes optimal ensemble weights.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils import setup_logging
from core.data_loader import ESPNDataLoader
from api.improved_prediction_service import ImprovedPredictionService


class Hybrid1X2Trainer:
    """
    Trainer for Hybrid 1X2 calibration and weight optimization
    """
    
    def __init__(self, config_path: str = "config/hybrid_1x2_config.yaml"):
        """Initialize trainer"""
        self.config_path = Path(config_path)
        self.logger = setup_logging()
        self.config = self._load_config()
        
        # Output directory
        self.output_dir = Path("models/1x2_hybrid_v1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.data_loader = ESPNDataLoader()
        self.prediction_service = None
        
        # Training data
        self.training_data = None
        self.validation_data = None
        
        self.logger.info("🎯 Hybrid 1X2 Trainer initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f).get('hybrid_1x2_config', {})
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'weights': {'ml': 0.45, 'scoreline': 0.25, 'poisson': 0.20, 'draw_specialist': 0.10},
            'calibration': {'enabled': True, 'method': 'temperature'},
            'training': {'test_size': 0.2, 'cv_folds': 5, 'min_samples': 1000}
        }
    
    def prepare_data(self):
        """Prepare training data"""
        self.logger.info("📊 Preparing training data...")
        
        # Load historical data
        df = self.data_loader.load_fixtures()
        if df is None or df.empty:
            raise ValueError("No training data available")
        
        # Filter recent data
        cutoff_date = datetime.now() - timedelta(days=2*365)  # 2 years
        df = df[df['date'] >= cutoff_date].copy()
        
        # Create target
        df['target'] = 0  # Home win
        df.loc[df['home_score'] == df['away_score'], 'target'] = 1  # Draw
        df.loc[df['home_score'] < df['away_score'], 'target'] = 2  # Away win
        
        # Time-based split
        split_date = df['date'].quantile(0.8)
        train_df = df[df['date'] <= split_date].copy()
        val_df = df[df['date'] > split_date].copy()
        
        self.logger.info(f"📈 Training: {len(train_df)} matches")
        self.logger.info(f"📈 Validation: {len(val_df)} matches")
        
        self.training_data = train_df
        self.validation_data = val_df
    
    def generate_predictions(self):
        """Generate predictions from all sources"""
        self.logger.info("🔮 Generating predictions from all sources...")
        
        # Initialize prediction service
        self.prediction_service = ImprovedPredictionService()
        
        # Generate predictions for validation set
        predictions = []
        
        for idx, row in self.validation_data.iterrows():
            try:
                home_team = str(row.get('home_team_id', 'Unknown'))
                away_team = str(row.get('away_team_id', 'Unknown'))
                league = str(row.get('league_id', 'Unknown'))
                
                # Get hybrid predictions (includes all sources)
                result = self.prediction_service.predict_with_hybrid_1x2(
                    home_team=home_team,
                    away_team=away_team,
                    league=league
                )
                
                if result and 'prediction_1x2' in result:
                    pred_data = {
                        'match_id': idx,
                        'target': row['target'],
                        'ml_home': result['prediction_1x2'].get('prob_home_win', 1/3),
                        'ml_draw': result['prediction_1x2'].get('prob_draw', 1/3),
                        'ml_away': result['prediction_1x2'].get('prob_away_win', 1/3)
                    }
                    
                    # Add hybrid components if available
                    if 'hybrid' in result['prediction_1x2']:
                        hybrid_info = result['prediction_1x2']['hybrid']
                        pred_data.update({
                            'hybrid_home': hybrid_info.get('1', 1/3),
                            'hybrid_draw': hybrid_info.get('X', 1/3),
                            'hybrid_away': hybrid_info.get('2', 1/3)
                        })
                        
                        # Add component sources
                        components = hybrid_info.get('components', {})
                        if components.get('scoreline'):
                            pred_data.update({
                                'scoreline_home': components['scoreline'].get('1', 1/3),
                                'scoreline_draw': components['scoreline'].get('X', 1/3),
                                'scoreline_away': components['scoreline'].get('2', 1/3)
                            })
                        
                        if components.get('poisson'):
                            pred_data.update({
                                'poisson_home': components['poisson'].get('1', 1/3),
                                'poisson_draw': components['poisson'].get('X', 1/3),
                                'poisson_away': components['poisson'].get('2', 1/3)
                            })
                        
                        if components.get('draw_specialist'):
                            pred_data['draw_specialist'] = components['draw_specialist']
                    
                    predictions.append(pred_data)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error generating prediction for match {idx}: {e}")
                continue
        
        self.predictions_df = pd.DataFrame(predictions)
        self.logger.info(f"✅ Generated {len(self.predictions_df)} predictions")
    
    def optimize_weights(self) -> Dict[str, float]:
        """Optimize ensemble weights"""
        self.logger.info("⚖️ Optimizing ensemble weights...")
        
        if self.predictions_df.empty:
            return self.config['weights']
        
        # Prepare data for optimization
        y_true = self.predictions_df['target'].values
        
        # Get available sources
        sources = []
        source_names = []
        
        if 'ml_home' in self.predictions_df.columns:
            ml_probs = self.predictions_df[['ml_home', 'ml_draw', 'ml_away']].values
            sources.append(ml_probs)
            source_names.append('ml')
        
        if 'scoreline_home' in self.predictions_df.columns:
            scoreline_probs = self.predictions_df[['scoreline_home', 'scoreline_draw', 'scoreline_away']].values
            sources.append(scoreline_probs)
            source_names.append('scoreline')
        
        if 'poisson_home' in self.predictions_df.columns:
            poisson_probs = self.predictions_df[['poisson_home', 'poisson_draw', 'poisson_away']].values
            sources.append(poisson_probs)
            source_names.append('poisson')
        
        if len(sources) < 2:
            self.logger.warning("⚠️ Insufficient sources for optimization")
            return self.config['weights']
        
        # Grid search for optimal weights
        best_weights = None
        best_score = float('inf')
        
        from itertools import product
        
        # Simple grid search
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        for weights in product(weight_options, repeat=len(sources)):
            if abs(sum(weights) - 1.0) > 0.01:  # Skip if weights don't sum to ~1
                continue
            
            # Combine predictions
            combined_probs = np.zeros((len(y_true), 3))
            for i, (source, weight) in enumerate(zip(sources, weights)):
                combined_probs += weight * source
            
            # Calculate log loss
            try:
                score = log_loss(y_true, combined_probs, labels=[0, 1, 2])
                if score < best_score:
                    best_score = score
                    best_weights = dict(zip(source_names, weights))
            except:
                continue
        
        if best_weights:
            self.logger.info(f"✅ Optimal weights found: {best_weights}")
            return best_weights
        else:
            self.logger.warning("⚠️ Weight optimization failed, using default")
            return self.config['weights']
    
    def train_calibrator(self) -> Any:
        """Train calibration layer"""
        self.logger.info("🔧 Training calibration layer...")
        
        if self.predictions_df.empty:
            return None
        
        # Use hybrid predictions if available, otherwise ML
        if 'hybrid_home' in self.predictions_df.columns:
            X = self.predictions_df[['hybrid_home', 'hybrid_draw', 'hybrid_away']].values
        else:
            X = self.predictions_df[['ml_home', 'ml_draw', 'ml_away']].values
        
        y = self.predictions_df['target'].values
        
        # Create dummy classifier for calibration
        class DummyClassifier(BaseEstimator, ClassifierMixin):
            def __init__(self):
                self.classes_ = np.array([0, 1, 2])
            
            def fit(self, X, y):
                return self
            
            def predict_proba(self, X):
                return X
        
        dummy = DummyClassifier()
        
        # Train calibrator
        try:
            calibrator = CalibratedClassifierCV(dummy, method='isotonic', cv=3)
            calibrator.fit(X, y)
            
            self.logger.info("✅ Calibrator trained successfully")
            return calibrator
            
        except Exception as e:
            self.logger.error(f"❌ Calibrator training failed: {e}")
            return None
    
    def evaluate_performance(self, weights: Dict[str, float], calibrator: Any) -> Dict[str, float]:
        """Evaluate hybrid performance"""
        self.logger.info("📊 Evaluating hybrid performance...")
        
        if self.predictions_df.empty:
            return {}
        
        y_true = self.predictions_df['target'].values
        
        # Get ML baseline
        if 'ml_home' in self.predictions_df.columns:
            ml_probs = self.predictions_df[['ml_home', 'ml_draw', 'ml_away']].values
            ml_pred = np.argmax(ml_probs, axis=1)
            ml_accuracy = accuracy_score(y_true, ml_pred)
            ml_logloss = log_loss(y_true, ml_probs, labels=[0, 1, 2])
        else:
            ml_accuracy = ml_logloss = 0.0
        
        # Get hybrid performance
        if 'hybrid_home' in self.predictions_df.columns:
            hybrid_probs = self.predictions_df[['hybrid_home', 'hybrid_draw', 'hybrid_away']].values
            
            # Apply calibration if available
            if calibrator:
                try:
                    hybrid_probs = calibrator.predict_proba(hybrid_probs)
                except:
                    pass
            
            hybrid_pred = np.argmax(hybrid_probs, axis=1)
            hybrid_accuracy = accuracy_score(y_true, hybrid_pred)
            hybrid_logloss = log_loss(y_true, hybrid_probs, labels=[0, 1, 2])
        else:
            hybrid_accuracy = hybrid_logloss = ml_accuracy, ml_logloss
        
        metrics = {
            'ml_accuracy': float(ml_accuracy),
            'ml_logloss': float(ml_logloss),
            'hybrid_accuracy': float(hybrid_accuracy),
            'hybrid_logloss': float(hybrid_logloss),
            'accuracy_improvement': float(hybrid_accuracy - ml_accuracy),
            'logloss_improvement': float(ml_logloss - hybrid_logloss),
            'samples_evaluated': len(y_true)
        }
        
        self.logger.info(f"📈 Performance metrics: {metrics}")
        return metrics
    
    def save_artifacts(self, weights: Dict[str, float], calibrator: Any, metrics: Dict[str, float]):
        """Save training artifacts"""
        self.logger.info("💾 Saving training artifacts...")
        
        # Save weights
        weights_file = self.output_dir / "hybrid_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(weights, f, indent=2)
        
        # Save calibrator
        if calibrator:
            calibrator_file = self.output_dir / "hybrid_calibrator.pkl"
            joblib.dump(calibrator, calibrator_file)
        
        # Save feature info
        feature_info = {
            'sources_available': list(weights.keys()),
            'calibration_enabled': calibrator is not None,
            'training_date': datetime.now().isoformat(),
            'config_used': self.config
        }
        
        feature_file = self.output_dir / "hybrid_feature_info.json"
        with open(feature_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save metrics
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"✅ Artifacts saved to {self.output_dir}")
    
    def train(self):
        """Run complete training pipeline"""
        self.logger.info("🚀 Starting Hybrid 1X2 training pipeline...")
        
        try:
            # Prepare data
            self.prepare_data()
            
            # Generate predictions
            self.generate_predictions()
            
            # Optimize weights
            optimal_weights = self.optimize_weights()
            
            # Train calibrator
            calibrator = self.train_calibrator()
            
            # Evaluate performance
            metrics = self.evaluate_performance(optimal_weights, calibrator)
            
            # Save artifacts
            self.save_artifacts(optimal_weights, calibrator, metrics)
            
            self.logger.info("✅ Hybrid 1X2 training completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False


def main():
    """Run training pipeline"""
    print("🎯 Training Hybrid 1X2 Model")
    print("=" * 50)
    
    trainer = Hybrid1X2Trainer()
    success = trainer.train()
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"📁 Artifacts saved to: models/1x2_hybrid_v1/")
    else:
        print("\n❌ Training failed!")
    
    return success


if __name__ == "__main__":
    main()
