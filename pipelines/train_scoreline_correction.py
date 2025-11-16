#!/usr/bin/env python3
"""
Scoreline Correction Model Training

Trains a LightGBM regression model to predict correction factors for Poisson scoreline probabilities.
Target: correction_factor = actual_scoreline_frequency / poisson_predicted_frequency

ADDITIVE - does not modify existing training pipelines.
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import ESPNDataLoader
from core.poisson_v2 import PoissonV2
from core.utils import setup_logging


class ScorelineCorrectionTrainer:
    """
    Trainer for scoreline correction model
    
    Creates regression model that predicts correction factors:
    correction = actual_frequency / poisson_frequency
    """
    
    def __init__(self, config: Dict = None):
        """Initialize trainer"""
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Components
        self.data_loader = ESPNDataLoader()
        self.poisson_v2 = PoissonV2()
        
        # Model storage
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        
        self.logger.info("🎯 Initialized Scoreline Correction Trainer")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'lookback_years': 3,
            'max_goals': 4,
            'min_matches_per_scoreline': 5,
            'smoothing_factor': 0.1,
            
            # Model parameters
            'lgb_params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            
            # Output
            'model_dir': 'models/scoreline_correction_v1',
            'log_file': 'logs/scoreline_correction_training.log'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        log_file = Path(self.config['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('ScorelineCorrectionTrainer')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
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
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare training data"""
        self.logger.info("📂 Loading match data...")
        
        # Load fixtures
        df = self.data_loader.load_fixtures()
        
        if df is None or df.empty:
            raise ValueError("No training data available")
        
        self.logger.info(f"📊 Loaded {len(df)} matches")
        
        # Add required columns
        df['league'] = df['league_id'].astype(str)
        df['home_team'] = df['home_team_id'].astype(str)
        df['away_team'] = df['away_team_id'].astype(str)
        
        # Filter recent data
        cutoff_date = datetime.now() - timedelta(days=self.config['lookback_years'] * 365)
        df = df[df['date'] >= cutoff_date].copy()
        
        # Filter valid scores
        max_goals = self.config['max_goals']
        df = df[
            (df['home_score'] <= max_goals) & 
            (df['away_score'] <= max_goals) &
            (df['home_score'] >= 0) & 
            (df['away_score'] >= 0)
        ].copy()
        
        self.logger.info(f"📅 Using {len(df)} matches from last {self.config['lookback_years']} years")
        
        return df
    
    def compute_scoreline_frequencies(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute actual and Poisson-predicted scoreline frequencies
        
        Returns:
            Dictionary with actual and predicted frequencies per scoreline
        """
        self.logger.info("📊 Computing scoreline frequencies...")
        
        max_goals = self.config['max_goals']
        
        # Count actual scorelines
        actual_counts = defaultdict(int)
        poisson_predictions = defaultdict(float)
        total_matches = 0
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                self.logger.info(f"   Processing match {idx}/{len(df)}")
            
            try:
                # Actual scoreline
                home_score = int(row['home_score'])
                away_score = int(row['away_score'])
                actual_scoreline = f"{home_score}-{away_score}"
                actual_counts[actual_scoreline] += 1
                
                # Poisson prediction for this match
                poisson_result = self.poisson_v2.predict_match(
                    row['home_team'], row['away_team'], df
                )
                
                lambda_home = poisson_result.get('expected_home_goals', 1.5)
                lambda_away = poisson_result.get('expected_away_goals', 1.2)
                
                # Add Poisson probabilities for all scorelines
                for i in range(max_goals + 1):
                    for j in range(max_goals + 1):
                        scoreline = f"{i}-{j}"
                        prob_home = poisson.pmf(i, lambda_home)
                        prob_away = poisson.pmf(j, lambda_away)
                        poisson_prob = prob_home * prob_away
                        poisson_predictions[scoreline] += poisson_prob
                
                total_matches += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error processing match {idx}: {e}")
                continue
        
        # Convert to frequencies
        frequencies = {}
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scoreline = f"{i}-{j}"
                
                actual_freq = actual_counts[scoreline] / total_matches if total_matches > 0 else 0
                poisson_freq = poisson_predictions[scoreline] / total_matches if total_matches > 0 else 0
                
                frequencies[scoreline] = {
                    'actual_frequency': actual_freq,
                    'poisson_frequency': poisson_freq,
                    'actual_count': actual_counts[scoreline],
                    'total_matches': total_matches
                }
        
        self.logger.info(f"✅ Computed frequencies for {len(frequencies)} scorelines")
        return frequencies
    
    def create_training_data(self, df: pd.DataFrame, frequencies: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create training dataset for correction model
        
        Returns:
            Training DataFrame and feature names
        """
        self.logger.info("🔧 Creating training dataset...")
        
        training_data = []
        max_goals = self.config['max_goals']
        
        # Create one sample per scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                scoreline = f"{i}-{j}"
                freq_data = frequencies[scoreline]
                
                # Skip scorelines with insufficient data
                if freq_data['actual_count'] < self.config['min_matches_per_scoreline']:
                    continue
                
                # Calculate correction factor with smoothing
                actual_freq = freq_data['actual_frequency']
                poisson_freq = freq_data['poisson_frequency']
                
                if poisson_freq > 0:
                    correction_factor = actual_freq / poisson_freq
                else:
                    correction_factor = 1.0
                
                # Apply smoothing to avoid extreme values
                smoothing = self.config['smoothing_factor']
                correction_factor = (1 - smoothing) * correction_factor + smoothing * 1.0
                
                # Create features for this scoreline
                features = self._create_scoreline_training_features(
                    i, j, df, freq_data
                )
                
                features['correction_factor'] = correction_factor
                training_data.append(features)
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_data)
        
        # Feature names (exclude target)
        feature_names = [col for col in training_df.columns if col != 'correction_factor']
        
        self.logger.info(f"✅ Created training dataset: {len(training_df)} samples, {len(feature_names)} features")
        
        return training_df, feature_names
    
    def _create_scoreline_training_features(self, home_goals: int, away_goals: int, 
                                          df: pd.DataFrame, freq_data: Dict) -> Dict[str, float]:
        """Create features for scoreline training"""
        
        # Basic scoreline features
        features = {
            'home_goals': float(home_goals),
            'away_goals': float(away_goals),
            'total_goals': float(home_goals + away_goals),
            'goal_difference': float(abs(home_goals - away_goals)),
            'is_draw': float(home_goals == away_goals),
            'is_home_win': float(home_goals > away_goals),
            'is_away_win': float(home_goals < away_goals),
            'is_over_25': float((home_goals + away_goals) > 2.5),
            'is_btts': float(home_goals > 0 and away_goals > 0),
            'is_clean_sheet_home': float(away_goals == 0),
            'is_clean_sheet_away': float(home_goals == 0)
        }
        
        # Statistical features from data
        features.update({
            'actual_count': float(freq_data['actual_count']),
            'total_matches': float(freq_data['total_matches']),
            'poisson_frequency': freq_data['poisson_frequency'],
            'actual_frequency': freq_data['actual_frequency']
        })
        
        # League-level features (aggregated from df)
        try:
            # Overall scoring statistics
            avg_home_goals = df['home_score'].mean()
            avg_away_goals = df['away_score'].mean()
            avg_total_goals = avg_home_goals + avg_away_goals
            
            features.update({
                'league_avg_home_goals': avg_home_goals,
                'league_avg_away_goals': avg_away_goals,
                'league_avg_total_goals': avg_total_goals,
                'league_draw_rate': (df['home_score'] == df['away_score']).mean(),
                'league_btts_rate': ((df['home_score'] > 0) & (df['away_score'] > 0)).mean(),
                'league_over25_rate': ((df['home_score'] + df['away_score']) > 2.5).mean()
            })
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error computing league features: {e}")
            # Add default values
            features.update({
                'league_avg_home_goals': 1.5,
                'league_avg_away_goals': 1.2,
                'league_avg_total_goals': 2.7,
                'league_draw_rate': 0.25,
                'league_btts_rate': 0.5,
                'league_over25_rate': 0.5
            })
        
        return features
    
    def train_model(self, training_df: pd.DataFrame, feature_names: List[str]) -> None:
        """Train the correction model"""
        self.logger.info("🏋️ Training scoreline correction model...")
        
        # Prepare data
        X = training_df[feature_names].values
        y = training_df['correction_factor'].values
        
        self.logger.info(f"📊 Training on {len(X)} samples with {len(feature_names)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        self.model = lgb.train(
            params=self.config['lgb_params'],
            train_set=train_data,
            num_boost_round=self.config['n_estimators'],
            callbacks=[lgb.early_stopping(self.config['early_stopping_rounds'])],
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test']
        )
        
        # Store feature names
        self.feature_names = feature_names
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        self.training_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'n_samples': len(X),
            'n_features': len(feature_names)
        }
        
        self.logger.info("📊 Training completed:")
        self.logger.info(f"   RMSE: {self.training_metrics['rmse']:.4f}")
        self.logger.info(f"   MAE: {self.training_metrics['mae']:.4f}")
        self.logger.info(f"   R²: {self.training_metrics['r2']:.4f}")
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info("🔝 Top 10 Features:")
        for i, (feature, imp) in enumerate(feature_importance[:10]):
            self.logger.info(f"   {i+1}. {feature}: {imp}")
    
    def save_model(self) -> None:
        """Save trained model and metadata"""
        self.logger.info("💾 Saving scoreline correction model...")
        
        model_dir = Path(self.config['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "correction_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature names
        features_file = model_dir / "feature_names.json"
        with open(features_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save metrics and config
        metadata_file = model_dir / "metadata.json"
        metadata = {
            'training_metrics': self.training_metrics,
            'config': self.config,
            'feature_count': len(self.feature_names),
            'model_type': 'lightgbm_scoreline_correction',
            'training_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"✅ Model saved to {model_dir}")
        self.logger.info(f"📊 Files created:")
        self.logger.info(f"   - {model_file}")
        self.logger.info(f"   - {features_file}")
        self.logger.info(f"   - {metadata_file}")
    
    def train_complete_pipeline(self) -> Dict[str, any]:
        """Run complete training pipeline"""
        self.logger.info("🚀 Starting scoreline correction training pipeline...")
        
        try:
            # Load data
            df = self.load_and_prepare_data()
            
            # Compute frequencies
            frequencies = self.compute_scoreline_frequencies(df)
            
            # Create training data
            training_df, feature_names = self.create_training_data(df, frequencies)
            
            if len(training_df) < 10:
                raise ValueError(f"Insufficient training data: {len(training_df)} samples")
            
            # Train model
            self.train_model(training_df, feature_names)
            
            # Save model
            self.save_model()
            
            # Compile results
            results = {
                'success': True,
                'model_type': 'scoreline_correction_v1',
                'training_samples': len(training_df),
                'feature_count': len(feature_names),
                'metrics': self.training_metrics,
                'model_path': self.config['model_dir']
            }
            
            self.logger.info("🎉 Scoreline correction training completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """Main execution"""
    print("🎯 Scoreline Correction Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ScorelineCorrectionTrainer()
    
    # Run training
    results = trainer.train_complete_pipeline()
    
    if results['success']:
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"📊 Model Type: {results['model_type']}")
        print(f"📈 Training Samples: {results['training_samples']}")
        print(f"🔧 Features: {results['feature_count']}")
        
        metrics = results['metrics']
        print(f"\n📊 Performance:")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        
        print(f"\n💾 Model saved to: {results['model_path']}")
    else:
        print(f"\n❌ TRAINING FAILED: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
