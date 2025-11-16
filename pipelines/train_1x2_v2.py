#!/usr/bin/env python3
"""
1X2 v2 Training Pipeline

Complete training pipeline implementing all 5 upgrades:
1. Per-league 1X2 models (like OU2.5 per-league)
2. Replace multi-class with 3 binary models (Home Win, Draw, Away Win)
3. Poisson v2 upgrade with time-decay
4. Multi-class calibration
5. 1X2-specific features

Architecture:
- Separate models for each major league
- 3 binary models per league (homewin, draw, awaywin)
- Poisson v2 integration
- Advanced calibration
- Comprehensive feature engineering
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from core.data_loader import ESPNDataLoader
from core.feature_engineering import FeatureEngineer
from core.features_1x2 import Features1X2
from core.poisson_v2 import PoissonV2Model
from core.calibration_multiclass import MulticlassCalibrator
from core.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class Train1X2V2:
    """
    Complete 1X2 v2 training pipeline with all upgrades
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize 1X2 v2 training pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Major leagues for per-league modeling
        self.major_leagues = [
            'premier_league',
            'la_liga', 
            'serie_a',
            'bundesliga',
            'ligue_1',
            'eredivisie',
            'primeira_liga',
            'championship'
        ]
        
        # Initialize components
        self.data_loader = ESPNDataLoader()
        self.feature_engineering = FeatureEngineer()
        self.features_1x2 = Features1X2()
        
        # Model storage
        self.models = {}
        self.calibrators = {}
        self.poisson_models = {}
        self.feature_lists = {}
        self.metrics = {}
        
        logger.info(f"🏗️ Initialized 1X2 v2 training pipeline")
        logger.info(f"📊 Target leagues: {', '.join(self.major_leagues)}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'min_matches_per_league': 300,
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'n_folds': 5,
            'model_type': 'lightgbm',  # or 'xgboost'
            'calibration_method': 'temperature',  # 'temperature', 'vector', 'binary'
            'poisson_time_decay': 0.8,
            'feature_lookback_days': 365,
            'lgb_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'xgb_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0
            }
        }
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Load and prepare training data
        
        Returns:
            Prepared DataFrame with features and targets
        """
        logger.info("📂 Loading and preparing training data...")
        
        # Load match data
        df = self.data_loader.load_fixtures()
        
        if df is None or df.empty:
            raise ValueError("No training data available")
        
        logger.info(f"📊 Loaded {len(df)} matches")
        
        # Add league column (use league_id as league for now)
        df['league'] = df['league_id'].astype(str)
        
        # Add team names (use team_id as team names for now)
        df['home_team'] = df['home_team_id'].astype(str)
        df['away_team'] = df['away_team_id'].astype(str)
        
        # Filter recent data (last 3 years for training)
        cutoff_date = datetime.now() - timedelta(days=3*365)
        df = df[df['date'] >= cutoff_date].copy()
        
        logger.info(f"📅 Using {len(df)} matches from last 3 years")
        
        # Create target variables for binary classification
        df['target_homewin'] = (df['home_score'] > df['away_score']).astype(int)
        df['target_draw'] = (df['home_score'] == df['away_score']).astype(int)
        df['target_awaywin'] = (df['home_score'] < df['away_score']).astype(int)
        
        # Create multi-class target (0=home, 1=draw, 2=away)
        df['target_1x2'] = 0  # Default to home win
        df.loc[df['target_draw'] == 1, 'target_1x2'] = 1  # Draw
        df.loc[df['target_awaywin'] == 1, 'target_1x2'] = 2  # Away win
        
        logger.info("✅ Created binary and multi-class targets")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for 1X2 prediction
        
        Args:
            df: Match data DataFrame
            
        Returns:
            DataFrame with features added
        """
        logger.info("🔧 Creating comprehensive feature set...")
        
        # Initialize feature lists
        all_features = []
        
        # Process matches in batches for memory efficiency
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"📊 Processing batch {batch_idx + 1}/{total_batches}")
            
            batch_features = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Standard features (simplified for now)
                    standard_features = {
                        'home_team_id': float(row['home_team_id']),
                        'away_team_id': float(row['away_team_id']),
                        'league_id': float(row['league_id']),
                        'venue_id': float(row.get('venue_id', 0))
                    }
                    
                    # 1X2-specific features
                    x1x2_features = self.features_1x2.create_1x2_features(
                        row['home_team'], row['away_team'], row['league'],
                        df, row['date']
                    )
                    
                    # Combine features
                    combined_features = {**standard_features, **x1x2_features}
                    combined_features['match_id'] = idx
                    
                    batch_features.append(combined_features)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error creating features for match {idx}: {e}")
                    # Create default features
                    default_features = {f'feature_{i}': 0.0 for i in range(50)}
                    default_features['match_id'] = idx
                    batch_features.append(default_features)
            
            all_features.extend(batch_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df.set_index('match_id', inplace=True)
        
        # Merge with original data
        df_with_features = df.join(features_df, how='inner')
        
        logger.info(f"✅ Created {len(features_df.columns)} features for {len(df_with_features)} matches")
        
        return df_with_features
    
    def train_poisson_v2_models(self, df: pd.DataFrame) -> Dict[str, PoissonV2Model]:
        """
        Train Poisson v2 models for each league
        
        Args:
            df: Training data
            
        Returns:
            Dictionary of trained Poisson models
        """
        logger.info("🎯 Training Poisson v2 models...")
        
        poisson_models = {}
        
        # Global Poisson model
        global_poisson = PoissonV2Model(
            time_decay_factor=self.config['poisson_time_decay']
        )
        global_poisson.fit(df)
        poisson_models['global'] = global_poisson
        
        # Per-league Poisson models
        for league in self.major_leagues:
            league_data = df[df['league'] == league].copy()
            
            if len(league_data) >= self.config['min_matches_per_league']:
                logger.info(f"🏆 Training Poisson v2 for {league} ({len(league_data)} matches)")
                
                league_poisson = PoissonV2Model(
                    time_decay_factor=self.config['poisson_time_decay']
                )
                league_poisson.fit(league_data)
                poisson_models[league] = league_poisson
            else:
                logger.info(f"⚠️ Insufficient data for {league} ({len(league_data)} matches), using global model")
        
        self.poisson_models = poisson_models
        logger.info(f"✅ Trained {len(poisson_models)} Poisson v2 models")
        
        return poisson_models
    
    def train_binary_models_for_league(self, df: pd.DataFrame, league: str) -> Dict[str, any]:
        """
        Train 3 binary models for a specific league
        
        Args:
            df: Training data for the league
            league: League name
            
        Returns:
            Dictionary with trained models and metadata
        """
        logger.info(f"🏋️ Training binary models for {league}...")
        
        # Prepare features
        feature_columns = [col for col in df.columns if col.startswith(('home_', 'away_', 'league_', 'match_', 'expected_', 'form_', 'possession_', 'shot_', 'fatigue_', 'vulnerability_'))]
        
        # Remove non-numeric columns and handle missing values
        X = df[feature_columns].select_dtypes(include=[np.number]).fillna(0)
        
        # Store feature list
        self.feature_lists[league] = list(X.columns)
        
        models = {}
        metrics = {}
        
        # Train 3 binary models
        targets = ['target_homewin', 'target_draw', 'target_awaywin']
        model_names = ['homewin', 'draw', 'awaywin']
        
        for target, model_name in zip(targets, model_names):
            logger.info(f"🎯 Training {model_name} model for {league}")
            
            y = df[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['test_size'], 
                random_state=self.config['random_state'], stratify=y
            )
            
            # Train model
            if self.config['model_type'] == 'lightgbm':
                model = lgb.LGBMClassifier(**self.config['lgb_params'])
            else:
                model = xgb.XGBClassifier(**self.config['xgb_params'])
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            models[model_name] = model
            metrics[f'{model_name}_accuracy'] = accuracy
            metrics[f'{model_name}_logloss'] = logloss
            
            logger.info(f"✅ {model_name}: accuracy={accuracy:.4f}, logloss={logloss:.4f}")
        
        return {
            'models': models,
            'metrics': metrics,
            'feature_list': list(X.columns)
        }
    
    def train_calibration_for_league(self, df: pd.DataFrame, league: str, 
                                   models: Dict) -> MulticlassCalibrator:
        """
        Train calibration for league models
        
        Args:
            df: Validation data
            league: League name
            models: Trained binary models
            
        Returns:
            Trained calibrator
        """
        logger.info(f"🎛️ Training calibration for {league}...")
        
        # Prepare features
        feature_columns = self.feature_lists[league]
        X = df[feature_columns].fillna(0)
        
        # Get predictions from binary models
        pred_homewin = models['homewin'].predict_proba(X)[:, 1]
        pred_draw = models['draw'].predict_proba(X)[:, 1]
        pred_awaywin = models['awaywin'].predict_proba(X)[:, 1]
        
        # Combine predictions
        predictions = np.column_stack([pred_homewin, pred_draw, pred_awaywin])
        
        # Normalize predictions
        row_sums = predictions.sum(axis=1, keepdims=True)
        predictions = predictions / np.maximum(row_sums, 1e-15)
        
        # True labels
        y_true = df['target_1x2'].values
        
        # Train calibrator
        calibrator = MulticlassCalibrator(method=self.config['calibration_method'])
        calibrator.fit(predictions, y_true)
        
        logger.info(f"✅ Trained {self.config['calibration_method']} calibrator for {league}")
        
        return calibrator
    
    def train_all_models(self) -> None:
        """
        Train all models for all leagues
        """
        logger.info("🚀 Starting complete 1X2 v2 training pipeline...")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Create features
        df_with_features = self.create_features(df)
        
        # Train Poisson v2 models
        self.train_poisson_v2_models(df_with_features)
        
        # Train models for each league
        for league in self.major_leagues:
            league_data = df_with_features[df_with_features['league'] == league].copy()
            
            if len(league_data) >= self.config['min_matches_per_league']:
                logger.info(f"🏆 Training models for {league} ({len(league_data)} matches)")
                
                # Split data for training and calibration
                train_data, cal_data = train_test_split(
                    league_data, test_size=0.3, random_state=self.config['random_state']
                )
                
                # Train binary models
                league_models = self.train_binary_models_for_league(train_data, league)
                self.models[league] = league_models
                
                # Train calibration
                calibrator = self.train_calibration_for_league(
                    cal_data, league, league_models['models']
                )
                self.calibrators[league] = calibrator
                
                # Store metrics
                self.metrics[league] = league_models['metrics']
                
            else:
                logger.info(f"⚠️ Insufficient data for {league} ({len(league_data)} matches)")
        
        # Train global fallback model
        logger.info("🌍 Training global fallback model...")
        global_models = self.train_binary_models_for_league(df_with_features, 'global')
        self.models['global'] = global_models
        
        # Global calibration
        train_data, cal_data = train_test_split(
            df_with_features, test_size=0.3, random_state=self.config['random_state']
        )
        global_calibrator = self.train_calibration_for_league(
            cal_data, 'global', global_models['models']
        )
        self.calibrators['global'] = global_calibrator
        self.metrics['global'] = global_models['metrics']
        
        logger.info("✅ Completed training for all leagues!")
    
    def train_league(self, league: str, output_dir: Path) -> bool:
        """
        Train 1X2 v2 models for a single league
        
        Args:
            league: League slug
            output_dir: Output directory for models
        
        Returns:
            True if successful
        """
        try:
            logger.info(f"🎯 Training 1X2 v2 for {league}...")
            
            # Load and prepare data
            df = self.load_and_prepare_data()
            
            # Filter for league
            league_data = df[df['league'] == league].copy()
            
            if len(league_data) < self.config['min_matches_per_league']:
                logger.warning(f"⚠️ Insufficient data for {league}: {len(league_data)} matches")
                return False
            
            # Create features
            df_with_features = self.create_features(league_data)
            
            # Split data
            train_data, cal_data = train_test_split(
                df_with_features, test_size=0.3, random_state=self.config['random_state']
            )
            
            # Train binary models
            league_models = self.train_binary_models_for_league(train_data, league)
            
            # Train calibration
            calibrator = self.train_calibration_for_league(
                cal_data, league, league_models['models']
            )
            
            # Save models
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model in league_models['models'].items():
                model_file = output_dir / f"{model_name}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save calibrator
            calibrator_file = output_dir / "calibrator.pkl"
            calibrator.save_calibrator(str(calibrator_file))
            
            # Save feature list
            feature_file = output_dir / "feature_list.json"
            with open(feature_file, 'w') as f:
                json.dump(league_models.get('feature_list', []), f, indent=2)
            
            # Save metrics
            metrics_file = output_dir / "metrics.json"
            metrics = {
                "train": league_models.get('metrics', {}),
                "league": league,
                "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_type": "1x2_v2_binary"
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"✅ Successfully trained 1X2 v2 for {league}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {league}: {e}")
            return False
    
    def save_models(self, base_path: str = "models/leagues") -> None:
        """
        Save all trained models
        
        Args:
            base_path: Base directory for saving models
        """
        logger.info(f"💾 Saving models to {base_path}...")
        
        base_path = Path(base_path)
        
        for league in list(self.models.keys()):
            league_path = base_path / league / "1x2_v2"
            league_path.mkdir(parents=True, exist_ok=True)
            
            # Save binary models
            models = self.models[league]['models']
            for model_name, model in models.items():
                model_file = league_path / f"{model_name}_model.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save calibrator
            if league in self.calibrators:
                calibrator_file = league_path / "calibrator.pkl"
                self.calibrators[league].save_calibrator(str(calibrator_file))
            
            # Save feature list
            feature_file = league_path / "feature_list.json"
            with open(feature_file, 'w') as f:
                json.dump(self.feature_lists.get(league, []), f, indent=2)
            
            # Save metrics
            metrics_file = league_path / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics.get(league, {}), f, indent=2)
            
            logger.info(f"✅ Saved models for {league}")
        
        # Save Poisson models
        poisson_path = base_path / "poisson_v2"
        poisson_path.mkdir(parents=True, exist_ok=True)
        
        for league, poisson_model in self.poisson_models.items():
            poisson_file = poisson_path / f"{league}_poisson_v2.pkl"
            poisson_model.save_model(str(poisson_file))
        
        logger.info("💾 All models saved successfully!")
    
    def generate_training_report(self) -> Dict:
        """
        Generate comprehensive training report
        
        Returns:
            Training report dictionary
        """
        logger.info("📊 Generating training report...")
        
        report = {
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'leagues_trained': list(self.models.keys()),
            'total_leagues': len(self.models),
            'poisson_models': len(self.poisson_models),
            'metrics_summary': {}
        }
        
        # Aggregate metrics
        for league, metrics in self.metrics.items():
            report['metrics_summary'][league] = {
                'homewin_accuracy': metrics.get('homewin_accuracy', 0),
                'draw_accuracy': metrics.get('draw_accuracy', 0),
                'awaywin_accuracy': metrics.get('awaywin_accuracy', 0),
                'avg_accuracy': np.mean([
                    metrics.get('homewin_accuracy', 0),
                    metrics.get('draw_accuracy', 0),
                    metrics.get('awaywin_accuracy', 0)
                ])
            }
        
        # Save report
        report_path = Path("logs/1x2_v2_reports")
        report_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"training_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Training report saved to {report_file}")
        
        return report


def main():
    """
    Main training execution
    """
    logger.info("🚀 Starting 1X2 v2 Training Pipeline")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = Train1X2V2()
    
    try:
        # Train all models
        trainer.train_all_models()
        
        # Save models
        trainer.save_models()
        
        # Generate report
        report = trainer.generate_training_report()
        
        # Print summary
        logger.info("📊 TRAINING SUMMARY")
        logger.info("=" * 30)
        logger.info(f"✅ Leagues trained: {report['total_leagues']}")
        logger.info(f"✅ Poisson models: {report['poisson_models']}")
        
        for league, metrics in report['metrics_summary'].items():
            avg_acc = metrics['avg_accuracy']
            logger.info(f"🏆 {league}: {avg_acc:.3f} avg accuracy")
        
        logger.info("🎉 1X2 v2 training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
