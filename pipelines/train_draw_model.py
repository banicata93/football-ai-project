#!/usr/bin/env python3
"""
Draw Model Training Pipeline

Trains a specialized binary classifier for draw prediction (DRAW vs NON-DRAW).
This is ADDITIVE - does not modify existing 1X2 models.

Uses LightGBM binary classifier with draw-specific features.
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

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import ESPNDataLoader
from core.draw_features import DrawFeatures
from core.utils import setup_logging


class DrawModelTrainer:
    """
    Trainer for specialized draw prediction model
    
    Creates binary classifier: DRAW (1) vs NON-DRAW (0)
    Uses draw-specific features and LightGBM.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize draw model trainer
        
        Args:
            config: Training configuration
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Components
        self.data_loader = ESPNDataLoader()
        self.draw_features = DrawFeatures(
            lookback_days=self.config.get('data', {}).get('feature_lookback_days', 180)
        )
        
        # Model storage
        self.model = None
        self.feature_list = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        self.logger.info("🎯 Initialized Draw Model Trainer")
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            # Data parameters
            'lookback_years': 3,
            'feature_lookback_days': 180,
            'min_matches_per_team': 10,
            'test_size_months': 6,
            
            # Model parameters
            'model_type': 'lightgbm',
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
            'n_estimators': 200,
            'early_stopping_rounds': 20,
            
            # Validation parameters
            'cv_folds': 5,
            'calibration': True,
            
            # Output parameters
            'output': {
                'model_dir': 'models/draw_model_v1',
                'log_file': 'logs/draw_training.log'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = Path(self.config.get('output', {}).get('log_file', 'logs/draw_model_training.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('DrawModelTrainer')
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
        """
        Load and prepare training data
        
        Returns:
            Prepared DataFrame with features and targets
        """
        self.logger.info("📂 Loading and preparing training data...")
        
        # Load match data
        df = self.data_loader.load_fixtures()
        
        if df is None or df.empty:
            raise ValueError("No training data available")
        
        self.logger.info(f"📊 Loaded {len(df)} matches")
        
        # Add required columns
        df['league'] = df['league_id'].astype(str)
        df['home_team'] = df['home_team_id'].astype(str)
        df['away_team'] = df['away_team_id'].astype(str)
        
        # Filter recent data
        lookback_years = self.config.get('data', {}).get('lookback_years', 3)
        cutoff_date = datetime.now() - timedelta(days=lookback_years * 365)
        df = df[df['date'] >= cutoff_date].copy()
        
        self.logger.info(f"📅 Using {len(df)} matches from last {lookback_years} years")
        
        # Create draw target
        df['is_draw'] = (df['home_score'] == df['away_score']).astype(int)
        
        draw_count = df['is_draw'].sum()
        draw_rate = draw_count / len(df)
        self.logger.info(f"🎯 Draw rate: {draw_rate:.1%} ({draw_count}/{len(df)} matches)")
        
        # Validate minimum data requirements
        min_total_matches = 1000
        min_draw_matches = 200
        
        if len(df) < min_total_matches:
            raise ValueError(f"Insufficient data: {len(df)} matches < {min_total_matches} required")
        
        if draw_count < min_draw_matches:
            raise ValueError(f"Insufficient draw data: {draw_count} draws < {min_draw_matches} required")
        
        self.logger.info(f"✅ Data validation passed: {len(df)} matches, {draw_count} draws")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features for all matches
        
        Args:
            df: Match data
            
        Returns:
            Tuple of (features_df, feature_names)
        """
        self.logger.info("🔧 Creating draw-specific features...")
        
        features_list = []
        batch_size = 500  # Reduced for stability
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        self.logger.info(f"📊 Processing {len(df)} matches in {total_batches} batches of {batch_size}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            self.logger.info(f"📊 Processing batch {batch_idx + 1}/{total_batches}")
            
            batch_features = []
            
            for idx, row in batch_df.iterrows():
                try:
                    # Create draw features
                    draw_features = self.draw_features.create_draw_features(
                        row['home_team'], row['away_team'], row['league'],
                        df, row['date']
                    )
                    
                    # Add basic match info
                    match_features = {
                        'match_id': idx,
                        'is_draw': row['is_draw'],
                        **draw_features
                    }
                    
                    batch_features.append(match_features)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error creating features for match {idx}: {e}")
                    continue
            
            features_list.extend(batch_features)
        
        # Convert to DataFrame
        if not features_list:
            raise ValueError("No valid features created - all matches failed processing")
        
        features_df = pd.DataFrame(features_list)
        
        # Validate features DataFrame
        if features_df.empty:
            raise ValueError("Features DataFrame is empty")
        
        # Get feature names (exclude match_id and target)
        feature_names = [col for col in features_df.columns 
                        if col not in ['match_id', 'is_draw']]
        
        if len(feature_names) < 5:
            raise ValueError(f"Too few features created: {len(feature_names)} < 5 required")
        
        # Check for missing values
        missing_pct = features_df[feature_names].isnull().sum().sum() / (len(features_df) * len(feature_names))
        if missing_pct > 0.5:
            self.logger.warning(f"⚠️ High missing values: {missing_pct:.1%}")
        
        # Fill missing values
        features_df[feature_names] = features_df[feature_names].fillna(0.5)
        
        self.logger.info(f"✅ Created {len(feature_names)} features for {len(features_df)} matches")
        self.logger.info(f"📊 Missing values filled: {missing_pct:.1%}")
        
        return features_df, feature_names
    
    def split_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets using time-based split
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        self.logger.info("📊 Splitting data into train/test sets...")
        
        # Sort by match_id (proxy for time)
        features_df = features_df.sort_values('match_id')
        
        # Time-based split
        test_size = int(len(features_df) * 0.2)  # 20% for test
        
        train_df = features_df.iloc[:-test_size].copy()
        test_df = features_df.iloc[-test_size:].copy()
        
        train_draw_rate = train_df['is_draw'].mean()
        test_draw_rate = test_df['is_draw'].mean()
        
        self.logger.info(f"📈 Train set: {len(train_df)} matches, draw rate: {train_draw_rate:.1%}")
        self.logger.info(f"📈 Test set: {len(test_df)} matches, draw rate: {test_draw_rate:.1%}")
        
        return train_df, test_df
    
    def train_model(self, train_df: pd.DataFrame, feature_names: List[str]) -> None:
        """
        Train the draw prediction model
        
        Args:
            train_df: Training data
            feature_names: List of feature names
        """
        self.logger.info("🏋️ Training draw prediction model...")
        
        # Prepare training data
        X_train = train_df[feature_names].values
        y_train = train_df['is_draw'].values
        
        self.logger.info(f"📊 Training on {len(X_train)} samples with {len(feature_names)} features")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Get training parameters
        lgb_params = self.config.get('model', {}).get('lightgbm_params', {})
        training_config = self.config.get('model', {}).get('training', {})
        
        # Train model
        self.model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=training_config.get('n_estimators', 100),
            callbacks=[lgb.early_stopping(training_config.get('early_stopping_rounds', 15))],
            valid_sets=[train_data],
            valid_names=['train']
        )
        
        # Apply calibration if requested
        if training_config.get('calibration', True):
            self.logger.info("🔧 Applying probability calibration...")
            
            # Create sklearn-compatible wrapper
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class LGBWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, lgb_model):
                    self.lgb_model = lgb_model
                    self.classes_ = np.array([0, 1])  # Required for ClassifierMixin
                    self.n_classes_ = 2  # Required for ClassifierMixin
                
                def fit(self, X, y):
                    # Already fitted, but set required attributes
                    self.classes_ = np.unique(y)
                    self.n_classes_ = len(self.classes_)
                    return self
                
                def predict_proba(self, X):
                    preds = self.lgb_model.predict(X)
                    # Ensure predictions are probabilities [0, 1]
                    preds = np.clip(preds, 1e-7, 1 - 1e-7)
                    # Convert to 2D array for binary classification
                    return np.column_stack([1 - preds, preds])
                
                def predict(self, X):
                    return (self.lgb_model.predict(X) > 0.5).astype(int)
                
                def decision_function(self, X):
                    # Required for some calibration methods
                    preds = self.lgb_model.predict(X)
                    # Convert probabilities to decision function scores
                    return np.log(np.clip(preds / (1 - preds), 1e-7, 1e7))
                
                @property
                def _estimator_type(self):
                    return "classifier"
            
            lgb_wrapper = LGBWrapper(self.model)
            
            # Test wrapper before calibration
            try:
                # Fit wrapper with training data
                lgb_wrapper.fit(X_train, y_train)
                
                # Test predict_proba on small sample
                test_sample = X_train[:10] if len(X_train) > 10 else X_train
                test_proba = lgb_wrapper.predict_proba(test_sample)
                
                if test_proba.shape[1] != 2:
                    raise ValueError(f"predict_proba should return 2 columns, got {test_proba.shape[1]}")
                
                self.logger.info(f"✅ LGBWrapper test passed: {test_proba.shape}")
                
            except Exception as e:
                self.logger.error(f"❌ LGBWrapper test failed: {e}")
                self.logger.warning("⚠️ Skipping calibration due to wrapper issues")
                return
            
            # Apply calibration
            try:
                calibrated_model = CalibratedClassifierCV(
                    lgb_wrapper, method='isotonic', cv=3
                )
                calibrated_model.fit(X_train, y_train)
                
                self.logger.info("✅ Calibration applied successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Calibration failed: {e}")
                self.logger.warning("⚠️ Using uncalibrated model")
                return
            
            self.model = calibrated_model
            self.logger.info("✅ Model calibration completed")
        
        # Store feature list
        self.feature_list = feature_names
        
        self.logger.info("✅ Model training completed")
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            test_df: Test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info("📊 Evaluating model performance...")
        
        # Prepare test data
        X_test = test_df[self.feature_list].values
        y_test = test_df['is_draw'].values
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba)
        }
        
        # Draw-specific metrics
        draw_mask = y_test == 1
        non_draw_mask = y_test == 0
        
        if draw_mask.sum() > 0:
            metrics['draw_recall'] = recall_score(y_test[draw_mask], y_pred[draw_mask], zero_division=0)
            metrics['draw_precision'] = precision_score(y_test[draw_mask], y_pred[draw_mask], zero_division=0)
        
        if non_draw_mask.sum() > 0:
            metrics['non_draw_recall'] = recall_score(y_test[non_draw_mask], y_pred[non_draw_mask], zero_division=0)
        
        # Log metrics
        self.logger.info("📊 Model Performance:")
        self.logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
        self.logger.info(f"   Draw Recall: {metrics.get('draw_recall', 0):.3f}")
        self.logger.info(f"   Draw Precision: {metrics.get('draw_precision', 0):.3f}")
        self.logger.info(f"   ROC AUC: {metrics['roc_auc']:.3f}")
        self.logger.info(f"   Log Loss: {metrics['log_loss']:.3f}")
        self.logger.info(f"   Brier Score: {metrics['brier_score']:.3f}")
        
        # Detailed classification report
        self.logger.info("\n📋 Classification Report:")
        report = classification_report(y_test, y_pred, target_names=['Non-Draw', 'Draw'])
        self.logger.info(f"\n{report}")
        
        self.validation_metrics = metrics
        return metrics
    
    def cross_validate(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Cross-validation metrics
        """
        self.logger.info(f"🔄 Performing {self.config['cv_folds']}-fold cross-validation...")
        
        X = features_df[self.feature_list].values
        y = features_df['is_draw'].values
        
        # Time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"   Fold {fold + 1}/{self.config['cv_folds']}")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train fold model
            train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
            fold_model = lgb.train(
                params=self.config['lgb_params'],
                train_set=train_data,
                num_boost_round=self.config['n_estimators'],
                callbacks=[lgb.early_stopping(self.config['early_stopping_rounds'])],
                valid_sets=[train_data],
                valid_names=['train']
            )
            
            # Predict
            y_pred_proba = fold_model.predict(X_val_cv)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val_cv, y_pred))
            cv_scores['precision'].append(precision_score(y_val_cv, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_cv, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_cv, y_pred, zero_division=0))
            cv_scores['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        self.logger.info("✅ Cross-validation results:")
        self.logger.info(f"   Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
        self.logger.info(f"   Draw Recall: {cv_results['recall_mean']:.3f} ± {cv_results['recall_std']:.3f}")
        self.logger.info(f"   ROC AUC: {cv_results['roc_auc_mean']:.3f} ± {cv_results['roc_auc_std']:.3f}")
        
        return cv_results
    
    def save_model(self) -> None:
        """Save the trained model and metadata"""
        self.logger.info("💾 Saving draw model...")
        
        model_dir = Path(self.config.get('output', {}).get('model_path', 'models/draw_model_v1/draw_model.pkl')).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "draw_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save feature list
        feature_file = model_dir / "feature_list.json"
        with open(feature_file, 'w') as f:
            json.dump(self.feature_list, f, indent=2)
        
        # Save metrics
        metrics_file = model_dir / "metrics.json"
        all_metrics = {
            'validation_metrics': self.validation_metrics,
            'training_config': self.config,
            'feature_count': len(self.feature_list),
            'model_type': 'lightgbm_binary_draw_classifier',
            'training_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        self.logger.info(f"✅ Model saved to {model_dir}")
        self.logger.info(f"📊 Files created:")
        self.logger.info(f"   - {model_file}")
        self.logger.info(f"   - {feature_file}")
        self.logger.info(f"   - {metrics_file}")
    
    def train_complete_pipeline(self) -> Dict[str, any]:
        """
        Run the complete training pipeline
        
        Returns:
            Training results
        """
        self.logger.info("🚀 Starting complete draw model training pipeline...")
        
        try:
            # Load and prepare data
            df = self.load_and_prepare_data()
            
            # Create features
            features_df, feature_names = self.create_features(df)
            
            if len(features_df) < 100:
                raise ValueError(f"Insufficient training data: {len(features_df)} samples")
            
            # Split data
            train_df, test_df = self.split_data(features_df)
            
            # Train model
            self.train_model(train_df, feature_names)
            
            # Evaluate model
            test_metrics = self.evaluate_model(test_df)
            
            # Cross-validate
            cv_metrics = self.cross_validate(features_df)
            
            # Save model
            self.save_model()
            
            # Compile results
            results = {
                'success': True,
                'model_type': 'draw_specialist_v1',
                'training_samples': len(train_df),
                'test_samples': len(test_df),
                'feature_count': len(feature_names),
                'test_metrics': test_metrics,
                'cv_metrics': cv_metrics,
                'model_path': self.config['model_dir']
            }
            
            self.logger.info("🎉 Draw model training completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """
    Main execution function
    """
    print("🎯 Draw Model Training Pipeline")
    print("=" * 50)
    
    # Initialize trainer
    trainer = DrawModelTrainer()
    
    # Run training
    results = trainer.train_complete_pipeline()
    
    if results['success']:
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"📊 Model Type: {results['model_type']}")
        print(f"📈 Training Samples: {results['training_samples']}")
        print(f"📈 Test Samples: {results['test_samples']}")
        print(f"🔧 Features: {results['feature_count']}")
        
        test_metrics = results['test_metrics']
        print(f"\n📊 Test Performance:")
        print(f"   Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"   Draw Recall: {test_metrics.get('draw_recall', 0):.3f}")
        print(f"   ROC AUC: {test_metrics['roc_auc']:.3f}")
        print(f"   Log Loss: {test_metrics['log_loss']:.3f}")
        
        cv_metrics = results['cv_metrics']
        print(f"\n🔄 Cross-Validation:")
        print(f"   Accuracy: {cv_metrics['accuracy_mean']:.3f} ± {cv_metrics['accuracy_std']:.3f}")
        print(f"   Draw Recall: {cv_metrics['recall_mean']:.3f} ± {cv_metrics['recall_std']:.3f}")
        
        print(f"\n💾 Model saved to: {results['model_path']}")
    else:
        print(f"\n❌ TRAINING FAILED: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
