"""
Stacking Ensemble (Meta-Learning) - Интелигентно комбиниране на модели

Използва мета-модели (LogisticRegression, XGBoost) които се учат да комбинират
predictions от базовите модели (Poisson, ML, Elo) за оптимална точност.
"""

import logging
from typing import Dict, Tuple, Optional, List, Any
import pickle
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import xgboost as xgb

from .utils import setup_logging


class StackingEnsemble:
    """
    Stacking Ensemble с мета-модели за комбиниране на базови predictions
    
    Архитектура:
    - Level 0: Базови модели (Poisson, XGBoost, LightGBM, Elo)
    - Level 1: Мета-модели които се учат да комбинират Level 0 predictions
    
    За всеки target (1X2, OU2.5, BTTS) има отделен мета-модел.
    """
    
    def __init__(
        self,
        meta_model_type: str = 'logistic',
        calibrate: bool = True,
        use_original_features: bool = False
    ):
        """
        Инициализация на Stacking Ensemble
        
        Args:
            meta_model_type: Тип на мета-модел ('logistic', 'xgboost')
            calibrate: Дали да калибрира вероятностите
            use_original_features: Дали да използва и оригинални features
        """
        self.logger = setup_logging()
        self.meta_model_type = meta_model_type
        self.calibrate = calibrate
        self.use_original_features = use_original_features
        
        # Мета-модели за различните targets
        self.meta_models = {
            '1x2': None,
            'ou25': None,
            'btts': None
        }
        
        # Feature importance от мета-моделите
        self.feature_importance = {}
        
        self.logger.info(f"StackingEnsemble инициализиран (meta_model={meta_model_type})")
    
    def _create_meta_model(self, task_type: str = 'multiclass', n_classes: int = 3):
        """
        Създаване на мета-модел
        
        Args:
            task_type: 'multiclass' или 'binary'
            n_classes: Брой класове
        
        Returns:
            Мета-модел
        """
        if self.meta_model_type == 'logistic':
            if task_type == 'multiclass':
                model = LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    C=1.0
                )
            else:
                model = LogisticRegression(
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=42,
                    C=1.0
                )
            
            # Калибриране ако е необходимо
            if self.calibrate:
                model = CalibratedClassifierCV(
                    model,
                    method='isotonic',
                    cv=5
                )
        
        elif self.meta_model_type == 'xgboost':
            if task_type == 'multiclass':
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss'
                )
            else:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
        
        else:
            raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")
        
        return model
    
    def _prepare_meta_features(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Подготовка на features за мета-модела
        
        Args:
            base_predictions: Predictions от базовите модели
            original_features: Оригинални features (опционално)
        
        Returns:
            Meta-features array
        """
        meta_features_list = []
        
        # Добавяме predictions от всички базови модели
        for model_name, preds in base_predictions.items():
            if preds.ndim == 1:
                # Binary predictions
                meta_features_list.append(preds.reshape(-1, 1))
            else:
                # Multi-class predictions
                meta_features_list.append(preds)
        
        # Опционално: добавяме оригинални features
        if self.use_original_features and original_features is not None:
            # Избираме само важните features
            important_cols = [
                'home_elo_before', 'away_elo_before', 'elo_diff',
                'home_form_5', 'away_form_5',
                'home_goals_scored_avg_5', 'away_goals_scored_avg_5',
                'home_shooting_efficiency', 'away_shooting_efficiency'
            ]
            
            available_cols = [col for col in important_cols if col in original_features.columns]
            if available_cols:
                meta_features_list.append(original_features[available_cols].values)
        
        # Concatenate всички features
        meta_features = np.hstack(meta_features_list)
        
        # Handle NaN values (replace with 0 or mean)
        if np.any(np.isnan(meta_features)):
            meta_features = np.nan_to_num(meta_features, nan=0.0)
        
        return meta_features
    
    def train_1x2(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        original_features: Optional[pd.DataFrame] = None
    ):
        """
        Обучение на мета-модел за 1X2 predictions
        
        Args:
            base_predictions: Dict с predictions от базови модели
                - 'poisson': (N, 3) array
                - 'ml': (N, 3) array
                - 'elo': (N, 3) array (опционално)
            y_true: True labels (N,)
            original_features: Оригинални features (опционално)
        """
        self.logger.info("Обучение на мета-модел за 1X2...")
        
        # Подготовка на meta-features
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        
        self.logger.info(f"Meta-features shape: {X_meta.shape}")
        
        # Създаване и обучение на мета-модел
        self.meta_models['1x2'] = self._create_meta_model('multiclass', n_classes=3)
        self.meta_models['1x2'].fit(X_meta, y_true)
        
        # Feature importance (само за XGBoost)
        if self.meta_model_type == 'xgboost':
            self.feature_importance['1x2'] = self.meta_models['1x2'].feature_importances_
        
        self.logger.info("✓ Мета-модел за 1X2 обучен")
    
    def train_ou25(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        original_features: Optional[pd.DataFrame] = None
    ):
        """
        Обучение на мета-модел за Over/Under 2.5
        
        Args:
            base_predictions: Dict с predictions от базови модели
                - 'poisson': (N,) array (prob over)
                - 'ml': (N,) array (prob over)
            y_true: True labels (N,)
            original_features: Оригинални features (опционално)
        """
        self.logger.info("Обучение на мета-модел за OU2.5...")
        
        # Подготовка на meta-features
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        
        self.logger.info(f"Meta-features shape: {X_meta.shape}")
        
        # Създаване и обучение на мета-модел
        self.meta_models['ou25'] = self._create_meta_model('binary')
        self.meta_models['ou25'].fit(X_meta, y_true)
        
        # Feature importance
        if self.meta_model_type == 'xgboost':
            self.feature_importance['ou25'] = self.meta_models['ou25'].feature_importances_
        
        self.logger.info("✓ Мета-модел за OU2.5 обучен")
    
    def train_btts(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        original_features: Optional[pd.DataFrame] = None
    ):
        """
        Обучение на мета-модел за BTTS
        
        Args:
            base_predictions: Dict с predictions от базови модели
                - 'poisson': (N,) array (prob yes)
                - 'ml': (N,) array (prob yes)
            y_true: True labels (N,)
            original_features: Оригинални features (опционално)
        """
        self.logger.info("Обучение на мета-модел за BTTS...")
        
        # Подготовка на meta-features
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        
        self.logger.info(f"Meta-features shape: {X_meta.shape}")
        
        # Създаване и обучение на мета-модел
        self.meta_models['btts'] = self._create_meta_model('binary')
        self.meta_models['btts'].fit(X_meta, y_true)
        
        # Feature importance
        if self.meta_model_type == 'xgboost':
            self.feature_importance['btts'] = self.meta_models['btts'].feature_importances_
        
        self.logger.info("✓ Мета-модел за BTTS обучен")
    
    def predict_1x2(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Prediction за 1X2 с мета-модел
        
        Returns:
            (N, 3) array с вероятности за Home/Draw/Away
        """
        if self.meta_models['1x2'] is None:
            raise ValueError("Мета-моделът за 1X2 не е обучен")
        
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        return self.meta_models['1x2'].predict_proba(X_meta)
    
    def predict_ou25(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Prediction за OU2.5 с мета-модел
        
        Returns:
            (N,) array с вероятности за Over
        """
        if self.meta_models['ou25'] is None:
            raise ValueError("Мета-моделът за OU2.5 не е обучен")
        
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        return self.meta_models['ou25'].predict_proba(X_meta)[:, 1]
    
    def predict_btts(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Prediction за BTTS с мета-модел
        
        Returns:
            (N,) array с вероятности за Yes
        """
        if self.meta_models['btts'] is None:
            raise ValueError("Мета-моделът за BTTS не е обучен")
        
        X_meta = self._prepare_meta_features(base_predictions, original_features)
        return self.meta_models['btts'].predict_proba(X_meta)[:, 1]
    
    def evaluate(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        task: str,
        original_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Оценка на мета-модел
        
        Args:
            base_predictions: Predictions от базови модели
            y_true: True labels
            task: '1x2', 'ou25' или 'btts'
            original_features: Оригинални features
        
        Returns:
            Dictionary с метрики
        """
        if task == '1x2':
            y_pred_proba = self.predict_1x2(base_predictions, original_features)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'log_loss': log_loss(y_true, y_pred_proba)
            }
        
        elif task in ['ou25', 'btts']:
            if task == 'ou25':
                y_pred_proba = self.predict_ou25(base_predictions, original_features)
            else:
                y_pred_proba = self.predict_btts(base_predictions, original_features)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'log_loss': log_loss(y_true, np.column_stack([1 - y_pred_proba, y_pred_proba])),
                'roc_auc': roc_auc_score(y_true, y_pred_proba)
            }
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return metrics
    
    def save(self, output_dir: str):
        """
        Запазване на мета-моделите
        
        Args:
            output_dir: Директория за запазване
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Запазване на мета-моделите
        for task, model in self.meta_models.items():
            if model is not None:
                model_path = output_path / f"meta_model_{task}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"✓ Мета-модел за {task} запазен: {model_path}")
        
        # Запазване на конфигурация
        config = {
            'meta_model_type': self.meta_model_type,
            'calibrate': self.calibrate,
            'use_original_features': self.use_original_features,
            'feature_importance': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                   for k, v in self.feature_importance.items()}
        }
        
        config_path = output_path / "stacking_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"✓ Конфигурация запазена: {config_path}")
    
    def load(self, input_dir: str):
        """
        Зареждане на мета-моделите
        
        Args:
            input_dir: Директория за зареждане
        """
        input_path = Path(input_dir)
        
        # Зареждане на конфигурация
        config_path = input_path / "stacking_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.meta_model_type = config['meta_model_type']
            self.calibrate = config['calibrate']
            self.use_original_features = config['use_original_features']
            self.feature_importance = config.get('feature_importance', {})
        
        # Зареждане на мета-моделите
        for task in ['1x2', 'ou25', 'btts']:
            model_path = input_path / f"meta_model_{task}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.meta_models[task] = pickle.load(f)
                self.logger.info(f"✓ Мета-модел за {task} зареден")
        
        self.logger.info("✓ Stacking Ensemble зареден")


def compare_ensembles(
    weighted_predictions: Dict[str, np.ndarray],
    stacking_predictions: Dict[str, np.ndarray],
    y_true: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Сравнение на Weighted Ensemble vs Stacking Ensemble
    
    Args:
        weighted_predictions: Predictions от weighted ensemble
        stacking_predictions: Predictions от stacking ensemble
        y_true: True labels
    
    Returns:
        DataFrame със сравнение на метриките
    """
    results = []
    
    for task in ['1x2', 'ou25', 'btts']:
        # Weighted ensemble metrics
        if task == '1x2':
            weighted_acc = accuracy_score(y_true[task], np.argmax(weighted_predictions[task], axis=1))
            weighted_ll = log_loss(y_true[task], weighted_predictions[task])
            
            stacking_acc = accuracy_score(y_true[task], np.argmax(stacking_predictions[task], axis=1))
            stacking_ll = log_loss(y_true[task], stacking_predictions[task])
            
            results.append({
                'Task': task.upper(),
                'Weighted_Accuracy': weighted_acc,
                'Stacking_Accuracy': stacking_acc,
                'Accuracy_Improvement': stacking_acc - weighted_acc,
                'Weighted_LogLoss': weighted_ll,
                'Stacking_LogLoss': stacking_ll,
                'LogLoss_Improvement': weighted_ll - stacking_ll
            })
        
        else:
            weighted_acc = accuracy_score(y_true[task], (weighted_predictions[task] > 0.5).astype(int))
            weighted_ll = log_loss(y_true[task], 
                                   np.column_stack([1 - weighted_predictions[task], 
                                                   weighted_predictions[task]]))
            
            stacking_acc = accuracy_score(y_true[task], (stacking_predictions[task] > 0.5).astype(int))
            stacking_ll = log_loss(y_true[task],
                                   np.column_stack([1 - stacking_predictions[task],
                                                   stacking_predictions[task]]))
            
            results.append({
                'Task': task.upper(),
                'Weighted_Accuracy': weighted_acc,
                'Stacking_Accuracy': stacking_acc,
                'Accuracy_Improvement': stacking_acc - weighted_acc,
                'Weighted_LogLoss': weighted_ll,
                'Stacking_LogLoss': stacking_ll,
                'LogLoss_Improvement': weighted_ll - stacking_ll
            })
    
    return pd.DataFrame(results)
