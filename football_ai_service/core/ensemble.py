"""
Ensemble Model - Комбиниране на Poisson, ML и Elo predictions
Football Intelligence Index (FII) - Интерпретируем индекс за качество
"""

import logging
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from .utils import setup_logging, sigmoid


class EnsembleModel:
    """
    Ensemble модел за комбиниране на множество predictions
    """
    
    def __init__(
        self,
        optimization_metric: str = 'log_loss',
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """
        Инициализация на Ensemble модел
        
        Args:
            optimization_metric: Метрика за оптимизация ('log_loss', 'accuracy')
            initial_weights: Начални тежести за моделите
        """
        self.logger = setup_logging()
        self.optimization_metric = optimization_metric
        
        # Default weights
        self.weights = initial_weights or {
            'poisson': 0.3,
            'ml': 0.5,
            'elo': 0.2
        }
        
        self.logger.info("EnsembleModel инициализиран")
    
    def optimize_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        method: str = 'SLSQP'
    ) -> Dict[str, float]:
        """
        Оптимизация на тежести за минимизиране на log loss
        
        Args:
            predictions: Dictionary с predictions от различни модели
            y_true: Истински labels
            method: Optimization метод
        
        Returns:
            Оптимизирани тежести
        """
        self.logger.info("Оптимизация на ensemble weights...")
        
        # Начални тежести
        initial = np.array([self.weights.get(k, 0.33) for k in predictions.keys()])
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(len(predictions))]
        
        # Objective function
        def objective(weights):
            combined = self._combine_predictions(predictions, weights)
            
            # Ensure valid probabilities
            combined = np.clip(combined, 1e-15, 1 - 1e-15)
            
            # Normalize to sum to 1 for each sample
            if combined.ndim == 2:
                combined = combined / combined.sum(axis=1, keepdims=True)
            
            try:
                return log_loss(y_true, combined)
            except:
                return 1e10  # Large penalty for invalid predictions
        
        # Optimize
        result = minimize(
            objective,
            initial,
            method=method,
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimized_weights = dict(zip(predictions.keys(), result.x))
            self.weights = optimized_weights
            
            self.logger.info("Оптимизация успешна!")
            for model, weight in optimized_weights.items():
                self.logger.info(f"  {model}: {weight:.4f}")
            
            return optimized_weights
        else:
            self.logger.warning("Оптимизация неуспешна, използваме начални тежести")
            return self.weights
    
    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Комбиниране на predictions с тежести
        
        Args:
            predictions: Dictionary с predictions
            weights: Тежести (ако None, използва self.weights)
        
        Returns:
            Комбинирани predictions
        """
        if weights is None:
            weights = np.array([self.weights.get(k, 0.33) for k in predictions.keys()])
        
        # Stack predictions
        pred_list = list(predictions.values())
        stacked = np.stack(pred_list, axis=-1)
        
        # Weighted average
        combined = np.average(stacked, axis=-1, weights=weights)
        
        return combined
    
    def predict(
        self,
        poisson_pred: np.ndarray,
        ml_pred: np.ndarray,
        elo_pred: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Ensemble prediction
        
        Args:
            poisson_pred: Poisson predictions
            ml_pred: ML model predictions
            elo_pred: Elo-based predictions (optional)
        
        Returns:
            Ensemble predictions
        """
        predictions = {
            'poisson': poisson_pred,
            'ml': ml_pred
        }
        
        if elo_pred is not None:
            predictions['elo'] = elo_pred
        
        return self._combine_predictions(predictions)


class FootballIntelligenceIndex:
    """
    Football Intelligence Index (FII) - Интерпретируем индекс за качество на прогнозата
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, List[float]]] = None
    ):
        """
        Инициализация на FII
        
        Args:
            weights: Тежести за различните компоненти
            thresholds: Прагове за Low/Medium/High
        """
        self.logger = setup_logging()
        
        # Default weights
        self.weights = weights or {
            'elo_diff': 0.25,
            'form_diff': 0.20,
            'xg_efficiency_diff': 0.20,
            'finishing_efficiency_diff': 0.15,
            'home_advantage': 0.20
        }
        
        # Default thresholds
        self.thresholds = thresholds or {
            'low': [0, 4],
            'medium': [4, 7],
            'high': [7, 10]
        }
        
        self.logger.info("FootballIntelligenceIndex инициализиран")
    
    def calculate_fii(
        self,
        elo_diff: float,
        form_diff: float,
        xg_efficiency_diff: float,
        finishing_efficiency_diff: float,
        is_home: int = 1
    ) -> Tuple[float, str]:
        """
        Изчисляване на FII
        
        Args:
            elo_diff: Разлика в Elo (home - away)
            form_diff: Разлика във форма
            xg_efficiency_diff: Разлика в xG efficiency
            finishing_efficiency_diff: Разлика във finishing
            is_home: Дали е домакин (1 или 0)
        
        Returns:
            Tuple (FII score, confidence level)
        """
        # Нормализация на компонентите
        elo_norm = self._normalize_elo_diff(elo_diff)
        form_norm = self._normalize_form_diff(form_diff)
        xg_norm = self._normalize_efficiency(xg_efficiency_diff)
        finishing_norm = self._normalize_efficiency(finishing_efficiency_diff)
        home_adv = is_home * 1.0  # 1 ако е домакин, 0 ако не
        
        # Weighted sum
        weighted_sum = (
            self.weights['elo_diff'] * elo_norm +
            self.weights['form_diff'] * form_norm +
            self.weights['xg_efficiency_diff'] * xg_norm +
            self.weights['finishing_efficiency_diff'] * finishing_norm +
            self.weights['home_advantage'] * home_adv
        )
        
        # Sigmoid трансформация за скала 0-10
        fii_score = 10 * sigmoid(weighted_sum)
        
        # Определяне на confidence level
        confidence = self._get_confidence_level(fii_score)
        
        return fii_score, confidence
    
    def calculate_fii_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Изчисляване на FII за целия dataset
        
        Args:
            df: DataFrame с необходимите колони
        
        Returns:
            DataFrame с добавени FII колони
        """
        self.logger.info(f"Изчисляване на FII за {len(df)} мача...")
        
        df = df.copy()
        
        # Изчисляване на разлики
        df['elo_diff_calc'] = df.get('home_elo_before', 0) - df.get('away_elo_before', 0)
        df['form_diff_calc'] = df.get('home_form_5', 0) - df.get('away_form_5', 0)
        
        # xG efficiency diff
        df['xg_eff_diff'] = df.get('home_xg_proxy', 0) - df.get('away_xg_proxy', 0)
        
        # Finishing efficiency diff
        df['finish_eff_diff'] = (
            df.get('home_shooting_efficiency', 0) - 
            df.get('away_shooting_efficiency', 0)
        )
        
        # Home advantage
        df['is_home_flag'] = df.get('is_home', 1)
        
        # Изчисляване на FII за всеки мач
        fii_scores = []
        confidence_levels = []
        
        for idx, row in df.iterrows():
            fii, conf = self.calculate_fii(
                elo_diff=row.get('elo_diff_calc', 0),
                form_diff=row.get('form_diff_calc', 0),
                xg_efficiency_diff=row.get('xg_eff_diff', 0),
                finishing_efficiency_diff=row.get('finish_eff_diff', 0),
                is_home=row.get('is_home_flag', 1)
            )
            fii_scores.append(fii)
            confidence_levels.append(conf)
        
        df['fii_score'] = fii_scores
        df['fii_confidence'] = confidence_levels
        
        self.logger.info(f"FII изчислен. Mean: {np.mean(fii_scores):.2f}")
        
        return df
    
    def _normalize_elo_diff(self, elo_diff: float) -> float:
        """Нормализация на Elo разлика (-400 до +400 -> -1 до +1)"""
        return np.clip(elo_diff / 400, -1, 1)
    
    def _normalize_form_diff(self, form_diff: float) -> float:
        """Нормализация на форма разлика (-1 до +1)"""
        return np.clip(form_diff, -1, 1)
    
    def _normalize_efficiency(self, efficiency_diff: float) -> float:
        """Нормализация на efficiency разлика"""
        # Clip на разумни граници
        return np.clip(efficiency_diff / 2, -1, 1)
    
    def _get_confidence_level(self, fii_score: float) -> str:
        """
        Определяне на confidence level
        
        Args:
            fii_score: FII score (0-10)
        
        Returns:
            'Low', 'Medium' или 'High'
        """
        if fii_score < self.thresholds['low'][1]:
            return 'Low'
        elif fii_score < self.thresholds['medium'][1]:
            return 'Medium'
        else:
            return 'High'
    
    def get_fii_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Разпределение на FII confidence levels
        
        Args:
            df: DataFrame с fii_confidence колона
        
        Returns:
            Dictionary с counts
        """
        if 'fii_confidence' not in df.columns:
            return {}
        
        distribution = df['fii_confidence'].value_counts().to_dict()
        
        self.logger.info("\nFII Confidence Distribution:")
        for level, count in distribution.items():
            pct = count / len(df) * 100
            self.logger.info(f"  {level}: {count} ({pct:.1f}%)")
        
        return distribution


class PredictionCombiner:
    """
    Клас за комбиниране на всички predictions и генериране на финален output
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.ensemble = EnsembleModel()
        self.fii = FootballIntelligenceIndex()
    
    def combine_all_predictions(
        self,
        df: pd.DataFrame,
        poisson_1x2: np.ndarray,
        ml_1x2: np.ndarray,
        poisson_ou25: np.ndarray,
        ml_ou25: np.ndarray,
        poisson_btts: np.ndarray,
        ml_btts: np.ndarray
    ) -> pd.DataFrame:
        """
        Комбиниране на всички predictions
        
        Args:
            df: Base DataFrame
            poisson_1x2: Poisson 1X2 predictions (N, 3)
            ml_1x2: ML 1X2 predictions (N, 3)
            poisson_ou25: Poisson OU2.5 predictions (N,)
            ml_ou25: ML OU2.5 predictions (N,)
            poisson_btts: Poisson BTTS predictions (N,)
            ml_btts: ML BTTS predictions (N,)
        
        Returns:
            DataFrame с всички predictions
        """
        self.logger.info("Комбиниране на всички predictions...")
        
        df = df.copy()
        
        # Ensemble 1X2
        ensemble_1x2 = self.ensemble.predict(poisson_1x2, ml_1x2)
        df['ensemble_prob_1'] = ensemble_1x2[:, 0]
        df['ensemble_prob_x'] = ensemble_1x2[:, 1]
        df['ensemble_prob_2'] = ensemble_1x2[:, 2]
        
        # Ensemble OU2.5
        ensemble_ou25 = self.ensemble.predict(
            poisson_ou25.reshape(-1, 1),
            ml_ou25.reshape(-1, 1)
        ).flatten()
        df['ensemble_prob_over25'] = ensemble_ou25
        
        # Ensemble BTTS
        ensemble_btts = self.ensemble.predict(
            poisson_btts.reshape(-1, 1),
            ml_btts.reshape(-1, 1)
        ).flatten()
        df['ensemble_prob_btts'] = ensemble_btts
        
        # Predicted outcomes
        df['ensemble_pred_1x2'] = ensemble_1x2.argmax(axis=1)
        df['ensemble_pred_ou25'] = (ensemble_ou25 > 0.5).astype(int)
        df['ensemble_pred_btts'] = (ensemble_btts > 0.5).astype(int)
        
        # FII
        df = self.fii.calculate_fii_batch(df)
        
        self.logger.info("Всички predictions комбинирани успешно")
        
        return df


if __name__ == "__main__":
    print("=== Ensemble & FII Test ===")
    print("Модулът е готов за използване!")
