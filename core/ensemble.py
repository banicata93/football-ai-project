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


def squash_prob_ou25(p: float, factor: float = 0.75) -> float:
    """
    Probability squashing function for OU2.5 to reduce overconfidence
    
    Args:
        p: Original probability
        factor: Squashing factor (0.75 = 25% reduction in extremeness)
        
    Returns:
        Squashed probability closer to 0.5
    """
    return 0.5 + (p - 0.5) * factor


# Historical league base rates for OU2.5
LEAGUE_OU25_RATES = {
    'Premier League': 0.57,
    'La Liga': 0.55,
    'Serie A': 0.56,
    'Bundesliga': 0.58,
    'Ligue 1': 0.56,
    'Eredivisie': 0.58,
    'Primeira Liga': 0.57,
    'Championship': 0.56,
    'default': 0.56
}


def get_league_ou25_rate(league: str) -> float:
    """Get historical OU2.5 rate for league"""
    return LEAGUE_OU25_RATES.get(league, LEAGUE_OU25_RATES['default'])


def apply_base_rate_regularization_ou25(prob: float, league_rate: float, weight: float = 0.15) -> float:
    """
    Apply base rate regularization for OU2.5
    
    Args:
        prob: Current probability
        league_rate: Historical league OU2.5 rate
        weight: Weight for regularization (0.15 = 15% league prior)
    
    Returns:
        Regularized probability
    """
    return (1 - weight) * prob + weight * league_rate


def apply_disagreement_penalty_ou25(prob: float, ml_prob: float, poisson_prob: float, threshold: float = 0.20) -> float:
    """
    Apply penalty when ML and Poisson strongly disagree for OU2.5
    
    Args:
        prob: Current probability
        ml_prob: ML model probability
        poisson_prob: Poisson model probability
        threshold: Disagreement threshold
    
    Returns:
        Penalized probability
    """
    if abs(ml_prob - poisson_prob) > threshold:
        return prob * 0.7 + 0.5 * 0.3
    return prob


def apply_soft_caps_ou25(prob: float, upper: float = 0.85, lower: float = 0.15) -> float:
    """
    Apply soft confidence caps for OU2.5
    
    Args:
        prob: Current probability
        upper: Upper cap
        lower: Lower cap
    
    Returns:
        Capped probability
    """
    if prob > upper:
        return upper
    if prob < lower:
        return lower
    return prob


class EnsembleModel:
    """
    Ensemble модел за комбиниране на множество predictions
    """
    
    def __init__(
        self,
        optimization_metric: str = 'log_loss',
        initial_weights: Optional[Dict[str, float]] = None,
        dynamic: bool = True,
        per_league_weights: Optional[Dict[int, Dict[str, float]]] = None
    ):
        """
        Инициализация на Ensemble модел
        
        Args:
            optimization_metric: Метрика за оптимизация ('log_loss', 'accuracy')
            initial_weights: Начални тежести за моделите
            dynamic: Дали да използва динамични корекции
            per_league_weights: Per-league тежести
        """
        self.logger = setup_logging()
        self.optimization_metric = optimization_metric
        self.dynamic = dynamic
        self.per_league_weights = per_league_weights or {}
        
        # Default weights
        self.weights = initial_weights or {
            'poisson': 0.3,
            'ml': 0.5,
            'elo': 0.2
        }
        
        self.logger.info(f"EnsembleModel инициализиран (dynamic={dynamic})")
    
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
    
    def _combine_predictions(self, predictions: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Комбинира predictions с тежести
        
        Args:
            predictions: Dict с predictions за всеки модел
            weights: Dict с тежести за всеки модел
            
        Returns:
            Combined predictions
        """
        if weights is None:
            weights = self.weights
        
        # Ensure weights are in same order as predictions
        if isinstance(weights, dict):
            weight_values = [weights.get(key, 0.0) for key in predictions.keys()]
        else:
            # weights is already an array/list
            weight_values = weights
        
        # Stack predictions
        pred_list = list(predictions.values())
        stacked = np.stack(pred_list, axis=-1)
        
        # Debug shapes
        self.logger.debug(f"Stacked shape: {stacked.shape}, Weight values: {weight_values}")
        
        # Weighted average
        try:
            combined = np.average(stacked, axis=-1, weights=weight_values)
        except ValueError as e:
            self.logger.error(f"Shape mismatch: stacked={stacked.shape}, weights={np.array(weight_values).shape}")
            # Fallback to equal weights
            combined = np.mean(stacked, axis=-1)
        
        return combined
    
    def predict_ou25(
        self,
        poisson_pred: np.ndarray,
        ml_pred: np.ndarray,
        league: str = None,
        elo_pred: Optional[np.ndarray] = None,
        league_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Enhanced OU2.5 prediction with overconfidence fixes
        
        Args:
            poisson_pred: Poisson predictions
            ml_pred: ML model predictions
            league: League name for base rate regularization
            elo_pred: Elo-based predictions (optional)
            league_id: League ID за per-league weights
        
        Returns:
            Enhanced OU2.5 predictions
        """
        # STEP 1: Apply probability squashing to ML pred (BEFORE ensemble)
        original_ml_pred = ml_pred.copy()
        ml_pred_squashed = np.array([squash_prob_ou25(p, factor=0.75) for p in ml_pred.flatten()])
        
        predictions = {
            'poisson': poisson_pred,
            'ml': ml_pred_squashed.reshape(-1, 1)
        }
        
        if elo_pred is not None:
            predictions['elo'] = elo_pred
        
        # Dynamic weight adjustments (backward compatibility)
        if hasattr(self, 'dynamic') and self.dynamic:
            weights = self._get_dynamic_weights(poisson_pred, ml_pred_squashed.reshape(-1, 1), league_id)
            ensemble_pred = self._combine_predictions(predictions, weights)
        else:
            ensemble_pred = self._combine_predictions(predictions)
        
        # STEP 2: Apply strong disagreement penalty (AFTER ensemble)
        ensemble_pred_penalized = np.array([
            apply_disagreement_penalty_ou25(ep, oml, pp) 
            for ep, oml, pp in zip(ensemble_pred.flatten(), original_ml_pred.flatten(), poisson_pred.flatten())
        ])
        
        # STEP 3: Apply base rate regularization (AFTER ensemble)
        league_rate = get_league_ou25_rate(league) if league else LEAGUE_OU25_RATES['default']
        ensemble_pred_regularized = np.array([
            apply_base_rate_regularization_ou25(ep, league_rate, weight=0.15)
            for ep in ensemble_pred_penalized
        ])
        
        # STEP 4: Apply soft confidence caps (AFTER base rate regularization)
        ensemble_pred_capped = np.array([
            apply_soft_caps_ou25(ep) for ep in ensemble_pred_regularized
        ])
        
        # STEP 5: Final validation
        final_pred = np.clip(ensemble_pred_capped, 0.01, 0.99)
        
        # Assertions
        for i, fp in enumerate(final_pred):
            assert 0.01 <= fp <= 0.99, f"OU2.5 probability out of bounds at index {i}: {fp}"
            assert not np.isnan(fp), f"OU2.5 probability is NaN at index {i}: {fp}"
        
        return final_pred.reshape(-1, 1)

    def predict(
        self,
        poisson_pred: np.ndarray,
        ml_pred: np.ndarray,
        elo_pred: Optional[np.ndarray] = None,
        league_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Ensemble prediction с динамични корекции
        
        Args:
            poisson_pred: Poisson predictions
            ml_pred: ML model predictions
            elo_pred: Elo-based predictions (optional)
            league_id: League ID за per-league weights
        
        Returns:
            Ensemble predictions
        """
        predictions = {
            'poisson': poisson_pred,
            'ml': ml_pred
        }
        
        if elo_pred is not None:
            predictions['elo'] = elo_pred
        
        # Dynamic weight adjustments (backward compatibility)
        if hasattr(self, 'dynamic') and self.dynamic:
            weights = self._get_dynamic_weights(poisson_pred, ml_pred, league_id)
            return self._combine_predictions(predictions, weights)
        else:
            return self._combine_predictions(predictions)
    
    def _get_dynamic_weights(
        self,
        poisson_pred: np.ndarray,
        ml_pred: np.ndarray,
        league_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Изчислява динамични тежести базирани на ентропия и разминаване
        
        Args:
            poisson_pred: Poisson predictions
            ml_pred: ML predictions
            league_id: League ID
        
        Returns:
            Динамични тежести
        """
        # Base weights (per-league или default)
        if league_id and league_id in self.per_league_weights:
            base_weights = self.per_league_weights[league_id].copy()
        else:
            base_weights = self.weights.copy()
        
        # Изчисли ентропия на ML prediction
        ml_flat = ml_pred.flatten() if ml_pred.ndim > 0 else np.array([ml_pred])
        ml_flat = np.clip(ml_flat, 1e-15, 1 - 1e-15)
        
        if len(ml_flat) == 3:  # 1X2
            entropy = -np.sum(ml_flat * np.log(ml_flat)) / np.log(3)
        else:  # Binary
            p = ml_flat[0] if len(ml_flat) == 1 else ml_flat
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p)) / np.log(2)
        
        # Изчисли разминаване ML vs Poisson
        poisson_flat = poisson_pred.flatten() if poisson_pred.ndim > 0 else np.array([poisson_pred])
        disagreement = np.mean(np.abs(ml_flat - poisson_flat))
        
        # Dynamic adjustments
        # Висока ентропия → увеличи Poisson weight
        entropy_scalar = np.mean(entropy) if hasattr(entropy, '__len__') else entropy
        if entropy_scalar > 0.8:
            base_weights['poisson'] = min(0.6, base_weights['poisson'] + 0.1)
            base_weights['ml'] = max(0.2, base_weights['ml'] - 0.1)
        
        # Голямо разминаване → shrink към 0.5
        if disagreement > 0.25:
            shrink_factor = 0.15
            for key in base_weights:
                base_weights[key] = base_weights[key] * (1 - shrink_factor) + 0.5 * shrink_factor
        
        # Normalize weights - return only poisson and ml weights
        total = base_weights['poisson'] + base_weights['ml']
        weights_array = np.array([base_weights['poisson'], base_weights['ml']])
        return weights_array / weights_array.sum()


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
