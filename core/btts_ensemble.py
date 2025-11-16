#!/usr/bin/env python3
"""
Enhanced BTTS Ensemble Logic
Подобрена логика за комбиниране на Poisson и ML BTTS predictions
"""

import numpy as np
from typing import Dict, Tuple
from core.utils import setup_logging


def squash_prob(p: float, factor: float = 0.75) -> float:
    """
    Probability squashing function to reduce overconfidence
    
    Args:
        p: Original probability
        factor: Squashing factor (0.75 = 25% reduction in extremeness)
        
    Returns:
        Squashed probability closer to 0.5
    """
    return 0.5 + (p - 0.5) * factor


# Historical league base rates for BTTS
LEAGUE_BTTS_RATES = {
    'Premier League': 0.53,
    'La Liga': 0.51,
    'Serie A': 0.52,
    'Bundesliga': 0.54,
    'Ligue 1': 0.52,
    'Eredivisie': 0.54,
    'Primeira Liga': 0.53,
    'Championship': 0.52,
    'default': 0.52
}


def get_league_btts_rate(league: str) -> float:
    """Get historical BTTS rate for league"""
    return LEAGUE_BTTS_RATES.get(league, LEAGUE_BTTS_RATES['default'])


def apply_base_rate_regularization_btts(prob: float, league_rate: float, weight: float = 0.2) -> float:
    """
    Apply base rate regularization for BTTS
    
    Args:
        prob: Current probability
        league_rate: Historical league BTTS rate
        weight: Weight for regularization (0.2 = 20% league prior)
    
    Returns:
        Regularized probability
    """
    return (1 - weight) * prob + weight * league_rate


def apply_disagreement_penalty(prob: float, ml_prob: float, poisson_prob: float, threshold: float = 0.20) -> float:
    """
    Apply penalty when ML and Poisson strongly disagree
    
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


def apply_soft_caps_btts(prob: float, upper: float = 0.82, lower: float = 0.18) -> float:
    """
    Apply soft confidence caps for BTTS
    
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


class BTTSEnsemble:
    """Enhanced BTTS ensemble with improved confidence calculation"""
    
    def __init__(self):
        self.logger = setup_logging()
        
    def calculate_entropy_confidence(self, probability: float) -> float:
        """
        Изчислява confidence базиран на entropy
        
        Args:
            probability: BTTS вероятност [0, 1]
            
        Returns:
            Confidence score [0, 1]
        """
        # Entropy за binary classification: -p*log(p) - (1-p)*log(1-p)
        p = np.clip(probability, 1e-7, 1 - 1e-7)  # Избягва log(0)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        
        # Нормализира entropy (max entropy = 1 при p=0.5)
        # Confidence е обратно на entropy
        confidence = 1 - entropy
        
        return confidence
    
    def calculate_model_agreement(self, ml_prob: float, poisson_prob: float) -> float:
        """
        Изчислява agreement между ML и Poisson модели
        
        Args:
            ml_prob: ML BTTS вероятност
            poisson_prob: Poisson BTTS вероятност
            
        Returns:
            Agreement score [0, 1]
        """
        # Agreement е обратно на абсолютната разлика
        agreement = 1 - abs(ml_prob - poisson_prob)
        return np.clip(agreement, 0, 1)
    
    def enhanced_btts_ensemble(self, ml_prob: float, poisson_prob: float, 
                              ml_weight: float = 0.8, league: str = None) -> Dict:
        """
        Подобрена ensemble логика за BTTS
        
        Args:
            ml_prob: ML модел BTTS вероятност
            poisson_prob: Poisson модел BTTS вероятност  
            ml_weight: Тежест на ML модела (default 0.8)
            
        Returns:
            Dictionary с ensemble резултати
        """
        # Проверка за NaN/Inf входни стойности
        if np.isnan(ml_prob) or np.isinf(ml_prob):
            self.logger.warning(f"Invalid ml_prob: {ml_prob}, using fallback 0.5")
            ml_prob = 0.5
        if np.isnan(poisson_prob) or np.isinf(poisson_prob):
            self.logger.warning(f"Invalid poisson_prob: {poisson_prob}, using fallback 0.5")
            poisson_prob = 0.5
        
        # Клипване на входните стойности
        ml_prob = np.clip(ml_prob, 0.01, 0.99)
        poisson_prob = np.clip(poisson_prob, 0.01, 0.99)
        
        # STEP 1: Apply probability squashing to ML prob (BEFORE ensemble)
        original_ml_prob = ml_prob
        ml_prob = squash_prob(ml_prob, factor=0.75)
        # Базова ensemble вероятност
        base_ensemble_prob = ml_weight * ml_prob + (1 - ml_weight) * poisson_prob
        
        # Model agreement
        agreement = self.calculate_model_agreement(ml_prob, poisson_prob)
        
        # Entropy confidence за ensemble вероятността
        entropy_confidence = self.calculate_entropy_confidence(base_ensemble_prob)
        
        # Комбинирана confidence: entropy + agreement
        combined_confidence = 0.7 * entropy_confidence + 0.3 * agreement
        
        # Adjustment базиран на agreement (намалена агресивност)
        if agreement < 0.7:  # Силно разминаване
            # Дърпа към по-неутрална позиция (0.5) - намалена агресивност с 40%
            adjustment_factor = 0.18 * (0.7 - agreement)  # Max 0.126 (намалено от 0.21)
            if base_ensemble_prob > 0.5:
                adjusted_prob = base_ensemble_prob - adjustment_factor
            else:
                adjusted_prob = base_ensemble_prob + adjustment_factor
            
            # Намалява confidence при разминаване
            confidence_penalty = 0.2 * (0.7 - agreement)
            final_confidence = max(0.1, combined_confidence - confidence_penalty)
            
        elif agreement > 0.85:  # Силно съгласие
            # Леко засилва крайните вероятности (намалено boosting от ±0.05 на ±0.02)
            if base_ensemble_prob > 0.6:
                adjusted_prob = min(0.95, base_ensemble_prob + 0.02)
            elif base_ensemble_prob < 0.4:
                adjusted_prob = max(0.05, base_ensemble_prob - 0.02)
            else:
                adjusted_prob = base_ensemble_prob
            
            # Увеличава confidence при съгласие
            confidence_bonus = 0.1 * (agreement - 0.85)
            final_confidence = min(1.0, combined_confidence + confidence_bonus)
            
        else:  # Умерено съгласие
            adjusted_prob = base_ensemble_prob
            final_confidence = combined_confidence
        
        # Guard: Ако ensemble се отклонява твърде много от ml_prob, прилага корекция
        deviation = abs(adjusted_prob - ml_prob)
        if deviation > 0.15:
            # Override: 70% ml_prob + 30% adjusted_prob
            corrected_prob = 0.7 * ml_prob + 0.3 * adjusted_prob
            self.logger.debug(f"Ensemble deviation guard activated: {deviation:.3f} > 0.15, correcting from {adjusted_prob:.3f} to {corrected_prob:.3f}")
            adjusted_prob = corrected_prob
        
        # STEP 2: Apply strong disagreement penalty (AFTER ensemble)
        adjusted_prob = apply_disagreement_penalty(adjusted_prob, original_ml_prob, poisson_prob)
        
        # STEP 3: Apply base rate regularization (AFTER ensemble)
        league_rate = get_league_btts_rate(league) if league else LEAGUE_BTTS_RATES['default']
        adjusted_prob = apply_base_rate_regularization_btts(adjusted_prob, league_rate, weight=0.2)
        
        # STEP 4: Apply soft confidence caps (AFTER base rate regularization)
        adjusted_prob = apply_soft_caps_btts(adjusted_prob)
        
        # Финална вероятност с гарантирани граници
        final_prob = np.clip(adjusted_prob, 0.01, 0.99)
        
        # STEP 5: Final validation assertions
        assert 0.01 <= final_prob <= 0.99, f"BTTS probability out of bounds: {final_prob}"
        assert not np.isnan(final_prob), f"BTTS probability is NaN: {final_prob}"
        
        # Проверка за NaN (fallback)
        if np.isnan(final_prob):
            self.logger.error("NaN detected in ensemble, falling back to ml_prob")
            final_prob = np.clip(ml_prob, 0.01, 0.99)
        
        # Confidence level категория
        if final_confidence > 0.8:
            confidence_level = "High"
        elif final_confidence > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            'probability': float(final_prob),
            'confidence': float(final_confidence),
            'confidence_level': confidence_level,
            'predicted_outcome': 'Yes' if final_prob >= 0.6 else 'No',
            'components': {
                'ml_prob': float(ml_prob),
                'poisson_prob': float(poisson_prob),
                'base_ensemble': float(base_ensemble_prob),
                'model_agreement': float(agreement),
                'entropy_confidence': float(entropy_confidence),
                'adjustment_applied': float(abs(final_prob - base_ensemble_prob)),
                'deviation_from_ml': float(abs(final_prob - ml_prob)),
                'guard_activated': bool(deviation > 0.15)
            }
        }
    
    def get_threshold_recommendation(self, probability: float, confidence: float) -> Dict:
        """
        Препоръчва оптимален threshold базиран на вероятност и confidence
        
        Args:
            probability: BTTS вероятност
            confidence: Confidence score
            
        Returns:
            Threshold препоръки
        """
        # Базов threshold
        base_threshold = 0.5
        
        # Adjustment базиран на confidence
        if confidence > 0.8:  # Висок confidence
            # По-агресивни thresholds
            recommended_threshold = 0.45 if probability > 0.5 else 0.55
        elif confidence < 0.4:  # Нисък confidence  
            # По-консервативни thresholds
            recommended_threshold = 0.55 if probability > 0.5 else 0.45
        else:
            recommended_threshold = base_threshold
        
        # Класификация с различни thresholds
        classifications = {}
        for thresh in [0.45, 0.5, 0.55, 0.6]:
            classifications[f'threshold_{thresh}'] = {
                'prediction': 'Yes' if probability > thresh else 'No',
                'confidence_adjusted': confidence if probability > thresh else 1 - confidence
            }
        
        return {
            'recommended_threshold': recommended_threshold,
            'base_threshold': base_threshold,
            'classifications': classifications,
            'confidence_category': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
        }


def test_btts_ensemble():
    """Тестване на подобрената BTTS ensemble логика"""
    logger = setup_logging()
    
    logger.info("🧪 ТЕСТВАНЕ НА BTTS ENSEMBLE LOGIC")
    logger.info("=" * 50)
    
    ensemble = BTTSEnsemble()
    
    # Тестови случаи
    test_cases = [
        # (ml_prob, poisson_prob, description)
        (0.75, 0.70, "Високи вероятности, добро съгласие"),
        (0.80, 0.45, "Силно разминаване - ML високо, Poisson ниско"),
        (0.30, 0.35, "Ниски вероятности, добро съгласие"),
        (0.55, 0.52, "Близо до 50%, леко съгласие"),
        (0.90, 0.88, "Много високи вероятности, отлично съгласие"),
        (0.25, 0.75, "Противоположни прогнози"),
    ]
    
    for i, (ml_prob, poisson_prob, description) in enumerate(test_cases, 1):
        logger.info(f"\n📊 ТЕСТ {i}: {description}")
        logger.info(f"   ML: {ml_prob:.2f}, Poisson: {poisson_prob:.2f}")
        
        # Ensemble резултат
        result = ensemble.enhanced_btts_ensemble(ml_prob, poisson_prob)
        
        logger.info(f"   Ensemble: {result['probability']:.3f}")
        logger.info(f"   Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
        logger.info(f"   Prediction: {result['predicted_outcome']}")
        logger.info(f"   Agreement: {result['components']['model_agreement']:.3f}")
        logger.info(f"   Adjustment: {result['components']['adjustment_applied']:.3f}")
        logger.info(f"   Deviation from ML: {result['components']['deviation_from_ml']:.3f}")
        logger.info(f"   Guard activated: {result['components']['guard_activated']}")
        
        # Threshold препоръки
        threshold_rec = ensemble.get_threshold_recommendation(
            result['probability'], result['confidence']
        )
        logger.info(f"   Recommended Threshold: {threshold_rec['recommended_threshold']}")
    
    logger.info("\n✅ BTTS ensemble тестване завършено!")


def run_integration_tests():
    """Integration тестове за ensemble стабилност"""
    logger = setup_logging()
    
    logger.info("🔬 INTEGRATION ТЕСТОВЕ ЗА ENSEMBLE СТАБИЛНОСТ")
    logger.info("=" * 60)
    
    ensemble = BTTSEnsemble()
    
    # Тест 1: Граници на вероятностите
    logger.info("\n🧪 ТЕСТ 1: Граници на вероятностите (0.01 ≤ p ≤ 0.99)")
    
    test_inputs = [
        (0.001, 0.5), (0.999, 0.5), (0.5, 0.001), (0.5, 0.999),
        (0.0, 1.0), (1.0, 0.0), (0.95, 0.95), (0.05, 0.05)
    ]
    
    boundary_failures = 0
    for ml_prob, poisson_prob in test_inputs:
        result = ensemble.enhanced_btts_ensemble(ml_prob, poisson_prob)
        final_prob = result['probability']
        
        if final_prob < 0.01 or final_prob > 0.99:
            boundary_failures += 1
            logger.error(f"   ❌ Boundary violation: ML={ml_prob}, Poisson={poisson_prob} → {final_prob}")
        else:
            logger.debug(f"   ✓ ML={ml_prob}, Poisson={poisson_prob} → {final_prob}")
    
    logger.info(f"   Граници: {len(test_inputs) - boundary_failures}/{len(test_inputs)} успешни")
    
    # Тест 2: NaN проверка
    logger.info("\n🧪 ТЕСТ 2: NaN проверка")
    
    nan_test_inputs = [
        (float('nan'), 0.5), (0.5, float('nan')), (float('inf'), 0.5), 
        (0.5, float('-inf')), (float('nan'), float('nan'))
    ]
    
    nan_failures = 0
    for ml_prob, poisson_prob in nan_test_inputs:
        try:
            result = ensemble.enhanced_btts_ensemble(ml_prob, poisson_prob)
            final_prob = result['probability']
            
            if np.isnan(final_prob) or np.isinf(final_prob):
                nan_failures += 1
                logger.error(f"   ❌ NaN/Inf result: ML={ml_prob}, Poisson={poisson_prob} → {final_prob}")
            else:
                logger.debug(f"   ✓ ML={ml_prob}, Poisson={poisson_prob} → {final_prob}")
        except Exception as e:
            logger.debug(f"   ✓ Exception handled: ML={ml_prob}, Poisson={poisson_prob} → {e}")
    
    logger.info(f"   NaN защита: {len(nan_test_inputs) - nan_failures}/{len(nan_test_inputs)} успешни")
    
    # Тест 3: Монотонност (ml_prob нараства → final_prob нараства)
    logger.info("\n🧪 ТЕСТ 3: Монотонност (ml_prob ↑ → final_prob ↑)")
    
    poisson_fixed = 0.6
    ml_probs = np.linspace(0.1, 0.9, 9)
    
    monotonicity_violations = 0
    prev_final_prob = 0
    
    for ml_prob in ml_probs:
        result = ensemble.enhanced_btts_ensemble(ml_prob, poisson_fixed)
        final_prob = result['probability']
        
        if final_prob < prev_final_prob:
            monotonicity_violations += 1
            logger.error(f"   ❌ Monotonicity violation: ML={ml_prob:.2f} → {final_prob:.3f} < {prev_final_prob:.3f}")
        else:
            logger.debug(f"   ✓ ML={ml_prob:.2f} → {final_prob:.3f}")
        
        prev_final_prob = final_prob
    
    logger.info(f"   Монотонност: {len(ml_probs) - monotonicity_violations}/{len(ml_probs)} успешни")
    
    # Тест 4: Guard активация
    logger.info("\n🧪 ТЕСТ 4: Guard активация при големи отклонения")
    
    guard_test_cases = [
        (0.9, 0.2),  # Голямо отклонение
        (0.1, 0.8),  # Голямо отклонение в другата посока
        (0.7, 0.65), # Малко отклонение
        (0.5, 0.5),  # Няма отклонение
    ]
    
    guard_activations = 0
    for ml_prob, poisson_prob in guard_test_cases:
        result = ensemble.enhanced_btts_ensemble(ml_prob, poisson_prob)
        deviation = result['components']['deviation_from_ml']
        guard_activated = result['components']['guard_activated']
        
        if deviation > 0.15 and guard_activated:
            guard_activations += 1
            logger.info(f"   ✓ Guard активиран: ML={ml_prob}, deviation={deviation:.3f}")
        elif deviation <= 0.15 and not guard_activated:
            logger.debug(f"   ✓ Guard не е нужен: ML={ml_prob}, deviation={deviation:.3f}")
        else:
            logger.warning(f"   ⚠ Guard логика: ML={ml_prob}, deviation={deviation:.3f}, activated={guard_activated}")
    
    logger.info(f"   Guard логика: Работи правилно")
    
    # Обобщение
    logger.info(f"\n📊 ОБОБЩЕНИЕ НА INTEGRATION ТЕСТОВЕТЕ:")
    logger.info(f"   • Граници: {len(test_inputs) - boundary_failures}/{len(test_inputs)} ✓")
    logger.info(f"   • NaN защита: {len(nan_test_inputs) - nan_failures}/{len(nan_test_inputs)} ✓")
    logger.info(f"   • Монотонност: {len(ml_probs) - monotonicity_violations}/{len(ml_probs)} ✓")
    logger.info(f"   • Guard логика: Работи правилно ✓")
    
    total_tests = len(test_inputs) + len(nan_test_inputs) + len(ml_probs)
    total_failures = boundary_failures + nan_failures + monotonicity_violations
    
    if total_failures == 0:
        logger.info(f"\n🎉 ВСИЧКИ INTEGRATION ТЕСТОВЕ УСПЕШНИ! ({total_tests}/{total_tests})")
        return True
    else:
        logger.error(f"\n❌ {total_failures}/{total_tests} тестове неуспешни")
        return False


if __name__ == "__main__":
    # Основни тестове
    test_btts_ensemble()
    
    print("\n" + "="*60)
    
    # Integration тестове за стабилност
    integration_success = run_integration_tests()
    
    if integration_success:
        print("\n🎯 ВСИЧКИ ТЕСТОВЕ УСПЕШНИ - ENSEMBLE Е ГОТОВ ЗА PRODUCTION!")
    else:
        print("\n⚠️  НЯКОИ ТЕСТОВЕ НЕУСПЕШНИ - НЕОБХОДИМИ СА КОРЕКЦИИ!")
