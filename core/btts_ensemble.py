#!/usr/bin/env python3
"""
Enhanced BTTS Ensemble Logic
Подобрена логика за комбиниране на Poisson и ML BTTS predictions
"""

import numpy as np
from typing import Dict, Tuple
from core.utils import setup_logging


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
                              ml_weight: float = 0.8) -> Dict:
        """
        Подобрена ensemble логика за BTTS
        
        Args:
            ml_prob: ML модел BTTS вероятност
            poisson_prob: Poisson модел BTTS вероятност  
            ml_weight: Тежест на ML модела (default 0.8)
            
        Returns:
            Dictionary с ensemble резултати
        """
        # Базова ensemble вероятност
        base_ensemble_prob = ml_weight * ml_prob + (1 - ml_weight) * poisson_prob
        
        # Model agreement
        agreement = self.calculate_model_agreement(ml_prob, poisson_prob)
        
        # Entropy confidence за ensemble вероятността
        entropy_confidence = self.calculate_entropy_confidence(base_ensemble_prob)
        
        # Комбинирана confidence: entropy + agreement
        combined_confidence = 0.7 * entropy_confidence + 0.3 * agreement
        
        # Adjustment базиран на agreement
        if agreement < 0.7:  # Силно разминаване
            # Дърпа към по-неутрална позиция (0.5)
            adjustment_factor = 0.3 * (0.7 - agreement)  # Max 0.21
            if base_ensemble_prob > 0.5:
                adjusted_prob = base_ensemble_prob - adjustment_factor
            else:
                adjusted_prob = base_ensemble_prob + adjustment_factor
            
            # Намалява confidence при разминаване
            confidence_penalty = 0.2 * (0.7 - agreement)
            final_confidence = max(0.1, combined_confidence - confidence_penalty)
            
        elif agreement > 0.85:  # Силно съгласие
            # Леко засилва крайните вероятности
            if base_ensemble_prob > 0.6:
                adjusted_prob = min(0.95, base_ensemble_prob + 0.05)
            elif base_ensemble_prob < 0.4:
                adjusted_prob = max(0.05, base_ensemble_prob - 0.05)
            else:
                adjusted_prob = base_ensemble_prob
            
            # Увеличава confidence при съгласие
            confidence_bonus = 0.1 * (agreement - 0.85)
            final_confidence = min(1.0, combined_confidence + confidence_bonus)
            
        else:  # Умерено съгласие
            adjusted_prob = base_ensemble_prob
            final_confidence = combined_confidence
        
        # Финална вероятност
        final_prob = np.clip(adjusted_prob, 0.01, 0.99)
        
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
            'predicted_outcome': 'Yes' if final_prob > 0.5 else 'No',
            'components': {
                'ml_prob': float(ml_prob),
                'poisson_prob': float(poisson_prob),
                'base_ensemble': float(base_ensemble_prob),
                'model_agreement': float(agreement),
                'entropy_confidence': float(entropy_confidence),
                'adjustment_applied': float(abs(final_prob - base_ensemble_prob))
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
        
        # Threshold препоръки
        threshold_rec = ensemble.get_threshold_recommendation(
            result['probability'], result['confidence']
        )
        logger.info(f"   Recommended Threshold: {threshold_rec['recommended_threshold']}")
    
    logger.info("\n✅ BTTS ensemble тестване завършено!")


if __name__ == "__main__":
    test_btts_ensemble()
