"""
🎯 АНАЛИЗ НА ВЕРОЯТНОСТНАТА ЛОГИКА - Football AI Project
Всички ключови части от кода за вероятности, калибрация и ensemble
"""

# =============================================================================
# 1️⃣ POISSON MODEL - Основни вероятностни изчисления
# Файл: core/poisson_utils.py
# =============================================================================

def calculate_lambda(
    self,
    home_team_id: int,
    away_team_id: int,
    league_id: Optional[int] = None
) -> Tuple[float, float]:
    """
    🔍 КЛЮЧОВА ФУНКЦИЯ: Изчисляване на λ (очаквани голове)
    
    ТЕКУЩА ФОРМУЛА:
    λ_home = league_avg_home * home_attack * away_defense * home_advantage
    λ_away = league_avg_away * away_attack * home_defense
    
    ПРОБЛЕМИ:
    - home_advantage = 1.15 (фиксиран) - може да е твърде висок
    - Няма league-specific home advantage
    - Няма momentum/form adjustment
    """
    # League average
    if league_id and league_id in self.league_avg_goals_home:
        avg_home = self.league_avg_goals_home[league_id]
        avg_away = self.league_avg_goals_away[league_id]
    else:
        avg_home = self.league_avg_goals_home.get(0, 1.5)  # 🚨 Default 1.5
        avg_away = self.league_avg_goals_away.get(0, 1.2)  # 🚨 Default 1.2
    
    # Team strengths
    home_attack = self.attack_strength.get(home_team_id, 1.0)   # 🚨 Default 1.0
    home_defense = self.defense_strength.get(home_team_id, 1.0)
    away_attack = self.attack_strength.get(away_team_id, 1.0)
    away_defense = self.defense_strength.get(away_team_id, 1.0)
    
    # 🎯 КРИТИЧНИ ФОРМУЛИ:
    lambda_home = avg_home * home_attack * away_defense * self.home_advantage  # 1.15
    lambda_away = avg_away * away_attack * home_defense
    
    return lambda_home, lambda_away


def predict_match_probabilities(
    self,
    home_team_id: int,
    away_team_id: int,
    league_id: Optional[int] = None,
    max_goals: int = 10
) -> Dict[str, float]:
    """
    🔍 КЛЮЧОВА ФУНКЦИЯ: Poisson вероятности за всички пазари
    
    ПРОБЛЕМИ:
    - max_goals=10 може да е недостатъчно за високи λ
    - Няма корекция за low-scoring leagues
    - Няма weather/venue adjustments
    """
    # Изчисляване на lambda
    lambda_home, lambda_away = self.calculate_lambda(
        home_team_id, away_team_id, league_id
    )
    
    # 🎯 МАТРИЦА С ВЕРОЯТНОСТИ:
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    
    # 🎯 1X2 ВЕРОЯТНОСТИ:
    prob_home_win = np.sum(np.tril(prob_matrix, -1))  # Под диагонала
    prob_draw = np.sum(np.diag(prob_matrix))          # Диагонал
    prob_away_win = np.sum(np.triu(prob_matrix, 1))   # Над диагонала
    
    # 🎯 OVER/UNDER 2.5:
    prob_over_25 = 0
    prob_under_25 = 0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j > 2.5:  # 🚨 Твърда граница 2.5
                prob_over_25 += prob_matrix[i, j]
            else:
                prob_under_25 += prob_matrix[i, j]
    
    # 🎯 BTTS (Both Teams To Score):
    prob_btts_yes = 0
    prob_btts_no = 0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i > 0 and j > 0:  # 🚨 И двата > 0
                prob_btts_yes += prob_matrix[i, j]
            else:
                prob_btts_no += prob_matrix[i, j]
    
    return {
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'prob_home_win': prob_home_win,
        'prob_draw': prob_draw,
        'prob_away_win': prob_away_win,
        'prob_over_25': prob_over_25,
        'prob_under_25': prob_under_25,
        'prob_btts_yes': prob_btts_yes,
        'prob_btts_no': prob_btts_no,
        'expected_home_goals': lambda_home,
        'expected_away_goals': lambda_away,
        'expected_total_goals': lambda_home + lambda_away
    }


# =============================================================================
# 2️⃣ ML MODELS - predict_proba() извиквания и калибрация
# Файл: api/prediction_service.py
# =============================================================================

def ml_predictions_and_calibration():
    """
    🔍 ML МОДЕЛИ: Как се правят predictions и calibration
    
    ПРОБЛЕМИ:
    - Само BTTS има calibration
    - 1X2 и OU2.5 използват сурови вероятности
    - Няма confidence adjustment
    """
    
    # 🎯 ML PREDICTIONS (редове 312-314):
    ml_1x2 = self.models['1x2'].predict_proba(X_1x2)[0]        # [prob_1, prob_X, prob_2]
    ml_ou25 = self.models['ou25'].predict_proba(X_ou25)[0, 1]  # prob_over (само класа 1)
    ml_btts_raw = self.models['btts'].predict_proba(X_btts)[0, 1]  # prob_yes (само класа 1)
    
    # 🎯 BTTS CALIBRATION (редове 316-318):
    # 🚨 ЗАЩО САМО BTTS ИМА CALIBRATION?
    ml_btts_calibrated = 0.5 + (ml_btts_raw - 0.5) * 0.85  # Намалява overconfidence с 15%
    ml_btts_calibrated = np.clip(ml_btts_calibrated, 0.05, 0.95)  # Clipping
    
    # 🎯 BLENDING С POISSON (ред 321):
    ml_btts = 0.8 * ml_btts_calibrated + 0.2 * poisson_pred['prob_btts']
    # 🚨 ЗАЩО 80%-20% SPLIT? ОПТИМИЗИРАНО ЛИ Е?
    
    return ml_1x2, ml_ou25, ml_btts


# =============================================================================
# 3️⃣ ENSEMBLE MODEL - Комбиниране на вероятности
# Файл: core/ensemble.py
# =============================================================================

class EnsembleModel:
    """
    🔍 ENSEMBLE: Комбиниране на Poisson, ML и Elo predictions
    
    ПРОБЛЕМИ:
    - Фиксирани weights без оптимизация
    - Няма dynamic weighting според confidence
    - Няма league-specific weights
    """
    
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        # 🎯 DEFAULT WEIGHTS (редове 38-42):
        self.weights = initial_weights or {
            'poisson': 0.3,  # 🚨 30% - Защо толкова ниско?
            'ml': 0.5,       # 🚨 50% - Най-висок weight
            'elo': 0.2       # 🚨 20% - Най-нисък weight
        }
        # 🚨 ТЕЗИ WEIGHTS СА ФИКСИРАНИ! НЯМА ОПТИМИЗАЦИЯ В PRODUCTION!
    
    def optimize_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        🔍 WEIGHT OPTIMIZATION: Минимизиране на log loss
        
        ПРОБЛЕМИ:
        - Използва се само в training, не в production
        - Няма regularization
        - Няма cross-validation
        """
        # Objective function (редове 75-88):
        def objective(weights):
            combined = self._combine_predictions(predictions, weights)
            combined = np.clip(combined, 1e-15, 1 - 1e-15)  # 🚨 Hard clipping
            
            # Normalization (редове 82-83):
            if combined.ndim == 2:
                combined = combined / combined.sum(axis=1, keepdims=True)
            
            return log_loss(y_true, combined)
        
        # 🚨 CONSTRAINTS: weights sum to 1, bounds [0,1]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(predictions))]
    
    def _combine_predictions(self, predictions: Dict[str, np.ndarray], weights: Optional[np.ndarray] = None):
        """
        🔍 PREDICTION COMBINING: Weighted average
        
        ПРОБЛЕМИ:
        - Само linear combination
        - Няма non-linear blending
        - Няма confidence weighting
        """
        if weights is None:
            weights = np.array([self.weights.get(k, 0.33) for k in predictions.keys()])
        
        # 🎯 WEIGHTED AVERAGE (редове 134-135):
        stacked = np.stack(list(predictions.values()), axis=-1)
        combined = np.average(stacked, axis=-1, weights=weights)
        # 🚨 САМО LINEAR COMBINATION!
        
        return combined


# =============================================================================
# 4️⃣ ENSEMBLE INFERENCE - Финални predictions в API
# Файл: api/prediction_service.py
# =============================================================================

def ensemble_inference():
    """
    🔍 ENSEMBLE INFERENCE: Как се правят финалните predictions
    
    ПРОБЛЕМИ:
    - Различни ensemble методи за различни пазари
    - Няма consistency между пазарите
    - Confidence се изчислява като max probability
    """
    
    # 🎯 1X2 ENSEMBLE (редове 324-327):
    ensemble_1x2 = self.models['ensemble'].predict(
        poisson_pred['probs_1x2'].reshape(1, -1),  # [prob_1, prob_X, prob_2]
        ml_1x2.reshape(1, -1)                      # [prob_1, prob_X, prob_2]
    )[0]
    
    # 🎯 OU2.5 ENSEMBLE (редове 329-332):
    ensemble_ou25 = self.models['ensemble'].predict(
        np.array([[poisson_pred['prob_over25']]]),  # Само prob_over
        np.array([[ml_ou25]])                       # Само prob_over
    )[0, 0]
    
    # 🎯 BTTS ENSEMBLE (редове 334-337):
    ensemble_btts = self.models['ensemble'].predict(
        np.array([[poisson_pred['prob_btts']]]),    # Само prob_yes
        np.array([[ml_btts]])                       # Само prob_yes (вече calibrated)
    )[0, 0]
    
    # 🎯 ФИНАЛНИ РЕЗУЛТАТИ (редове 356-376):
    result = {
        'prediction_1x2': {
            'prob_home_win': float(ensemble_1x2[0]),
            'prob_draw': float(ensemble_1x2[1]),
            'prob_away_win': float(ensemble_1x2[2]),
            'predicted_outcome': ['1', 'X', '2'][np.argmax(ensemble_1x2)],
            'confidence': float(np.max(ensemble_1x2))  # 🚨 MAX PROBABILITY = CONFIDENCE?
        },
        'prediction_ou25': {
            'prob_over': float(ensemble_ou25),
            'prob_under': float(1 - ensemble_ou25),
            'predicted_outcome': 'Over' if ensemble_ou25 > 0.5 else 'Under',  # 🚨 Hard 0.5 threshold
            'confidence': float(max(ensemble_ou25, 1 - ensemble_ou25))
        },
        'prediction_btts': {
            'prob_yes': float(ensemble_btts),
            'prob_no': float(1 - ensemble_btts),
            'predicted_outcome': self._get_btts_outcome(ensemble_btts, elo_diff),  # 🚨 Custom logic
            'confidence': float(max(ensemble_btts, 1 - ensemble_btts))
        }
    }


# =============================================================================
# 5️⃣ CALIBRATION & NORMALIZATION - Вероятностни корекции
# Файлове: pipelines/train_ml_models.py, core/poisson_utils.py
# =============================================================================

def calibration_logic():
    """
    🔍 CALIBRATION: Как се калибрират вероятностите
    
    ПРОБЛЕМИ:
    - Само binary models имат Isotonic Regression
    - 1X2 няма calibration
    - Различни clipping стратегии
    """
    
    # 🎯 ISOTONIC REGRESSION CALIBRATION (train_ml_models.py:334-351):
    from sklearn.calibration import CalibratedClassifierCV
    
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',  # 🚨 Само isotonic, няма Platt scaling
        cv=3
    )
    calibrated_model.fit(X_val, y_val)
    
    # Calibrated predictions:
    y_train_proba = calibrated_model.predict_proba(X_train)[:, 1]
    y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    
    # 🎯 POISSON PROBABILITY NORMALIZATION (poisson_utils.py:427-432):
    # 1X2 normalization:
    y_pred_1x2 = np.nan_to_num(y_pred_1x2, nan=0.33)  # 🚨 Default uniform
    y_pred_1x2 = np.clip(y_pred_1x2, 1e-15, 1 - 1e-15)  # 🚨 Hard clipping
    
    # Ensure sum to 1:
    row_sums = y_pred_1x2.sum(axis=1, keepdims=True)
    y_pred_1x2 = y_pred_1x2 / row_sums
    
    # Binary clipping:
    y_pred_over25 = np.clip(y_pred_over25, 1e-15, 1 - 1e-15)
    y_pred_btts = np.clip(y_pred_btts, 1e-15, 1 - 1e-15)


# =============================================================================
# 6️⃣ CONFIDENCE SCORING - Как се изчислява confidence
# =============================================================================

def confidence_calculation():
    """
    🔍 CONFIDENCE: Как се изчислява увереността в прогнозата
    
    ПРОБЛЕМИ:
    - Confidence = max(probabilities) - твърде опростено
    - Няма entropy-based confidence
    - Няма model agreement scoring
    """
    
    # 🎯 ТЕКУЩ МЕТОД:
    confidence_1x2 = float(np.max(ensemble_1x2))  # 🚨 MAX от [prob_1, prob_X, prob_2]
    confidence_ou25 = float(max(ensemble_ou25, 1 - ensemble_ou25))  # 🚨 MAX от [prob_over, prob_under]
    confidence_btts = float(max(ensemble_btts, 1 - ensemble_btts))  # 🚨 MAX от [prob_yes, prob_no]
    
    # 🚨 ПРОБЛЕМИ:
    # - High probability != High confidence
    # - Не отчита model disagreement
    # - Не отчита data quality
    # - Няма calibration на confidence scores


# =============================================================================
# 7️⃣ FII (Football Intelligence Index) - Интерпретируем индекс
# Файл: core/ensemble.py, api/prediction_service.py
# =============================================================================

def fii_calculation():
    """
    🔍 FII: Football Intelligence Index за качество на прогнозата
    
    ПРОБЛЕМИ:
    - Опростена формула
    - Няма machine learning за FII
    - Фиксирани weights
    """
    
    # 🎯 FII COMPONENTS (prediction_service.py:340-346):
    fii_score, fii_conf = self.models['fii'].calculate_fii(
        elo_diff=match_df['elo_diff'].iloc[0],
        form_diff=match_df['home_form_5'].iloc[0] - match_df['away_form_5'].iloc[0],
        xg_efficiency_diff=match_df['home_xg_proxy'].iloc[0] - match_df['away_xg_proxy'].iloc[0],
        finishing_efficiency_diff=match_df['home_shooting_efficiency'].iloc[0] - match_df['away_shooting_efficiency'].iloc[0],
        is_home=1
    )
    
    # 🚨 FII WEIGHTS (вероятно фиксирани):
    # elo_weight = ?
    # form_weight = ?
    # xg_weight = ?
    # finishing_weight = ?
    # home_weight = ?


# =============================================================================
# 8️⃣ FALLBACK VALUES - Default стойности при грешки
# =============================================================================

def fallback_values():
    """
    🔍 FALLBACK: Какви default стойности се използват
    
    ПРОБЛЕМИ:
    - Твърде опростени fallbacks
    - Няма league-specific defaults
    - Няма confidence penalty за fallbacks
    """
    
    # 🎯 POISSON FALLBACK (prediction_service.py:284-291):
    poisson_pred = {
        'probs_1x2': np.array([0.33, 0.33, 0.34]),  # 🚨 Uniform distribution
        'prob_over25': 0.5,                          # 🚨 50-50
        'prob_btts': 0.5,                            # 🚨 50-50
        'lambda_home': 1.5,                          # 🚨 League average?
        'lambda_away': 1.2,                          # 🚨 League average?
        'expected_goals': 2.7                        # 🚨 1.5 + 1.2
    }
    
    # 🎯 TEAM STRENGTH FALLBACK (poisson_utils.py:134, 174):
    self.attack_strength[team_id] = 1.0   # 🚨 Average team
    self.defense_strength[team_id] = 1.0  # 🚨 Average team


# =============================================================================
# 🎯 КРИТИЧНИ ПРОБЛЕМИ ЗА ОПТИМИЗАЦИЯ:
# =============================================================================

"""
1️⃣ POISSON BIAS:
   - home_advantage = 1.15 може да е твърде висок
   - Няма league-specific adjustments
   - max_goals = 10 може да е недостатъчно

2️⃣ ML CALIBRATION:
   - Само BTTS има calibration (защо?)
   - 1X2 и OU2.5 използват сурови вероятности
   - Calibration коефициент 0.85 не е оптимизиран

3️⃣ ENSEMBLE WEIGHTS:
   - Фиксирани weights: poisson=30%, ml=50%, elo=20%
   - Няма dynamic weighting
   - Няма league/confidence-based adjustments

4️⃣ CONFIDENCE SCORING:
   - confidence = max(probabilities) е твърде опростено
   - Няма entropy-based confidence
   - Няма model agreement scoring

5️⃣ PROBABILITY CLIPPING:
   - Hard clipping 1e-15 до 1-1e-15
   - Може да влошава калибрацията
   - Няма soft boundaries

6️⃣ NORMALIZATION:
   - Различни методи за различни пазари
   - Няма consistency
   - Fallback стойности са твърде опростени

7️⃣ FII CALCULATION:
   - Опростена формула
   - Фиксирани weights
   - Няма ML-based FII
"""

# =============================================================================
# 🚀 СЛЕДВАЩИ СТЪПКИ ЗА ОПТИМИЗАЦИЯ:
# =============================================================================

"""
1. Калибрация на Poisson λ формулите
2. ML model calibration за всички пазари
3. Dynamic ensemble weighting
4. Entropy-based confidence scoring
5. Soft probability boundaries
6. League-specific adjustments
7. Improved FII calculation
8. Bias reduction (Over 2.5, Home Win)
"""
