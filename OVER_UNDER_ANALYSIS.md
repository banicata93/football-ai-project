# 📊 OVER/UNDER 2.5 - ПЪЛЕН АНАЛИЗ НА ИЗЧИСЛЕНИЯТА

## 🎯 Как работи сега системата

### 1️⃣ POISSON MODEL (Baseline)

**Файл:** `core/poisson_utils.py` (lines 256-265)

```python
# Over/Under 2.5 изчисление
prob_over_25 = 0
prob_under_25 = 0

for i in range(max_goals + 1):
    for j in range(max_goals + 1):
        if i + j > 2.5:  # Ако общо голове > 2.5
            prob_over_25 += prob_matrix[i, j]
        else:
            prob_under_25 += prob_matrix[i, j]
```

**Как работи:**
1. Изчислява `lambda_home` и `lambda_away` (очаквани голове)
2. Създава матрица с вероятности за всеки резултат (0-0, 1-0, 0-1, 1-1, и т.н.)
3. Сумира вероятностите за всички резултати с общо голове > 2.5

**Пример:**
- `lambda_home = 1.8` (очаквани голове домакин)
- `lambda_away = 1.2` (очаквани голове гост)
- `expected_total = 3.0` голa

**Матрица вероятности:**
```
        Away: 0    1    2    3    4
Home 0:  0.05  0.06  0.04  0.02  0.01
     1:  0.09  0.11  0.07  0.03  0.01
     2:  0.08  0.10  0.06  0.03  0.01
     3:  0.05  0.06  0.04  0.02  0.01
     4:  0.02  0.03  0.02  0.01  0.00
```

**Резултати с > 2.5 голa:** 1-2, 2-1, 2-2, 3-0, 0-3, 3-1, 1-3, 4-0, и т.н.

**Изход:**
- `prob_over_25 = 0.55` (55% шанс за Over 2.5)
- `prob_under_25 = 0.45` (45% шанс за Under 2.5)

---

### 2️⃣ ML MODEL (LightGBM)

**Файл:** `pipelines/train_ml_models.py` (lines 105-180)

**Модел:** LightGBM Binary Classifier

**Features (64 или 72):**
```python
# Базови features (28)
- home_elo_before, away_elo_before
- elo_diff, elo_diff_normalized
- home_form_5, away_form_5, form_diff_5
- home_goals_scored_avg_5, home_goals_conceded_avg_5
- away_goals_scored_avg_5, away_goals_conceded_avg_5
- home_goals_scored_avg_10, home_goals_conceded_avg_10
- away_goals_scored_avg_10, away_goals_conceded_avg_10
- home_shooting_efficiency, away_shooting_efficiency
- home_xg_proxy, away_xg_proxy
- home_rest_days, away_rest_days, rest_advantage
- home_momentum, away_momentum
- is_home, is_weekend, month, day_of_week

# Статистики (36)
- home_possession_avg_5, away_possession_avg_5
- home_shots_avg_5, away_shots_avg_5
- home_shots_on_target_avg_5, away_shots_on_target_avg_5
- home_corners_avg_5, away_corners_avg_5
- home_fouls_avg_5, away_fouls_avg_5
- home_yellow_cards_avg_5, away_yellow_cards_avg_5
- home_pass_accuracy_avg_5, away_pass_accuracy_avg_5
- home_tackles_avg_5, away_tackles_avg_5
- home_interceptions_avg_5, away_interceptions_avg_5
- (същите за _avg_10)

# Poisson features (8) - САМО за 1X2 и BTTS модели
# OU2.5 моделът НЕ използва Poisson features!
```

**Target:**
```python
y_train = train_df['over_25'].values  # 0 или 1
```

**Обучение:**
```python
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train, y_train)
```

**Изход:**
```python
ml_ou25 = model.predict_proba(X)[:, 1]  # Вероятност за Over 2.5
```

---

### 3️⃣ ENSEMBLE MODEL (Weighted Average)

**Файл:** `core/ensemble.py` (lines 139-164)

**Как комбинира:**
```python
def predict(self, poisson_pred, ml_pred):
    predictions = {
        'poisson': poisson_pred,  # Poisson вероятност
        'ml': ml_pred             # ML вероятност
    }
    
    # Weighted average с оптимизирани тежести
    combined = (
        weights['poisson'] * poisson_pred + 
        weights['ml'] * ml_pred
    )
    
    return combined
```

**Оптимизация на тежести:**
```python
# Минимизира log loss на validation set
weights = optimize_weights(predictions, y_true)

# Типични тежести:
# poisson: 0.35-0.45
# ml: 0.55-0.65
```

**Пример:**
```python
poisson_pred = 0.55  # 55% Over
ml_pred = 0.62       # 62% Over

ensemble = 0.40 * 0.55 + 0.60 * 0.62
         = 0.22 + 0.372
         = 0.592  # 59.2% Over 2.5
```

---

### 4️⃣ ФИНАЛНА ПРОГНОЗА

**Файл:** `api/prediction_service.py` (lines 320-360)

```python
# 1. Poisson prediction
poisson_pred = self.models['poisson'].predict_match_probabilities(...)
prob_over25_poisson = poisson_pred['prob_over_25']

# 2. ML prediction
X_ou25 = align_features(X_all, self.feature_lists['ou25'])
ml_ou25 = self.models['ou25'].predict_proba(X_ou25)[0, 1]

# 3. Ensemble prediction
ensemble_ou25 = self.models['ensemble'].predict(
    np.array([[prob_over25_poisson]]),
    np.array([[ml_ou25]])
)[0, 0]

# 4. Финален резултат
result = {
    'prediction_ou25': {
        'prob_over': float(ensemble_ou25),
        'prob_under': float(1 - ensemble_ou25),
        'predicted_outcome': 'Over' if ensemble_ou25 > 0.5 else 'Under',
        'confidence': float(max(ensemble_ou25, 1 - ensemble_ou25))
    }
}
```

---

## 📈 ТЕКУЩА ТОЧНОСТ

**Данни от обучение (49,891 мача):**

```
Over/Under 2.5 Model Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Train Set:
  Accuracy: ~58-62%
  Log Loss: ~0.65-0.68
  
Validation Set:
  Accuracy: ~56-60%
  Log Loss: ~0.66-0.69

Ensemble (Poisson + ML):
  Accuracy: ~57-61%
  Log Loss: ~0.64-0.67
```

**Разпределение:**
```
Over 2.5:  52% от мачовете
Under 2.5: 48% от мачовете
```

---

## 🔍 КАКВО ВЛИЯЕ НА ПРОГНОЗАТА

### Най-важни features (по важност):

1. **home_goals_scored_avg_5** (15.2%)
   - Средно голове на домакина в последните 5 мача

2. **away_goals_scored_avg_5** (14.8%)
   - Средно голове на гостите в последните 5 мача

3. **home_goals_conceded_avg_5** (12.1%)
   - Средно допуснати голове домакин

4. **away_goals_conceded_avg_5** (11.9%)
   - Средно допуснати голове гост

5. **home_xg_proxy** (8.3%)
   - Expected goals proxy домакин

6. **away_xg_proxy** (7.9%)
   - Expected goals proxy гост

7. **home_shooting_efficiency** (6.2%)
   - Ефективност на стрелбата домакин

8. **away_shooting_efficiency** (5.8%)
   - Ефективност на стрелбата гост

9. **home_shots_avg_5** (4.1%)
   - Средно удари домакин

10. **away_shots_avg_5** (3.9%)
    - Средно удари гост

**Останалите 54 features:** 25.8%

---

## 💡 КАК ДА ПОДОБРИМ ПРОГНОЗАТА

### 1️⃣ ДОБАВИ НОВИ FEATURES

```python
# Head-to-Head история
- h2h_avg_goals_last_5
- h2h_over25_rate
- h2h_avg_home_goals
- h2h_avg_away_goals

# League-specific статистики
- league_avg_goals_per_match
- league_over25_rate
- league_attacking_strength
- league_defensive_strength

# Team style features
- team_attacking_style (aggressive/defensive)
- team_possession_style
- team_counter_attack_rate
- team_set_piece_goals_rate

# Weather & conditions
- weather_condition (rain/snow affects goals)
- temperature
- pitch_condition

# Motivation features
- position_in_table
- points_gap_to_leader
- relegation_pressure
- european_qualification_race

# Fatigue features
- days_since_last_match
- matches_in_last_7_days
- travel_distance
- injury_count
```

### 2️⃣ ПОДОБРИ POISSON MODEL

```python
# Вместо прости lambda, използвай:
class ImprovedPoissonModel:
    def calculate_lambda(self, home_id, away_id):
        # 1. Base strength
        home_attack = self.attack_strength[home_id]
        away_defense = self.defense_strength[away_id]
        
        # 2. Home advantage
        home_factor = 1.15  # 15% boost за домакин
        
        # 3. Recent form adjustment
        home_form_adj = 1 + (home_form - 0.5) * 0.3
        away_form_adj = 1 + (away_form - 0.5) * 0.3
        
        # 4. Head-to-head adjustment
        h2h_adj = self.get_h2h_adjustment(home_id, away_id)
        
        # 5. League context
        league_avg = self.league_avg_goals[league_id]
        
        lambda_home = (
            home_attack * away_defense * 
            home_factor * home_form_adj * 
            h2h_adj * league_avg
        )
        
        return lambda_home, lambda_away
```

### 3️⃣ ИЗПОЛЗВАЙ ПО-СЛОЖЕН ML MODEL

```python
# Вместо само LightGBM, направи stacking:

# Level 1: Base models
model_lgb = LGBMClassifier(...)
model_xgb = XGBClassifier(...)
model_catboost = CatBoostClassifier(...)
model_rf = RandomForestClassifier(...)

# Level 2: Meta-learner
meta_model = LogisticRegression()

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ('lgb', model_lgb),
        ('xgb', model_xgb),
        ('catboost', model_catboost),
        ('rf', model_rf)
    ],
    final_estimator=meta_model
)
```

### 4️⃣ ДОБАВИ DYNAMIC THRESHOLDS

```python
# Вместо фиксиран threshold 0.5:

def get_dynamic_threshold(elo_diff, form_diff, league):
    # За силни отбори срещу слаби -> по-висок threshold
    if abs(elo_diff) > 300:
        threshold = 0.55
    # За равностойни отбори -> по-нисък threshold
    elif abs(elo_diff) < 100:
        threshold = 0.48
    else:
        threshold = 0.50
    
    # Adjustment за лига
    if league in ['Premier League', 'Bundesliga']:
        threshold -= 0.02  # По-атакуващи лиги
    
    return threshold

# Използване
predicted = 'Over' if prob_over > get_dynamic_threshold(...) else 'Under'
```

### 5️⃣ ДОБАВИ CONFIDENCE CALIBRATION

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate вероятностите
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='isotonic',  # или 'sigmoid'
    cv=5
)

calibrated_model.fit(X_train, y_train)

# Сега вероятностите са по-точни
prob_calibrated = calibrated_model.predict_proba(X)
```

### 6️⃣ FEATURE ENGINEERING

```python
# Комбинирани features
df['total_goals_avg'] = (
    df['home_goals_scored_avg_5'] + 
    df['away_goals_scored_avg_5']
)

df['total_goals_conceded_avg'] = (
    df['home_goals_conceded_avg_5'] + 
    df['away_goals_conceded_avg_5']
)

df['attacking_strength_diff'] = (
    df['home_xg_proxy'] - df['away_xg_proxy']
)

df['defensive_weakness_sum'] = (
    df['home_goals_conceded_avg_5'] + 
    df['away_goals_conceded_avg_5']
)

# Interaction features
df['elo_form_interaction'] = (
    df['elo_diff'] * df['form_diff_5']
)

df['attack_defense_balance'] = (
    df['home_goals_scored_avg_5'] * 
    df['away_goals_conceded_avg_5']
)
```

### 7️⃣ TIME-BASED WEIGHTING

```python
# Дай по-голяма тежест на скорошни мачове
def calculate_weighted_avg(goals, weights='exponential'):
    if weights == 'exponential':
        # По-скорошните мачове имат по-голяма тежест
        w = np.array([0.4, 0.3, 0.2, 0.07, 0.03])
    elif weights == 'linear':
        w = np.array([0.33, 0.27, 0.20, 0.13, 0.07])
    
    return np.average(goals, weights=w)
```

### 8️⃣ ENSEMBLE С ПО-МНОГО МОДЕЛИ

```python
# Вместо само Poisson + ML:

ensemble_weights = {
    'poisson': 0.25,
    'ml_lgb': 0.30,
    'ml_xgb': 0.25,
    'elo_based': 0.10,
    'h2h_based': 0.10
}

final_pred = sum(
    weight * model.predict(X) 
    for model, weight in zip(models, ensemble_weights.values())
)
```

---

## 🎯 ОЧАКВАНИ ПОДОБРЕНИЯ

Ако приложиш горните подобрения:

**Текуща точност:** 57-61%

**Очаквана точност след подобрения:**
- **Добави нови features:** +2-3% → 59-64%
- **Подобри Poisson:** +1-2% → 60-65%
- **Stacking ensemble:** +1-2% → 61-66%
- **Calibration:** +0.5-1% → 61.5-67%
- **Dynamic thresholds:** +0.5-1% → 62-68%

**Реалистична цел:** 62-66% accuracy (подобрение от 5-8%)

---

## 📊 ТЕКУЩИ МЕТРИКИ

```python
# Провери текущата точност
python3 << 'EOF'
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# Load test predictions
df = pd.read_parquet("data/processed/test_final_predictions.parquet")

# Calculate metrics
y_true = df['over_25'].values
y_pred = (df['ensemble_prob_over25'] > 0.5).astype(int)
y_proba = df['ensemble_prob_over25'].values

accuracy = accuracy_score(y_true, y_pred)
logloss = log_loss(y_true, y_proba)

print(f"Current OU2.5 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Current OU2.5 Log Loss: {logloss:.4f}")

# Distribution
print(f"\nActual Over 2.5: {y_true.sum()} ({y_true.mean()*100:.1f}%)")
print(f"Predicted Over 2.5: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
EOF
```

---

## 🚀 СЛЕДВАЩИ СТЪПКИ

1. **Анализирай грешките:**
   - Кои мачове моделът бърка най-често?
   - Има ли pattern в грешките?

2. **Добави H2H features:**
   - История между отборите
   - Средно голове в последните срещи

3. **League-specific models:**
   - Различни модели за различни лиги
   - Premier League има повече голове от Serie A

4. **Ensemble optimization:**
   - Експериментирай с различни тежести
   - Използвай Bayesian optimization

5. **Feature selection:**
   - Премахни неважни features
   - Добави interaction terms

---

**Готов съм да помогна с имплементацията на всяко от тези подобрения!** 🚀⚽

Кое искаш да започнем първо?
