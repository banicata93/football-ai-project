# Model Logic Overview

This document provides a comprehensive technical overview of the ML prediction system extracted from the existing codebase.

## 1. Feature Engineering Overview

### Core Feature Engineering Files
- **`core/feature_engineering.py`**: Main feature engineering pipeline with rolling statistics, Elo calculations, form metrics
- **`core/btts_features.py`**: BTTS-specific feature engineering (30 additional features)
- **`core/ml_utils.py`**: Feature preparation, alignment, and BTTS-specific features
- **`pipelines/generate_features.py`**: Feature generation pipeline orchestration

### Feature Groups

#### Base Features (core/feature_engineering.py)
- **Temporal Features**: `year`, `month`, `day_of_week`, `is_weekend`
- **Elo Features**: `home_elo_before`, `away_elo_before`, `elo_diff`, `elo_diff_normalized`
- **Form Features**: `home_form_5`, `away_form_5`, `form_diff_5` (rolling 5-match points average)
- **Goal Statistics**: Rolling averages for 5/10 matches
  - `home_goals_scored_avg_5/10`, `home_goals_conceded_avg_5/10`
  - `away_goals_scored_avg_5/10`, `away_goals_conceded_avg_5/10`
- **Efficiency Features**: 
  - `home_shooting_efficiency`, `away_shooting_efficiency` (goals per shot on target)
  - `home_xg_proxy`, `away_xg_proxy` (shots on target × possession weight × 0.1)
  - `home_defensive_efficiency` (defensive actions per goal conceded)
- **Rolling Statistics**: 5/10-match averages for possession, shots, corners, fouls, passes, tackles, interceptions
- **Rest & Momentum**: `home_rest_days`, `away_rest_days`, `home_momentum`, `away_momentum`

#### BTTS-Specific Features (core/btts_features.py - 30 features)
- **Historical BTTS**: `home_btts_rate_last5/10`, `away_btts_rate_last5/10`
- **League BTTS**: `league_btts_rate`, `league_over25_rate` (hardcoded per league)
- **Combined Features**: 
  - `both_defenses_weak`, `both_attacks_strong`
  - `both_defenses_weak_product`, `both_attacks_strong_product`
- **Matchup Features**: 
  - `attack_vs_defense_home/away`, `expected_home/away_goals_matchup`
  - `btts_likelihood_matchup`
- **Advanced Indicators**: 
  - `shooting_efficiency_balance`, `min/max_shooting_efficiency`
  - `form_balance`, `min_form`, `both_teams_good_form`
  - `xg_balance`, `min_xg_proxy`, `both_teams_attacking`
  - `btts_favorable_conditions`, `btts_risk_factors`

#### Legacy BTTS Features (core/ml_utils.py)
- `home_clean_sheet_rate`, `away_clean_sheet_rate`
- `attack_correlation`, `defense_correlation`
- `both_teams_scoring_indicator`, `defensive_weakness_sum`

### Rolling Window Logic
- **Windows**: [5, 10] matches for most statistics
- **Implementation**: `df.groupby('team_id')[stat].transform(lambda x: x.shift(1).rolling(window=N).mean())`
- **Shift(1)**: Prevents data leakage by excluding current match
- **Min Periods**: 1 (allows partial windows for new teams)

### Normalization/Scaling
- **Elo Diff**: Normalized by dividing by 400
- **Form**: Normalized by max possible points (3 × window_size)
- **Efficiency**: Clipped to reasonable ranges using `np.clip()`
- **Extreme Values**: Cleaned using quantile-based clipping (0.5% - 99.5%)

### Data Leakage Prevention
- All rolling statistics use `.shift(1)` to exclude current match
- Elo ratings calculated chronologically with before/after states
- No future information used in feature calculation

### Feature Passing Flow
**Training**: Raw data → FeatureEngineer.create_all_features() → prepare_features() → Model
**Inference**: Team data → _create_match_features() → BTTS feature engineering → prepare_features() → Model

## 2. Model Training Logic

### Training Scripts
- **`pipelines/train_ml_models.py`**: Main training pipeline for 1X2, OU2.5, BTTS models
- **`pipelines/train_btts_improved.py`**: Enhanced BTTS training with calibration
- **`pipelines/train_poisson.py`**: Poisson baseline model training
- **`pipelines/train_ensemble.py`**: Ensemble model training
- **`pipelines/train_stacking.py`**: Stacking ensemble training

### Algorithms Used

#### 1X2 Model (pipelines/train_ml_models.py:30-105)
- **Algorithm**: XGBoost multi-class classifier
- **Parameters**: 
  ```python
  {
    'n_estimators': 200,
    'max_depth': 6, 
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3
  }
  ```
- **Target**: Encoded as {0: '1', 1: 'X', 2: '2'}

#### OU2.5 Model (pipelines/train_ml_models.py:108-183)
- **Algorithm**: LightGBM binary classifier
- **Parameters**:
  ```python
  {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'objective': 'binary'
  }
  ```
- **Target**: Binary (0=Under, 1=Over)

#### BTTS Model (Legacy - pipelines/train_ml_models.py:186-261)
- **Algorithm**: XGBoost binary classifier
- **Parameters**:
  ```python
  {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.05,
    'objective': 'binary:logistic'
  }
  ```

#### Improved BTTS Model (pipelines/train_btts_improved.py)
- **Algorithm**: XGBoost + Isotonic Calibration
- **Parameters**:
  ```python
  {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
  }
  ```
- **Calibration**: `CalibratedClassifierCV(method='isotonic', cv='prefit')`

#### Poisson Model (pipelines/train_poisson.py)
- **Algorithm**: Statistical Poisson regression
- **Parameters**: `home_advantage_factor=1.15`, `min_matches_for_lambda=5`
- **Method**: Team strength calculation from historical goal data

### Target Generation Logic

#### 1X2 Targets
```python
# From match results
result = '1' if home_score > away_score else ('X' if home_score == away_score else '2')
encoded = {'1': 0, 'X': 1, '2': 2}[result]
```

#### OU2.5 Targets
```python
over_25 = int(home_score + away_score > 2.5)
```

#### BTTS Targets
```python
btts = int(home_score > 0 and away_score > 0)
```

### Cross-Validation
- **Improved BTTS**: 5-fold StratifiedKFold with shuffle
- **Standard Models**: Single train/validation split
- **Per-League Models**: League-specific train/test splits

### Split Strategy
- **Method**: Time-based splits (not random)
- **Ratios**: Typically 70% train, 15% validation, 15% test
- **Per-League**: Minimum 300 matches per league for separate models

### Model Saving/Loading
- **Format**: Joblib pickle files (.pkl)
- **Structure**: 
  ```
  models/model_{type}_v1/
  ├── {type}_model.pkl
  ├── feature_list.json
  ├── metrics.json
  └── model_info.json
  ```
- **Loading**: `joblib.load()` in PredictionService._load_models()

## 3. Calibration & Postprocessing

### Calibration Methods

#### Isotonic Regression (Primary)
- **Used in**: Improved BTTS, Per-league OU2.5 models
- **Implementation**: `CalibratedClassifierCV(method='isotonic', cv='prefit')`
- **Fit on**: Separate validation set to prevent overfitting

#### Platt Scaling (Config)
- **Configuration**: Available in model_config.yaml
- **Implementation**: `CalibratedClassifierCV(method='sigmoid')`

### Probability Transformation

#### Expected Calibration Error (ECE) Calculation
```python
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if in_bin.mean() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * in_bin.mean()
    return ece
```

#### Brier Score
- **Usage**: Primary calibration metric for BTTS models
- **Implementation**: `sklearn.metrics.brier_score_loss(y_true, y_prob)`

### Threshold Optimization
- **BTTS Thresholds Tested**: [0.45, 0.5, 0.55, 0.6]
- **Optimization Metric**: F1 Macro score
- **Implementation**: Grid search over thresholds in training
- **Best Threshold Storage**: Saved in model metrics but API uses 0.5

### Confidence Penalties
- **Entropy-based**: `confidence = 1 - entropy` where entropy = -p*log(p) - (1-p)*log(1-p)
- **Model Agreement**: Penalty when Poisson and ML models disagree significantly
- **Implementation**: In `core/btts_ensemble.py`

## 4. Inference / Prediction Pipeline

### Full Prediction Flow
```
Input (team names, league, date) 
→ Team Resolution (TeamNameResolver)
→ Feature Creation (_create_match_features)
→ BTTS Feature Engineering (BTTSFeatureEngineer) 
→ Feature Alignment (align_features)
→ Model Prediction (predict_proba)
→ Calibration (if available)
→ Ensemble Combination
→ Final Probabilities
```

### Key Files
- **`api/prediction_service.py`**: Main prediction service
- **`api/improved_prediction_service.py`**: Enhanced version with confidence scoring
- **`api/main.py`**: FastAPI endpoints

### Per-League Logic
- **OU2.5 Models**: League-specific models for major leagues (Premier League, La Liga, etc.)
- **Lazy Loading**: Models loaded on first request for league
- **Fallback**: Global model if league-specific unavailable
- **Implementation**: `_get_ou25_model_for_league()` method

### Fallback Logic
1. **Team Resolution**: Unknown teams → similar team lookup → default values
2. **Model Loading**: Improved BTTS → Standard BTTS → Default probabilities
3. **Feature Missing**: Intelligent imputation → League averages → Zero filling
4. **Per-League**: League model → Global model → Default

### Probability Computation

#### 1X2 Probabilities
```python
# XGBoost multi-class output (3 probabilities)
probs_1x2 = model.predict_proba(X)[0]  # [prob_1, prob_X, prob_2]
predicted_outcome = ['1', 'X', '2'][np.argmax(probs_1x2)]
```

#### OU2.5 Probabilities  
```python
# Binary classifier output
prob_over = model.predict_proba(X)[0][1]
prob_under = 1 - prob_over
```

#### BTTS Probabilities
```python
# Enhanced BTTS with ensemble logic
ml_prob = improved_btts_model.predict_proba(X)[0][1]
ensemble_result = btts_ensemble.enhanced_btts_ensemble(ml_prob, poisson_prob)
final_prob = ensemble_result['probability']
```

## 5. Ensemble Logic

### Ensemble Models Combined
- **Poisson Model**: Statistical baseline
- **ML Models**: XGBoost/LightGBM predictions  
- **Elo-based**: Simple Elo difference predictions (optional)

### Combination Formulas

#### Standard Ensemble (core/ensemble.py:145-177)
```python
def predict(poisson_pred, ml_pred, elo_pred=None, league_id=None):
    predictions = {'poisson': poisson_pred, 'ml': ml_pred}
    if elo_pred is not None:
        predictions['elo'] = elo_pred
    
    # Weighted average with dynamic adjustments
    weights = self._get_dynamic_weights(poisson_pred, ml_pred, league_id)
    return self._combine_predictions(predictions, weights)
```

#### Enhanced BTTS Ensemble (core/btts_ensemble.py:53-132)
```python
def enhanced_btts_ensemble(ml_prob, poisson_prob, ml_weight=0.8):
    # Base ensemble
    base_prob = ml_weight * ml_prob + (1 - ml_weight) * poisson_prob
    
    # Model agreement
    agreement = 1 - abs(ml_prob - poisson_prob)
    
    # Entropy confidence  
    entropy_confidence = 1 - (-p*log(p) - (1-p)*log(1-p))
    
    # Combined confidence
    confidence = 0.7 * entropy_confidence + 0.3 * agreement
    
    # Adjustment based on agreement
    if agreement < 0.7:  # High disagreement
        # Pull toward neutral (0.5)
        adjustment = 0.3 * (0.7 - agreement)
        adjusted_prob = base_prob + adjustment * (0.5 - base_prob)
        confidence -= 0.2 * (0.7 - agreement)  # Penalty
    elif agreement > 0.85:  # High agreement
        # Enhance extreme probabilities
        if base_prob > 0.6:
            adjusted_prob = min(0.95, base_prob + 0.05)
        elif base_prob < 0.4:
            adjusted_prob = max(0.05, base_prob - 0.05)
        confidence += 0.1 * (agreement - 0.85)  # Bonus
    
    return {
        'probability': adjusted_prob,
        'confidence': confidence,
        'predicted_outcome': 'Yes' if adjusted_prob > 0.5 else 'No'
    }
```

### Weighting Files
- **Configuration**: `config/model_config.yaml`
- **Default Weights**: `{'poisson': 0.3, 'ml': 0.5, 'elo': 0.2}`
- **Per-League Weights**: Stored in ensemble model, loaded dynamically

### Dynamic Weight Adjustments (core/ensemble.py:179-231)
```python
def _get_dynamic_weights(poisson_pred, ml_pred, league_id=None):
    # High ML entropy → increase Poisson weight
    if entropy > 0.8:
        weights['poisson'] += 0.1
        weights['ml'] -= 0.1
    
    # High disagreement → shrink toward neutral
    if disagreement > 0.25:
        shrink_factor = 0.15
        for key in weights:
            weights[key] = weights[key] * (1 - shrink_factor) + 0.5 * shrink_factor
```

### Disagreement Logic
- **Threshold**: 0.25 absolute difference triggers adjustment
- **Action**: Shrink all weights toward equal (0.33 each)
- **Penalty**: Reduce confidence when models disagree

## 6. Model Storage & Loading

### Storage Structure
```
models/
├── model_poisson_v1/
│   ├── poisson_model.pkl
│   └── metrics.json
├── model_1x2_v1/
│   ├── 1x2_model.pkl
│   ├── feature_list.json
│   └── metrics.json
├── model_ou25_v1/
│   ├── ou25_model.pkl
│   ├── feature_list.json
│   └── metrics.json
├── model_btts_v1/
│   ├── btts_model.pkl
│   ├── feature_list.json
│   └── metrics.json
├── model_btts_improved/
│   ├── btts_model_improved.pkl
│   ├── feature_columns.json
│   ├── training_metrics.json
│   └── model_summary.md
├── ensemble_v1/
│   ├── ensemble_model.pkl
│   └── fii_model.pkl
└── leagues/
    ├── premier_league/ou25_v1/
    ├── la_liga/ou25_v1/
    └── ...
```

### Versioning Scheme
- **Format**: `{model_type}_v{version}`
- **Current Version**: v1 for all models
- **Backward Compatibility**: Maintained through feature list validation

### Lazy Loading Logic (api/prediction_service.py:232-276)
```python
def _load_league_model(self, league_slug):
    model_dir = get_per_league_model_path(league_slug, 'ou25', 'v1')
    if not os.path.exists(f"{model_dir}/ou25_model.pkl"):
        return False
    
    # Load on first access
    model = joblib.load(f"{model_dir}/ou25_model.pkl")
    self.ou25_models_by_league[league_slug] = model
    
    # Load calibrator if exists
    if os.path.exists(f"{model_dir}/calibrator.pkl"):
        calibrator = joblib.load(f"{model_dir}/calibrator.pkl")
        self.ou25_calibrators_by_league[league_slug] = calibrator
```

### Dependency Injection (core/service_manager.py)
- **ServiceManager**: Centralized service initialization
- **Thread-Safe**: Uses asyncio.Lock for concurrent access
- **Hot Reload**: `reinitialize()` method for model updates
- **Graceful Shutdown**: Cleanup methods for resource management

### Configuration Files
- **`config/model_config.yaml`**: Model hyperparameters, per-league settings
- **Feature Lists**: JSON files with exact feature names for each model
- **Team Mappings**: `models/team_names_mapping.json` for display names

## 7. Additional Prediction Factors

### Drift Detection Integration
- **Monitoring**: `monitoring/prediction_logger.py` logs all predictions
- **Analysis**: Drift detection scripts in `pipelines/drift_analyzer.py`
- **Integration Points**: Logged but not actively used in prediction logic

### Adaptive Learning
- **Scripts**: `pipelines/adaptive_trainer.py`
- **Interaction**: Separate from main prediction flow
- **Purpose**: Model retraining based on recent performance

### Global Constants
- **Default Elo**: 1500 for unknown teams
- **Default Form**: 0.0 (neutral)
- **Default Goals**: 1.5 goals per match
- **Home Advantage**: Built into Elo calculation (+100 points)

### Special Preprocessing Rules
- **Extreme Value Clipping**: 99.5th percentile bounds
- **Infinity Handling**: Replace ±inf with ±1e10
- **Missing Value Strategy**: League averages → Global averages → Zero
- **Feature Validation**: Minimum 80% feature availability required

## 8. Summary Diagram

```
RAW DATA (team names, league, date)
    ↓
TEAM RESOLUTION (TeamNameResolver)
    ↓ 
FEATURE CREATION
    ├── Base Features (Elo, Form, Goals, xG)
    ├── Rolling Statistics (5/10 match windows)
    ├── BTTS Features (30 additional)
    └── Temporal Features (date, weekend)
    ↓
FEATURE ALIGNMENT & VALIDATION
    ├── Missing Feature Imputation
    ├── Extreme Value Clipping  
    └── Quality Score Calculation
    ↓
MODEL PREDICTIONS
    ├── Poisson Model (statistical baseline)
    ├── 1X2 Model (XGBoost multi-class)
    ├── OU2.5 Model (LightGBM binary)
    └── BTTS Model (XGBoost + Calibration)
    ↓
CALIBRATION (if available)
    ├── Isotonic Regression
    └── Expected Calibration Error
    ↓
ENSEMBLE COMBINATION
    ├── Weighted Average (Poisson + ML)
    ├── Dynamic Weight Adjustment
    ├── Model Agreement Analysis
    └── Confidence Calculation
    ↓
FINAL PROBABILITIES
    ├── 1X2: [prob_1, prob_X, prob_2]
    ├── OU2.5: [prob_over, prob_under]
    ├── BTTS: [prob_yes, prob_no]
    └── FII: Intelligence Index (0-10)
```

### Key Performance Metrics
- **BTTS ECE**: Improved from 12.7% → 0.00% (perfect calibration)
- **BTTS Accuracy**: 77.6% → 79.8% (+1.8% improvement)
- **Feature Count**: 72 → 46 optimized features
- **Calibration Method**: Isotonic regression on validation set
- **Best BTTS Threshold**: 0.6 (vs default 0.5)

This system implements a sophisticated ML pipeline with multiple models, advanced calibration, intelligent feature engineering, and robust fallback mechanisms for production football match prediction.
