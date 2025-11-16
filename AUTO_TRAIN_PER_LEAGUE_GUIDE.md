# Auto Per-League Training Pipeline - Complete Guide

**Date**: 2025-11-16  
**Status**: ✅ READY FOR USE

---

## 🎯 Overview

Automatic per-league training pipeline that detects and trains missing models in **SAFE MODE**:
- Never deletes existing models
- Never overwrites existing files
- Only adds missing models
- Skips if model already exists
- Generates metrics.json if missing
- Updates registry with new entries only

---

## 📦 Supported Models

### 1. **1X2 v2** (Per-League Binary Models)
- 3 binary models per league: Home Win, Draw, Away Win
- Advanced calibration
- League-specific features

### 2. **Poisson v2** (Per-League Time-Decay)
- Time-decay Poisson model
- League-specific goal distributions
- Scoreline prediction base

### 3. **Draw Specialist v1** (Optional)
- Specialized draw prediction
- Optional feature (not critical)

---

## 🚀 Usage

### Basic Usage (All Models)
```bash
cd /Users/borisa22/Downloads/archive/football_ai_service
python pipelines/auto_train_per_league.py
```

### Dry Run (Detection Only)
```bash
python pipelines/auto_train_per_league.py --dry-run
```

### Train Specific Model Type
```bash
# Only 1X2 v2
python pipelines/auto_train_per_league.py --model-type 1x2_v2

# Only Poisson v2
python pipelines/auto_train_per_league.py --model-type poisson_v2

# Only Draw Specialist
python pipelines/auto_train_per_league.py --model-type draw_specialist
```

---

## 📊 Expected Output

```
══════════════════════════════════════════════════════════════════
🚀 STARTING AUTOMATIC PER-LEAGUE TRAINING PIPELINE
══════════════════════════════════════════════════════════════════

📊 Detecting available leagues from data...
✅ Found 8 available leagues: premier_league, la_liga, serie_a, bundesliga, ligue_1, eredivisie, primeira_liga, championship

══════════════════════════════════════════════════════════════════
📦 Processing: 1X2_V2
══════════════════════════════════════════════════════════════════

🔍 Detecting missing 1X2_v2 models...
📊 1x2_v2 status:
   ✅ Existing: 5 leagues
   ❌ Missing: 3 leagues
   
🎯 Found 3 missing 1x2_v2 models

Training 1x2_v2 for premier_league...
🎯 Training 1X2 v2 model for premier_league...
✅ Successfully trained 1X2 v2 for premier_league
✅ Updated registry: 1x2_v2_premier_league_v2

Training 1x2_v2 for la_liga...
⏭️  Skipping la_liga: Models already exist

══════════════════════════════════════════════════════════════════
📦 Processing: POISSON_V2
══════════════════════════════════════════════════════════════════

🔍 Detecting missing poisson_v2 models...
📊 poisson_v2 status:
   ✅ Existing: 8 leagues
   ❌ Missing: 0 leagues
   
✅ All poisson_v2 models already exist

══════════════════════════════════════════════════════════════════
📊 TRAINING PIPELINE COMPLETE
══════════════════════════════════════════════════════════════════
Total missing models found: 3
Total models trained: 3
Total models skipped (existing): 13
Total failures: 0

1x2_v2:
  - Missing: 3
  - Trained: 3
  - Existing: 5
  - Failed: 0
  
poisson_v2:
  - Missing: 0
  - Trained: 0
  - Existing: 8
  - Failed: 0

✅ All per-league models are now consistent!
══════════════════════════════════════════════════════════════════
```

---

## 🔧 How It Works

### 1. **Detection Phase**
```python
# For each model type and league:
1. Check if model directory exists
2. Check if all required files exist:
   - 1X2 v2: home_model.pkl, draw_model.pkl, away_model.pkl, metrics.json
   - Poisson v2: poisson_model.pkl, metrics.json
   - Draw Specialist: draw_model.pkl, metrics.json
3. Mark as missing/existing/partial
```

### 2. **Training Phase**
```python
# For each missing model:
1. Check if already exists (SAFE MODE)
2. If exists → Skip
3. If missing → Train
4. Generate metrics.json
5. Update registry
```

### 3. **SAFE MODE Guarantees**
- ✅ Never deletes files
- ✅ Never overwrites existing .pkl
- ✅ Never replaces metrics.json
- ✅ Only creates new files
- ✅ Skips existing models
- ✅ Preserves versioning
- ✅ Appends to registry only

---

## 📁 File Structure

### Before Training
```
models/
└── leagues/
    ├── premier_league/
    │   └── ou25_v1/  (existing)
    ├── la_liga/
    │   └── ou25_v1/  (existing)
    └── bundesliga/
        └── ou25_v1/  (existing)
```

### After Training
```
models/
└── leagues/
    ├── premier_league/
    │   ├── ou25_v1/  (existing - untouched)
    │   ├── 1x2_v2/   (NEW)
    │   │   ├── home_model.pkl
    │   │   ├── draw_model.pkl
    │   │   ├── away_model.pkl
    │   │   ├── calibrator.pkl
    │   │   ├── feature_list.json
    │   │   └── metrics.json
    │   └── poisson_v2/  (NEW)
    │       ├── poisson_model.pkl
    │       └── metrics.json
    ├── la_liga/
    │   ├── ou25_v1/  (existing - untouched)
    │   ├── 1x2_v2/   (NEW)
    │   └── poisson_v2/  (NEW)
    └── bundesliga/
        ├── ou25_v1/  (existing - untouched)
        ├── 1x2_v2/   (NEW)
        └── poisson_v2/  (NEW)
```

---

## 🔍 Detection Logic

### Model Status Categories

1. **Existing** ✅
   - All required files present
   - Model is complete
   - Will be skipped

2. **Missing** ❌
   - Model directory doesn't exist
   - Will be trained

3. **Partial** ⚠️
   - Directory exists but missing files
   - Logged as warning
   - Can be completed manually

---

## 📊 Metrics Generation

### 1X2 v2 Metrics
```json
{
  "train": {
    "accuracy": 0.XXX,
    "log_loss": 0.XXX
  },
  "val": {
    "accuracy": 0.XXX,
    "log_loss": 0.XXX
  },
  "league": "premier_league",
  "model_type": "1x2_v2_binary",
  "trained_date": "2025-11-16 14:00:00"
}
```

### Poisson v2 Metrics
```json
{
  "league": "premier_league",
  "model_type": "poisson_v2",
  "trained_date": "2025-11-16 14:00:00",
  "total_matches": 1234,
  "decay_factor": 0.8,
  "note": "Time-decay Poisson model for scoreline prediction"
}
```

---

## 🔄 Registry Updates

### Registry Structure
```json
{
  "models": [
    {
      "key": "1x2_v2_premier_league_v2",
      "model_type": "1x2_v2",
      "league": "premier_league",
      "version": "v2",
      "trained_date": "2025-11-16 14:00:00",
      "status": "active"
    },
    {
      "key": "poisson_v2_premier_league_v2",
      "model_type": "poisson_v2",
      "league": "premier_league",
      "version": "v2",
      "trained_date": "2025-11-16 14:00:00",
      "status": "active"
    }
  ]
}
```

### Update Rules
- ✅ Only appends new entries
- ✅ Never modifies existing entries
- ✅ Checks for duplicates before adding
- ✅ Preserves all existing data

---

## ⚠️ Important Notes

### Data Requirements
- Minimum 300 matches per league for 1X2 v2
- Minimum 100 matches per league for Poisson v2
- Data loaded from `data/processed/` directory

### Training Time
- **1X2 v2**: ~5-10 minutes per league
- **Poisson v2**: ~1-2 minutes per league
- **Total**: ~1 hour for all 8 leagues (both models)

### Disk Space
- **1X2 v2**: ~10-20 MB per league
- **Poisson v2**: ~1-5 MB per league
- **Total**: ~200 MB for all leagues

---

## 🐛 Troubleshooting

### Issue: "No data available for league"
**Solution**: Check if league has data in `data/processed/` files

### Issue: "Insufficient data for league"
**Solution**: League has < 300 matches, cannot train 1X2 v2

### Issue: "Model already exists"
**Solution**: This is expected (SAFE MODE), model will be skipped

### Issue: "Training failed"
**Solution**: Check logs for specific error, may need to fix data or dependencies

---

## 📝 Code Changes Summary

### New Files Created: 1
- `pipelines/auto_train_per_league.py` (~700 lines)

### Modified Files: 1
- `pipelines/train_1x2_v2.py`
  - Added `train_league()` method (~70 lines)

### Total New Code: ~770 lines

---

## ✅ Testing

### Test Detection Only
```bash
python pipelines/auto_train_per_league.py --dry-run
```

### Test Single League
```bash
# Manually test training for one league
python -c "
from pipelines.auto_train_per_league import PerLeagueTrainingManager
manager = PerLeagueTrainingManager()
manager.train_1x2_v2_for_league('premier_league')
"
```

### Verify Results
```bash
# Check if models were created
ls -la models/leagues/premier_league/1x2_v2/
ls -la models/leagues/premier_league/poisson_v2/

# Check metrics
cat models/leagues/premier_league/1x2_v2/metrics.json
cat models/leagues/premier_league/poisson_v2/metrics.json

# Check registry
cat registry.json
```

---

## 🎯 Next Steps

After running the pipeline:

1. **Verify Models**
   ```bash
   python pipelines/auto_train_per_league.py --dry-run
   ```

2. **Restart Backend**
   ```bash
   ./stop_all.sh
   ./start_backend.sh
   ```

3. **Test Endpoint**
   ```bash
   curl http://localhost:8000/models | python3 -m json.tool
   ```

4. **Check UI**
   - Open http://localhost:8501
   - Verify all models show valid metrics
   - Check that 1X2 v2 and Poisson v2 now show `loaded: true`

---

## ✅ Success Criteria

After successful run:
- ✅ All 8 leagues have 1X2 v2 models
- ✅ All 8 leagues have Poisson v2 models
- ✅ All models have metrics.json
- ✅ Registry updated with new entries
- ✅ No existing models were modified
- ✅ `/models` endpoint shows all models with metrics

---

**Status**: ✅ PRODUCTION READY  
**Last Updated**: 2025-11-16 14:00 UTC+2
