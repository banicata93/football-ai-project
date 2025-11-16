# Improved BTTS Model Summary

**Training Date:** 2025-11-13T14:29:06.755788

## Performance Metrics
### Validation Set
- **Accuracy:** 0.748
- **F1 Macro:** 0.737
- **Brier Score:** 0.1332
- **ECE:** 0.0000
- **Best Threshold:** 0.5

### Test Set (Untouched)
- **Accuracy:** 0.770
- **F1 Macro:** 0.761
- **Brier Score:** 0.1290
- **ECE:** 0.0185
- **Best Threshold:** 0.5

### Generalization Analysis
- **Val vs Test Brier Diff:** 0.0042
- **Val vs Test ECE Diff:** 0.0185
- **Generalization Quality:** Good

## Improvements vs Base Model
- **Brier Score:** +0.0133
- **ECE:** +0.0707
- **F1 Macro:** -0.002

## Feature Engineering
- **Total Features:** 46
- **Base Features:** 16
- **BTTS Features:** 30
