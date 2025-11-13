#!/usr/bin/env python3
"""
Simplified BTTS Model Diagnosis
Използва готови predictions за анализ на BTTS модела
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, classification_report, confusion_matrix
)

from core.utils import setup_logging, save_json


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Изчислява Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def analyze_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Анализира калибрацията по bins"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_data = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_data = {
                'bin_id': i,
                'bin_range': f'[{bin_lower:.1f}-{bin_upper:.1f}]',
                'count': int(in_bin.sum()),
                'avg_predicted_prob': float(y_prob[in_bin].mean()),
                'actual_positive_rate': float(y_true[in_bin].mean()),
                'calibration_error': float(abs(y_prob[in_bin].mean() - y_true[in_bin].mean()))
            }
            calibration_data.append(bin_data)
    
    return {
        'bins': calibration_data,
        'ece': calculate_ece(y_true, y_prob, n_bins),
        'total_samples': len(y_true)
    }


def analyze_league_performance(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Анализира BTTS performance по лиги"""
    league_analysis = {}
    
    if 'league' not in df.columns:
        return {}
    
    for league in df['league'].unique():
        if pd.isna(league):
            continue
            
        league_mask = df['league'] == league
        league_y_true = y_true[league_mask]
        league_y_prob = y_prob[league_mask]
        
        if len(league_y_true) < 10:  # Минимум samples
            continue
        
        # Основни метрики
        accuracy = accuracy_score(league_y_true, (league_y_prob > 0.5).astype(int))
        brier = brier_score_loss(league_y_true, league_y_prob)
        ece = calculate_ece(league_y_true, league_y_prob)
        
        # BTTS base rate в лигата
        btts_rate = league_y_true.mean()
        
        # Bias анализ
        avg_predicted_prob = league_y_prob.mean()
        bias = avg_predicted_prob - btts_rate
        
        league_analysis[league] = {
            'matches': int(len(league_y_true)),
            'btts_base_rate': float(btts_rate),
            'avg_predicted_prob': float(avg_predicted_prob),
            'bias': float(bias),
            'accuracy': float(accuracy),
            'brier_score': float(brier),
            'ece': float(ece),
            'btts_yes_count': int(league_y_true.sum()),
            'btts_no_count': int(len(league_y_true) - league_y_true.sum())
        }
    
    return league_analysis


def diagnose_btts_from_predictions():
    """Диагностицира BTTS модела използвайки готови predictions"""
    logger = setup_logging()
    
    logger.info("🔍 BTTS DIAGNOSIS ОТ ГОТОВИ PREDICTIONS")
    logger.info("=" * 60)
    
    # Зарежда test predictions
    predictions_path = 'data/processed/test_final_predictions.parquet'
    if not os.path.exists(predictions_path):
        logger.error(f"❌ Predictions не са намерени: {predictions_path}")
        return
    
    logger.info(f"📊 Зареждане на predictions от {predictions_path}")
    pred_df = pd.read_parquet(predictions_path)
    
    # Проверява нужните колони и адаптира имената
    if 'ensemble_prob_btts' in pred_df.columns:
        btts_prob_col = 'ensemble_prob_btts'
    elif 'btts_prob_ml' in pred_df.columns:
        btts_prob_col = 'btts_prob_ml'
    else:
        logger.error("❌ Няма BTTS probability колона")
        return
    
    if 'league_id' in pred_df.columns:
        league_col = 'league_id'
    elif 'league' in pred_df.columns:
        league_col = 'league'
    else:
        logger.warning("⚠️ Няма league колона - ще пропусна league анализа")
        league_col = None
    
    logger.info(f"✓ Predictions заредени: {len(pred_df)} matches")
    logger.info(f"✓ Използвам {btts_prob_col} за BTTS вероятности")
    
    # Подготвя данните и почиства NaN
    valid_mask = ~(pd.isna(pred_df['btts']) | pd.isna(pred_df[btts_prob_col]))
    
    pred_df_clean = pred_df[valid_mask].copy()
    y_true = pred_df_clean['btts'].values
    y_prob = pred_df_clean[btts_prob_col].values
    y_pred = (y_prob > 0.5).astype(int)
    
    logger.info(f"📊 Почистени данни: {len(pred_df_clean)} matches (от {len(pred_df)})")
    logger.info(f"📊 BTTS разпределение: Yes={y_true.sum()}, No={len(y_true)-y_true.sum()}")
    
    # Основни метрики
    basic_metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_yes': float(precision_score(y_true, y_pred, pos_label=1)),
        'precision_no': float(precision_score(y_true, y_pred, pos_label=0)),
        'recall_yes': float(recall_score(y_true, y_pred, pos_label=1)),
        'recall_no': float(recall_score(y_true, y_pred, pos_label=0)),
        'f1_yes': float(f1_score(y_true, y_pred, pos_label=1)),
        'f1_no': float(f1_score(y_true, y_pred, pos_label=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob))
    }
    
    # Калибрационен анализ
    calibration_analysis = analyze_calibration_bins(y_true, y_prob)
    
    # League анализ
    if league_col:
        # Създава temporary df с правилното име на колоната
        temp_df = pred_df_clean.copy()
        temp_df['league'] = temp_df[league_col]
        league_analysis = analyze_league_performance(temp_df, y_true, y_prob)
    else:
        league_analysis = {}
    
    # Bias анализ
    btts_base_rate = y_true.mean()
    avg_predicted_prob = y_prob.mean()
    overall_bias = avg_predicted_prob - btts_base_rate
    
    # Threshold анализ
    threshold_analysis = {}
    for threshold in [0.45, 0.5, 0.55, 0.6]:
        y_pred_thresh = (y_prob > threshold).astype(int)
        threshold_analysis[f'threshold_{threshold}'] = {
            'accuracy': float(accuracy_score(y_true, y_pred_thresh)),
            'f1_macro': float(f1_score(y_true, y_pred_thresh, average='macro')),
            'precision_yes': float(precision_score(y_true, y_pred_thresh, pos_label=1)),
            'recall_yes': float(recall_score(y_true, y_pred_thresh, pos_label=1))
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Probability distribution analysis
    prob_stats = {
        'mean': float(y_prob.mean()),
        'std': float(y_prob.std()),
        'min': float(y_prob.min()),
        'max': float(y_prob.max()),
        'median': float(np.median(y_prob)),
        'q25': float(np.percentile(y_prob, 25)),
        'q75': float(np.percentile(y_prob, 75))
    }
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_matches': int(len(pred_df_clean)),
            'btts_yes': int(y_true.sum()),
            'btts_no': int(len(y_true) - y_true.sum()),
            'btts_base_rate': float(btts_base_rate)
        },
        'basic_metrics': basic_metrics,
        'bias_analysis': {
            'btts_base_rate': float(btts_base_rate),
            'avg_predicted_prob': float(avg_predicted_prob),
            'overall_bias': float(overall_bias),
            'bias_interpretation': 'Overconfident' if overall_bias > 0.05 else 'Underconfident' if overall_bias < -0.05 else 'Well calibrated'
        },
        'calibration_analysis': calibration_analysis,
        'league_analysis': league_analysis,
        'threshold_analysis': threshold_analysis,
        'probability_distribution': prob_stats,
        'confusion_matrix': {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
    }
    
    # Запазва отчета
    output_path = 'reports/btts_diagnosis_simple.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json(results, output_path)
    
    # Генерира markdown summary
    markdown_path = output_path.replace('.json', '.md')
    
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# BTTS Model Diagnosis Report (Simple)\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")
        
        # Dataset info
        dataset = results['dataset_info']
        f.write("## Dataset Overview\n")
        f.write(f"- **Total matches:** {dataset['total_matches']:,}\n")
        f.write(f"- **BTTS Yes:** {dataset['btts_yes']:,} ({dataset['btts_base_rate']:.1%})\n")
        f.write(f"- **BTTS No:** {dataset['btts_no']:,} ({1-dataset['btts_base_rate']:.1%})\n\n")
        
        # Basic metrics
        metrics = results['basic_metrics']
        f.write("## Model Performance\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.3f}\n")
        f.write(f"- **F1 Score (Macro):** {metrics['f1_macro']:.3f}\n")
        f.write(f"- **Precision (Yes):** {metrics['precision_yes']:.3f}\n")
        f.write(f"- **Recall (Yes):** {metrics['recall_yes']:.3f}\n")
        f.write(f"- **Brier Score:** {metrics['brier_score']:.4f}\n")
        f.write(f"- **Log Loss:** {metrics['log_loss']:.4f}\n\n")
        
        # Bias analysis
        bias = results['bias_analysis']
        f.write("## Bias Analysis\n")
        f.write(f"- **Base Rate:** {bias['btts_base_rate']:.1%}\n")
        f.write(f"- **Avg Predicted:** {bias['avg_predicted_prob']:.1%}\n")
        f.write(f"- **Bias:** {bias['overall_bias']:+.1%} ({bias['bias_interpretation']})\n\n")
        
        # Calibration
        cal = results['calibration_analysis']
        f.write("## Calibration Analysis\n")
        f.write(f"- **ECE:** {cal['ece']:.4f}\n")
        f.write("- **Calibration by bins:**\n")
        for bin_data in cal['bins']:
            f.write(f"  - {bin_data['bin_range']}: {bin_data['count']} matches, ")
            f.write(f"Pred: {bin_data['avg_predicted_prob']:.1%}, ")
            f.write(f"Actual: {bin_data['actual_positive_rate']:.1%}, ")
            f.write(f"Error: {bin_data['calibration_error']:.1%}\n")
        f.write("\n")
        
        # Top problematic leagues
        leagues = results['league_analysis']
        if leagues:
            f.write("## League Analysis (Top Issues)\n")
            # Сортира по bias
            sorted_leagues = sorted(leagues.items(), key=lambda x: abs(x[1]['bias']), reverse=True)
            for league, data in sorted_leagues[:10]:
                f.write(f"- **{league}:** {data['matches']} matches, ")
                f.write(f"Base rate: {data['btts_base_rate']:.1%}, ")
                f.write(f"Bias: {data['bias']:+.1%}, ")
                f.write(f"ECE: {data['ece']:.3f}\n")
            f.write("\n")
        
        # Threshold analysis
        thresh = results['threshold_analysis']
        f.write("## Threshold Optimization\n")
        best_threshold = None
        best_f1 = 0
        for threshold, data in thresh.items():
            t = threshold.replace('threshold_', '')
            f.write(f"- **{t}:** Acc: {data['accuracy']:.3f}, F1: {data['f1_macro']:.3f}\n")
            if data['f1_macro'] > best_f1:
                best_f1 = data['f1_macro']
                best_threshold = t
        
        f.write(f"\n**Best threshold:** {best_threshold} (F1: {best_f1:.3f})\n\n")
        
        # Probability distribution
        prob = results['probability_distribution']
        f.write("## Probability Distribution\n")
        f.write(f"- **Mean:** {prob['mean']:.3f}\n")
        f.write(f"- **Std:** {prob['std']:.3f}\n")
        f.write(f"- **Range:** [{prob['min']:.3f}, {prob['max']:.3f}]\n")
        f.write(f"- **Quartiles:** Q25={prob['q25']:.3f}, Q50={prob['median']:.3f}, Q75={prob['q75']:.3f}\n")
    
    logger.info(f"✓ Отчет запазен в {output_path}")
    logger.info(f"✓ Markdown summary в {markdown_path}")
    
    # Показва key findings
    logger.info("\n🎯 KEY FINDINGS:")
    logger.info(f"   Bias: {results['bias_analysis']['overall_bias']:+.1%} ({results['bias_analysis']['bias_interpretation']})")
    logger.info(f"   Accuracy: {results['basic_metrics']['accuracy']:.3f}")
    logger.info(f"   F1 (Macro): {results['basic_metrics']['f1_macro']:.3f}")
    logger.info(f"   Brier Score: {results['basic_metrics']['brier_score']:.4f}")
    logger.info(f"   ECE: {results['calibration_analysis']['ece']:.4f}")
    
    # Най-проблемни bins
    worst_bins = sorted(results['calibration_analysis']['bins'], 
                       key=lambda x: x['calibration_error'], reverse=True)[:3]
    logger.info("\n📊 WORST CALIBRATED BINS:")
    for bin_data in worst_bins:
        logger.info(f"   {bin_data['bin_range']}: Error {bin_data['calibration_error']:.1%} ({bin_data['count']} matches)")
    
    logger.info("\n✅ BTTS диагностика завършена успешно!")
    
    return results


if __name__ == "__main__":
    diagnose_btts_from_predictions()
