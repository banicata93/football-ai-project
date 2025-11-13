#!/usr/bin/env python3
"""
BTTS Model Comprehensive Diagnosis
Анализира текущия BTTS модел за accuracy, calibration, bias и league performance
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, log_loss, classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve

from core.utils import setup_logging, save_json
from core.ml_utils import prepare_features, get_feature_columns


class BTTSDiagnostics:
    """Comprehensive BTTS model diagnostics"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.model = None
        self.calibrator = None
        self.feature_cols = None
        
    def load_btts_model(self):
        """Зарежда BTTS модела и калибратора"""
        try:
            # Зарежда основния модел
            model_path = 'models/model_btts_v1/btts_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.logger.info(f"✓ BTTS модел зареден от {model_path}")
            else:
                self.logger.error(f"❌ BTTS модел не е намерен: {model_path}")
                return False
            
            # Зарежда feature columns
            feature_path = 'models/model_btts_v1/feature_columns.json'
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    feature_data = json.load(f)
                    # Проверява дали е nested структура
                    if isinstance(feature_data, dict) and 'features' in feature_data:
                        self.feature_cols = feature_data['features']
                    else:
                        self.feature_cols = feature_data
                self.logger.info(f"✓ Feature columns заредени: {len(self.feature_cols)} features")
            else:
                self.feature_cols = get_feature_columns()
                self.logger.warning("⚠️ Използвам default feature columns")
            
            # Зарежда калибратор ако съществува
            calibrator_path = 'models/model_btts_v1/calibrator.pkl'
            if os.path.exists(calibrator_path):
                self.calibrator = joblib.load(calibrator_path)
                self.logger.info("✓ BTTS калибратор зареден")
            else:
                self.logger.warning("⚠️ BTTS калибратор не е намерен")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Грешка при зареждане на BTTS модел: {e}")
            return False
    
    def calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Изчислява Expected Calibration Error (ECE)
        
        Args:
            y_true: Истински labels
            y_prob: Предсказани вероятности
            n_bins: Брой bins за калибрация
            
        Returns:
            ECE стойност
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Намира samples в този bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def analyze_calibration_bins(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Анализира калибрацията по bins
        
        Args:
            y_true: Истински labels
            y_prob: Предсказани вероятности
            n_bins: Брой bins
            
        Returns:
            Dictionary с анализ по bins
        """
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
            'ece': self.calculate_ece(y_true, y_prob, n_bins),
            'total_samples': len(y_true)
        }
    
    def analyze_league_performance(self, df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Анализира BTTS performance по лиги
        
        Args:
            df: Dataset с league информация
            y_true: Истински BTTS labels
            y_prob: Предсказани BTTS вероятности
            
        Returns:
            Dictionary с league анализ
        """
        league_analysis = {}
        
        if 'league' not in df.columns:
            self.logger.warning("⚠️ Няма league колона за анализ")
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
            ece = self.calculate_ece(league_y_true, league_y_prob)
            
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
    
    def comprehensive_evaluation(self, df: pd.DataFrame) -> Dict:
        """
        Пълна оценка на BTTS модела
        
        Args:
            df: Test/validation dataset
            
        Returns:
            Comprehensive evaluation results
        """
        if self.model is None:
            raise ValueError("BTTS модел не е зареден")
        
        self.logger.info("🔍 Започване на comprehensive BTTS evaluation...")
        
        # Подготвя features
        X = prepare_features(df, self.feature_cols)
        y_true = df['btts'].values
        
        # Проверява дали моделът е вече калибриран
        from sklearn.calibration import CalibratedClassifierCV
        
        if isinstance(self.model, CalibratedClassifierCV):
            # Моделът е вече калибриран
            y_pred = self.model.predict(X)
            y_prob_calibrated = self.model.predict_proba(X)[:, 1]
            
            # За raw predictions използваме base estimator
            base_estimator = self.model.estimator
            y_prob_raw = base_estimator.predict_proba(X)[:, 1]
        else:
            # Обикновен модел
            y_pred = self.model.predict(X)
            y_prob_raw = self.model.predict_proba(X)[:, 1]
            
            # Калибрирани predictions ако има калибратор
            if self.calibrator is not None:
                y_prob_calibrated = self.calibrator.predict_proba(X)[:, 1]
            else:
                y_prob_calibrated = y_prob_raw
        
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
            'brier_score_raw': float(brier_score_loss(y_true, y_prob_raw)),
            'brier_score_calibrated': float(brier_score_loss(y_true, y_prob_calibrated)),
            'log_loss_raw': float(log_loss(y_true, y_prob_raw)),
            'log_loss_calibrated': float(log_loss(y_true, y_prob_calibrated))
        }
        
        # Калибрационен анализ
        calibration_raw = self.analyze_calibration_bins(y_true, y_prob_raw)
        calibration_calibrated = self.analyze_calibration_bins(y_true, y_prob_calibrated)
        
        # League анализ
        league_analysis = self.analyze_league_performance(df, y_true, y_prob_calibrated)
        
        # Bias анализ
        btts_base_rate = y_true.mean()
        avg_predicted_prob = y_prob_calibrated.mean()
        overall_bias = avg_predicted_prob - btts_base_rate
        
        # Threshold анализ
        threshold_analysis = {}
        for threshold in [0.45, 0.5, 0.55, 0.6]:
            y_pred_thresh = (y_prob_calibrated > threshold).astype(int)
            threshold_analysis[f'threshold_{threshold}'] = {
                'accuracy': float(accuracy_score(y_true, y_pred_thresh)),
                'f1_macro': float(f1_score(y_true, y_pred_thresh, average='macro')),
                'precision_yes': float(precision_score(y_true, y_pred_thresh, pos_label=1)),
                'recall_yes': float(recall_score(y_true, y_pred_thresh, pos_label=1))
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_matches': int(len(df)),
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
            'calibration_analysis': {
                'raw_model': calibration_raw,
                'calibrated_model': calibration_calibrated,
                'calibration_improvement': float(calibration_raw['ece'] - calibration_calibrated['ece'])
            },
            'league_analysis': league_analysis,
            'threshold_analysis': threshold_analysis,
            'confusion_matrix': {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
        }
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = 'reports/btts_diagnosis_report.json'):
        """
        Генерира детайлен отчет
        
        Args:
            results: Резултати от evaluation
            output_path: Път за запазване на отчета
        """
        # Запазва JSON отчета
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_json(results, output_path)
        
        # Генерира markdown summary
        markdown_path = output_path.replace('.json', '.md')
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# BTTS Model Diagnosis Report\n\n")
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
            f.write(f"- **Brier Score (Raw):** {metrics['brier_score_raw']:.4f}\n")
            f.write(f"- **Brier Score (Calibrated):** {metrics['brier_score_calibrated']:.4f}\n\n")
            
            # Bias analysis
            bias = results['bias_analysis']
            f.write("## Bias Analysis\n")
            f.write(f"- **Base Rate:** {bias['btts_base_rate']:.1%}\n")
            f.write(f"- **Avg Predicted:** {bias['avg_predicted_prob']:.1%}\n")
            f.write(f"- **Bias:** {bias['overall_bias']:+.1%} ({bias['bias_interpretation']})\n\n")
            
            # Calibration
            cal = results['calibration_analysis']
            f.write("## Calibration Analysis\n")
            f.write(f"- **ECE (Raw):** {cal['raw_model']['ece']:.4f}\n")
            f.write(f"- **ECE (Calibrated):** {cal['calibrated_model']['ece']:.4f}\n")
            f.write(f"- **Improvement:** {cal['calibration_improvement']:+.4f}\n\n")
            
            # Top problematic leagues
            leagues = results['league_analysis']
            if leagues:
                f.write("## League Analysis (Top Issues)\n")
                # Сортира по bias
                sorted_leagues = sorted(leagues.items(), key=lambda x: abs(x[1]['bias']), reverse=True)
                for league, data in sorted_leagues[:5]:
                    f.write(f"- **{league}:** {data['matches']} matches, ")
                    f.write(f"Base rate: {data['btts_base_rate']:.1%}, ")
                    f.write(f"Bias: {data['bias']:+.1%}, ")
                    f.write(f"ECE: {data['ece']:.3f}\n")
                f.write("\n")
            
            # Threshold analysis
            thresh = results['threshold_analysis']
            f.write("## Threshold Optimization\n")
            for threshold, data in thresh.items():
                t = threshold.replace('threshold_', '')
                f.write(f"- **{t}:** Acc: {data['accuracy']:.3f}, F1: {data['f1_macro']:.3f}\n")
        
        self.logger.info(f"✓ Отчет запазен в {output_path}")
        self.logger.info(f"✓ Markdown summary в {markdown_path}")


def main():
    """Основна функция за BTTS диагностика"""
    logger = setup_logging()
    
    logger.info("🔍 СТАРТИРАНЕ НА BTTS MODEL DIAGNOSIS")
    logger.info("=" * 60)
    
    # Инициализира диагностиката
    diagnostics = BTTSDiagnostics()
    
    # Зарежда модела
    if not diagnostics.load_btts_model():
        logger.error("❌ Не мога да заредя BTTS модел")
        return
    
    # Зарежда test features data
    test_features_path = 'data/processed/test_features.parquet'
    if not os.path.exists(test_features_path):
        logger.error(f"❌ Test features не са намерени: {test_features_path}")
        return
    
    logger.info(f"📊 Зареждане на test features от {test_features_path}")
    test_df = pd.read_parquet(test_features_path)
    
    if 'btts' not in test_df.columns:
        logger.error("❌ Няма 'btts' колона в test data")
        return
    
    logger.info(f"✓ Test data зареден: {len(test_df)} matches")
    
    # Изпълнява диагностиката
    results = diagnostics.comprehensive_evaluation(test_df)
    
    # Генерира отчета
    diagnostics.generate_report(results)
    
    # Показва key findings
    logger.info("\n🎯 KEY FINDINGS:")
    bias = results['bias_analysis']
    logger.info(f"   Bias: {bias['overall_bias']:+.1%} ({bias['bias_interpretation']})")
    
    metrics = results['basic_metrics']
    logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"   F1 (Macro): {metrics['f1_macro']:.3f}")
    logger.info(f"   Brier Score: {metrics['brier_score_calibrated']:.4f}")
    
    cal = results['calibration_analysis']
    logger.info(f"   ECE: {cal['calibrated_model']['ece']:.4f}")
    
    logger.info("\n✅ BTTS диагностика завършена успешно!")


if __name__ == "__main__":
    main()
