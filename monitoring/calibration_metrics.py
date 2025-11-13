"""
Калибрационни метрики за мониторинг на точността на вероятностите
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import os


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """
    Изчислява Expected Calibration Error (ECE)
    
    Args:
        y_true: Истински labels (0/1)
        y_prob: Predicted probabilities
        bins: Брой bins за калибрация
    
    Returns:
        ECE стойност (0-1, по-ниско е по-добре)
    """
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Намери samples в този bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy в този bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Average confidence в този bin
            avg_confidence_in_bin = y_prob[in_bin].mean()
            # Добави към ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """
    Изчислява Maximum Calibration Error (MCE)
    
    Args:
        y_true: Истински labels (0/1)
        y_prob: Predicted probabilities
        bins: Брой bins за калибрация
    
    Returns:
        MCE стойност (0-1, по-ниско е по-добре)
    """
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
    
    return max_error


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> Dict:
    """
    Comprehensive калибрационна оценка
    
    Args:
        y_true: Истински labels (0/1)
        y_prob: Predicted probabilities
        bins: Брой bins за калибрация
    
    Returns:
        Dictionary с всички метрики
    """
    # Основни метрики
    ece = expected_calibration_error(y_true, y_prob, bins)
    mce = maximum_calibration_error(y_true, y_prob, bins)
    brier = brier_score_loss(y_true, y_prob)
    
    # Log loss (с clipping за стабилност)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logloss = log_loss(y_true, y_prob_clipped)
    
    # Accuracy
    y_pred = (y_prob > 0.5).astype(int)
    accuracy = (y_true == y_pred).mean()
    
    # Reliability curve data
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=bins, strategy='uniform'
    )
    
    # Confidence distribution
    confidence_stats = {
        'mean_confidence': float(y_prob.mean()),
        'std_confidence': float(y_prob.std()),
        'min_confidence': float(y_prob.min()),
        'max_confidence': float(y_prob.max()),
        'median_confidence': float(np.median(y_prob))
    }
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': float(brier),
        'log_loss': float(logloss),
        'accuracy': float(accuracy),
        'n_samples': len(y_true),
        'reliability_curve': {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        },
        'confidence_stats': confidence_stats,
        'timestamp': datetime.now().isoformat()
    }


def evaluate_multiclass_calibration(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> Dict:
    """
    Калибрационна оценка за multiclass (1X2)
    
    Args:
        y_true: Истински labels (0, 1, 2)
        y_prob: Predicted probabilities matrix (N, 3)
        bins: Брой bins за калибрация
    
    Returns:
        Dictionary с метрики за всеки клас
    """
    n_classes = y_prob.shape[1]
    results = {}
    
    # Overall метрики
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    y_prob_normalized = y_prob_clipped / y_prob_clipped.sum(axis=1, keepdims=True)
    
    overall_logloss = log_loss(y_true, y_prob_normalized)
    overall_brier = np.mean([((y_true == i).astype(float) - y_prob_normalized[:, i])**2 
                           for i in range(n_classes)])
    
    # Accuracy
    y_pred = np.argmax(y_prob, axis=1)
    accuracy = (y_true == y_pred).mean()
    
    results['overall'] = {
        'log_loss': float(overall_logloss),
        'brier_score': float(overall_brier),
        'accuracy': float(accuracy),
        'n_samples': len(y_true)
    }
    
    # Per-class метрики (one-vs-rest)
    class_names = ['home_win', 'draw', 'away_win']
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        class_metrics = evaluate_calibration(y_true_binary, y_prob_binary, bins)
        results[class_name] = class_metrics
    
    results['timestamp'] = datetime.now().isoformat()
    return results


class CalibrationMonitor:
    """
    Клас за мониторинг на калибрацията във времето
    """
    
    def __init__(self, history_file: str = "logs/predictions_history.parquet"):
        self.history_file = history_file
        self.reports_dir = "reports/calibration"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def log_prediction(self, prediction_data: Dict):
        """
        Логва една прогноза за бъдещо сравнение
        
        Args:
            prediction_data: Dictionary с прогнозата и мета данни
        """
        # Конвертира в DataFrame
        df_new = pd.DataFrame([prediction_data])
        
        # Добавя към историята
        if os.path.exists(self.history_file):
            df_existing = pd.read_parquet(self.history_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # Запазва
        df_combined.to_parquet(self.history_file, index=False)
    
    def load_recent_predictions(self, days: int = 30, max_samples: int = 5000) -> pd.DataFrame:
        """
        Зарежда скорошни прогнози с известни резултати
        
        Args:
            days: Брой дни назад
            max_samples: Максимален брой samples
        
        Returns:
            DataFrame с прогнози и резултати
        """
        if not os.path.exists(self.history_file):
            return pd.DataFrame()
        
        df = pd.read_parquet(self.history_file)
        
        # Филтрира по дата
        cutoff_date = datetime.now() - timedelta(days=days)
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        df_recent = df[df['prediction_date'] >= cutoff_date].copy()
        
        # Филтрира само мачове с известни резултати
        df_with_results = df_recent.dropna(subset=['actual_home_score', 'actual_away_score'])
        
        # Ограничава броя samples
        if len(df_with_results) > max_samples:
            df_with_results = df_with_results.tail(max_samples)
        
        return df_with_results
    
    def calculate_actual_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Изчислява реалните outcomes от резултатите
        
        Args:
            df: DataFrame с actual_home_score и actual_away_score
        
        Returns:
            DataFrame с добавени actual outcomes
        """
        df = df.copy()
        
        # 1X2 outcomes
        conditions_1x2 = [
            df['actual_home_score'] > df['actual_away_score'],  # Home win
            df['actual_home_score'] == df['actual_away_score'], # Draw
            df['actual_home_score'] < df['actual_away_score']   # Away win
        ]
        df['actual_1x2'] = np.select(conditions_1x2, [0, 1, 2])
        
        # Over/Under 2.5
        total_goals = df['actual_home_score'] + df['actual_away_score']
        df['actual_ou25'] = (total_goals > 2.5).astype(int)
        
        # BTTS
        df['actual_btts'] = ((df['actual_home_score'] > 0) & 
                            (df['actual_away_score'] > 0)).astype(int)
        
        return df
    
    def generate_calibration_report(self, days: int = 30) -> Dict:
        """
        Генерира comprehensive калибрационен отчет
        
        Args:
            days: Брой дни за анализ
        
        Returns:
            Dictionary с отчета
        """
        df = self.load_recent_predictions(days)
        
        if len(df) == 0:
            return {
                'error': 'No predictions with results found',
                'timestamp': datetime.now().isoformat()
            }
        
        df = self.calculate_actual_outcomes(df)
        
        report = {
            'period': f'Last {days} days',
            'n_matches': len(df),
            'date_range': {
                'start': df['prediction_date'].min().isoformat(),
                'end': df['prediction_date'].max().isoformat()
            }
        }
        
        # 1X2 Calibration
        if 'pred_1x2_probs' in df.columns:
            y_true_1x2 = df['actual_1x2'].values
            y_prob_1x2 = np.array(df['pred_1x2_probs'].tolist())
            report['1x2'] = evaluate_multiclass_calibration(y_true_1x2, y_prob_1x2)
        
        # OU2.5 Calibration
        if 'pred_ou25_prob' in df.columns:
            y_true_ou25 = df['actual_ou25'].values
            y_prob_ou25 = df['pred_ou25_prob'].values
            report['ou25'] = evaluate_calibration(y_true_ou25, y_prob_ou25)
        
        # BTTS Calibration
        if 'pred_btts_prob' in df.columns:
            y_true_btts = df['actual_btts'].values
            y_prob_btts = df['pred_btts_prob'].values
            report['btts'] = evaluate_calibration(y_true_btts, y_prob_btts)
        
        report['timestamp'] = datetime.now().isoformat()
        
        # Запазва отчета
        report_file = f"{self.reports_dir}/calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def plot_reliability_curves(self, report: Dict, save_path: Optional[str] = None):
        """
        Създава reliability plots
        
        Args:
            report: Calibration report
            save_path: Път за запазване на plot-а
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        markets = ['ou25', 'btts']
        market_names = ['Over/Under 2.5', 'BTTS']
        
        for i, (market, name) in enumerate(zip(markets, market_names)):
            if market in report:
                reliability_data = report[market]['reliability_curve']
                
                axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[i].plot(
                    reliability_data['mean_predicted_value'],
                    reliability_data['fraction_of_positives'],
                    'o-', label=f'{name} (ECE: {report[market]["ece"]:.3f})'
                )
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].set_title(f'{name} Reliability Curve')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 1X2 - показва само home win за простота
        if '1x2' in report and 'home_win' in report['1x2']:
            reliability_data = report['1x2']['home_win']['reliability_curve']
            
            axes[2].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            axes[2].plot(
                reliability_data['mean_predicted_value'],
                reliability_data['fraction_of_positives'],
                'o-', label=f'Home Win (ECE: {report["1x2"]["home_win"]["ece"]:.3f})'
            )
            axes[2].set_xlabel('Mean Predicted Probability')
            axes[2].set_ylabel('Fraction of Positives')
            axes[2].set_title('1X2 Home Win Reliability Curve')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Тестване на функциите
    print("🧪 Тестване на калибрационни метрики...")
    
    # Симулирани данни
    np.random.seed(42)
    n_samples = 1000
    
    # Perfect calibration
    y_true_perfect = np.random.binomial(1, 0.3, n_samples)
    y_prob_perfect = np.full(n_samples, 0.3)
    
    # Poor calibration (overconfident)
    y_true_poor = np.random.binomial(1, 0.3, n_samples)
    y_prob_poor = np.random.beta(8, 2, n_samples)  # Biased towards high probabilities
    
    print("Perfect calibration:")
    perfect_metrics = evaluate_calibration(y_true_perfect, y_prob_perfect)
    print(f"  ECE: {perfect_metrics['ece']:.4f}")
    print(f"  Brier: {perfect_metrics['brier_score']:.4f}")
    
    print("\nPoor calibration:")
    poor_metrics = evaluate_calibration(y_true_poor, y_prob_poor)
    print(f"  ECE: {poor_metrics['ece']:.4f}")
    print(f"  Brier: {poor_metrics['brier_score']:.4f}")
    
    print("\n✅ Калибрационни метрики работят!")
