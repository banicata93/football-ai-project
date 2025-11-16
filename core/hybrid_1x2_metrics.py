#!/usr/bin/env python3
"""
Hybrid 1X2 Metrics Module

Utility functions for evaluating and comparing 1X2 prediction models.
Provides comprehensive metrics for ML, Scoreline, and Hybrid approaches.

ADDITIVE - does not modify existing evaluation code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class Hybrid1X2Evaluator:
    """
    Comprehensive evaluator for 1X2 prediction models
    
    Provides metrics, calibration analysis, and visualization tools
    for comparing different 1X2 prediction approaches.
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_cache = {}
        
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate 1X2 predictions with comprehensive metrics
        
        Args:
            y_true: True outcomes (0=Home, 1=Draw, 2=Away)
            y_pred: Predicted probabilities [N, 3]
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Ensure proper shapes
            y_true = np.array(y_true).astype(int)
            y_pred = np.array(y_pred)
            
            if y_pred.ndim == 1:
                # Convert single predictions to probability format
                y_pred_probs = np.zeros((len(y_pred), 3))
                for i, pred in enumerate(y_pred):
                    y_pred_probs[i, int(pred)] = 1.0
                y_pred = y_pred_probs
            
            # Basic metrics
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_true, y_pred_classes)
            
            # Probabilistic metrics
            try:
                logloss = log_loss(y_true, y_pred, labels=[0, 1, 2])
            except:
                logloss = float('inf')
            
            # Brier score (multi-class)
            brier_scores = []
            for class_idx in range(3):
                y_true_binary = (y_true == class_idx).astype(int)
                y_pred_binary = y_pred[:, class_idx]
                try:
                    brier = brier_score_loss(y_true_binary, y_pred_binary)
                    brier_scores.append(brier)
                except:
                    brier_scores.append(1.0)
            
            avg_brier = np.mean(brier_scores)
            
            # Class-specific accuracies
            home_mask = y_true == 0
            draw_mask = y_true == 1
            away_mask = y_true == 2
            
            home_accuracy = accuracy_score(y_true[home_mask], y_pred_classes[home_mask]) if np.sum(home_mask) > 0 else 0.0
            draw_accuracy = accuracy_score(y_true[draw_mask], y_pred_classes[draw_mask]) if np.sum(draw_mask) > 0 else 0.0
            away_accuracy = accuracy_score(y_true[away_mask], y_pred_classes[away_mask]) if np.sum(away_mask) > 0 else 0.0
            
            # Calibration metrics
            calibration_metrics = self._calculate_calibration_metrics(y_true, y_pred)
            
            # Confidence and uncertainty metrics
            confidence_metrics = self._calculate_confidence_metrics(y_pred)
            
            # Bias metrics
            bias_metrics = self._calculate_bias_metrics(y_true, y_pred)
            
            # Combine all metrics
            metrics = {
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'brier_score': float(avg_brier),
                'home_accuracy': float(home_accuracy),
                'draw_accuracy': float(draw_accuracy),
                'away_accuracy': float(away_accuracy),
                'home_precision': self._calculate_precision(y_true, y_pred_classes, 0),
                'draw_precision': self._calculate_precision(y_true, y_pred_classes, 1),
                'away_precision': self._calculate_precision(y_true, y_pred_classes, 2),
                'home_recall': self._calculate_recall(y_true, y_pred_classes, 0),
                'draw_recall': self._calculate_recall(y_true, y_pred_classes, 1),
                'away_recall': self._calculate_recall(y_true, y_pred_classes, 2),
                **calibration_metrics,
                **confidence_metrics,
                **bias_metrics
            }
            
            # Cache results
            self.metrics_cache[model_name] = metrics
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return self._get_default_metrics()
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics"""
        try:
            calibration_errors = []
            
            for class_idx in range(3):
                y_true_binary = (y_true == class_idx).astype(int)
                y_pred_binary = y_pred[:, class_idx]
                
                try:
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true_binary, y_pred_binary, n_bins=10, strategy='uniform'
                    )
                    
                    # Expected Calibration Error (ECE)
                    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                    calibration_errors.append(ece)
                except:
                    calibration_errors.append(0.5)
            
            return {
                'calibration_error': float(np.mean(calibration_errors)),
                'home_calibration_error': float(calibration_errors[0]),
                'draw_calibration_error': float(calibration_errors[1]),
                'away_calibration_error': float(calibration_errors[2])
            }
            
        except Exception as e:
            return {
                'calibration_error': 0.5,
                'home_calibration_error': 0.5,
                'draw_calibration_error': 0.5,
                'away_calibration_error': 0.5
            }
    
    def _calculate_confidence_metrics(self, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate confidence and uncertainty metrics"""
        try:
            # Average confidence (max probability)
            max_probs = np.max(y_pred, axis=1)
            avg_confidence = np.mean(max_probs)
            
            # Entropy (uncertainty)
            entropies = -np.sum(y_pred * np.log(np.clip(y_pred, 1e-10, 1.0)), axis=1)
            avg_entropy = np.mean(entropies)
            max_entropy = np.log(3)  # Maximum entropy for 3 classes
            normalized_entropy = avg_entropy / max_entropy
            
            # Prediction strength distribution
            confidence_std = np.std(max_probs)
            
            return {
                'avg_confidence': float(avg_confidence),
                'avg_entropy': float(avg_entropy),
                'normalized_entropy': float(normalized_entropy),
                'confidence_std': float(confidence_std)
            }
            
        except Exception as e:
            return {
                'avg_confidence': 0.33,
                'avg_entropy': 1.0,
                'normalized_entropy': 0.91,
                'confidence_std': 0.1
            }
    
    def _calculate_bias_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction bias metrics"""
        try:
            # Class distribution bias
            true_dist = np.bincount(y_true, minlength=3) / len(y_true)
            pred_dist = np.mean(y_pred, axis=0)
            
            # KL divergence for distribution comparison
            kl_div = np.sum(true_dist * np.log(np.clip(true_dist / pred_dist, 1e-10, 1e10)))
            
            # Individual class biases
            home_bias = pred_dist[0] - true_dist[0]
            draw_bias = pred_dist[1] - true_dist[1]
            away_bias = pred_dist[2] - true_dist[2]
            
            return {
                'distribution_kl_divergence': float(kl_div),
                'home_bias': float(home_bias),
                'draw_bias': float(draw_bias),
                'away_bias': float(away_bias),
                'total_bias': float(np.sum(np.abs([home_bias, draw_bias, away_bias])))
            }
            
        except Exception as e:
            return {
                'distribution_kl_divergence': 0.0,
                'home_bias': 0.0,
                'draw_bias': 0.0,
                'away_bias': 0.0,
                'total_bias': 0.0
            }
    
    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:
        """Calculate precision for specific class"""
        try:
            tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
            fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:
        """Calculate recall for specific class"""
        try:
            tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
            fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        except:
            return 0.0
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metrics for error cases"""
        return {
            'accuracy': 0.33,
            'log_loss': 1.0,
            'brier_score': 0.5,
            'home_accuracy': 0.33,
            'draw_accuracy': 0.33,
            'away_accuracy': 0.33,
            'home_precision': 0.33,
            'draw_precision': 0.33,
            'away_precision': 0.33,
            'home_recall': 0.33,
            'draw_recall': 0.33,
            'away_recall': 0.33,
            'calibration_error': 0.5,
            'home_calibration_error': 0.5,
            'draw_calibration_error': 0.5,
            'away_calibration_error': 0.5,
            'avg_confidence': 0.33,
            'avg_entropy': 1.0,
            'normalized_entropy': 0.91,
            'confidence_std': 0.1,
            'distribution_kl_divergence': 0.0,
            'home_bias': 0.0,
            'draw_bias': 0.0,
            'away_bias': 0.0,
            'total_bias': 0.0
        }
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Compare multiple models and determine best performer
        
        Args:
            results: Dictionary of model_name -> metrics
            
        Returns:
            Comparison analysis
        """
        try:
            if not results:
                return {'best_model': 'none', 'comparison': {}}
            
            # Key metrics for ranking
            key_metrics = ['accuracy', 'draw_accuracy', 'log_loss', 'brier_score', 'calibration_error']
            
            # Calculate composite scores (lower is better for loss metrics)
            composite_scores = {}
            for model_name, metrics in results.items():
                score = 0.0
                score += metrics.get('accuracy', 0.0) * 0.3  # Higher is better
                score += metrics.get('draw_accuracy', 0.0) * 0.2  # Higher is better
                score -= metrics.get('log_loss', 1.0) * 0.2  # Lower is better
                score -= metrics.get('brier_score', 0.5) * 0.15  # Lower is better
                score -= metrics.get('calibration_error', 0.5) * 0.15  # Lower is better
                composite_scores[model_name] = score
            
            # Find best model
            best_model = max(composite_scores, key=composite_scores.get)
            
            # Create comparison table
            comparison_table = {}
            for metric in key_metrics:
                comparison_table[metric] = {}
                for model_name in results.keys():
                    comparison_table[metric][model_name] = results[model_name].get(metric, 0.0)
            
            # Calculate improvements
            improvements = {}
            if len(results) > 1:
                baseline_model = list(results.keys())[0]  # Use first model as baseline
                baseline_metrics = results[baseline_model]
                
                for model_name, metrics in results.items():
                    if model_name != baseline_model:
                        improvements[model_name] = {}
                        for metric in key_metrics:
                            baseline_val = baseline_metrics.get(metric, 0.0)
                            current_val = metrics.get(metric, 0.0)
                            
                            if metric in ['log_loss', 'brier_score', 'calibration_error']:
                                # Lower is better
                                improvement = (baseline_val - current_val) / baseline_val * 100 if baseline_val > 0 else 0.0
                            else:
                                # Higher is better
                                improvement = (current_val - baseline_val) / baseline_val * 100 if baseline_val > 0 else 0.0
                            
                            improvements[model_name][metric] = improvement
            
            return {
                'best_model': best_model,
                'composite_scores': composite_scores,
                'comparison_table': comparison_table,
                'improvements': improvements,
                'summary': self._generate_comparison_summary(results, best_model)
            }
            
        except Exception as e:
            print(f"❌ Error comparing models: {e}")
            return {'best_model': 'unknown', 'comparison': {}}
    
    def _generate_comparison_summary(self, results: Dict[str, Dict[str, float]], 
                                   best_model: str) -> Dict[str, str]:
        """Generate human-readable comparison summary"""
        try:
            summary = {}
            
            # Overall winner
            summary['winner'] = f"{best_model} performs best overall"
            
            # Accuracy comparison
            accuracies = {name: metrics.get('accuracy', 0.0) for name, metrics in results.items()}
            best_accuracy_model = max(accuracies, key=accuracies.get)
            summary['accuracy'] = f"{best_accuracy_model} has highest accuracy ({accuracies[best_accuracy_model]:.3f})"
            
            # Draw accuracy comparison
            draw_accuracies = {name: metrics.get('draw_accuracy', 0.0) for name, metrics in results.items()}
            best_draw_model = max(draw_accuracies, key=draw_accuracies.get)
            summary['draw_detection'] = f"{best_draw_model} best at detecting draws ({draw_accuracies[best_draw_model]:.3f})"
            
            # Calibration comparison
            calibrations = {name: metrics.get('calibration_error', 1.0) for name, metrics in results.items()}
            best_calibration_model = min(calibrations, key=calibrations.get)
            summary['calibration'] = f"{best_calibration_model} has best calibration (error: {calibrations[best_calibration_model]:.3f})"
            
            return summary
            
        except Exception as e:
            return {'error': f"Could not generate summary: {e}"}
    
    def plot_calibration_curves(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], 
                              save_path: str = None) -> None:
        """
        Plot calibration curves for multiple models
        
        Args:
            y_true: True outcomes
            predictions: Dictionary of model_name -> predictions
            save_path: Path to save plot (optional)
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            class_names = ['Home', 'Draw', 'Away']
            
            for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
                y_true_binary = (y_true == class_idx).astype(int)
                
                for model_name, y_pred in predictions.items():
                    if y_pred.ndim == 1:
                        continue  # Skip non-probability predictions
                    
                    y_pred_binary = y_pred[:, class_idx]
                    
                    try:
                        fraction_of_positives, mean_predicted_value = calibration_curve(
                            y_true_binary, y_pred_binary, n_bins=10, strategy='uniform'
                        )
                        
                        ax.plot(mean_predicted_value, fraction_of_positives, 
                               marker='o', label=model_name, linewidth=2)
                    except:
                        continue
                
                # Perfect calibration line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Fraction of Positives')
                ax.set_title(f'{class_name} Calibration')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Calibration curves saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error plotting calibration curves: {e}")
    
    def plot_distribution_comparison(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                   save_path: str = None) -> None:
        """
        Plot prediction distribution comparison
        
        Args:
            y_true: True outcomes
            predictions: Dictionary of model_name -> predictions
            save_path: Path to save plot (optional)
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # True distribution
            true_dist = np.bincount(y_true, minlength=3) / len(y_true)
            class_names = ['Home', 'Draw', 'Away']
            
            ax1.bar(class_names, true_dist, alpha=0.7, label='True Distribution', color='gray')
            
            # Predicted distributions
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (model_name, y_pred) in enumerate(predictions.items()):
                if y_pred.ndim == 2:
                    pred_dist = np.mean(y_pred, axis=0)
                else:
                    pred_dist = np.bincount(y_pred.astype(int), minlength=3) / len(y_pred)
                
                ax1.bar(class_names, pred_dist, alpha=0.6, 
                       label=f'{model_name} Predicted', 
                       color=colors[i % len(colors)])
            
            ax1.set_ylabel('Probability')
            ax1.set_title('Class Distribution Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bias comparison
            biases = {}
            for model_name, y_pred in predictions.items():
                if y_pred.ndim == 2:
                    pred_dist = np.mean(y_pred, axis=0)
                else:
                    pred_dist = np.bincount(y_pred.astype(int), minlength=3) / len(y_pred)
                
                bias = pred_dist - true_dist
                biases[model_name] = bias
            
            x = np.arange(len(class_names))
            width = 0.8 / len(biases)
            
            for i, (model_name, bias) in enumerate(biases.items()):
                ax2.bar(x + i * width, bias, width, label=model_name, 
                       color=colors[i % len(colors)], alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Bias (Predicted - True)')
            ax2.set_title('Prediction Bias by Class')
            ax2.set_xticks(x + width * (len(biases) - 1) / 2)
            ax2.set_xticklabels(class_names)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Distribution comparison saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error plotting distribution comparison: {e}")
    
    def plot_draw_accuracy_comparison(self, results: Dict[str, Dict[str, float]], 
                                    save_path: str = None) -> None:
        """
        Plot draw accuracy comparison across models
        
        Args:
            results: Dictionary of model_name -> metrics
            save_path: Path to save plot (optional)
        """
        try:
            models = list(results.keys())
            draw_accuracies = [results[model].get('draw_accuracy', 0.0) for model in models]
            overall_accuracies = [results[model].get('accuracy', 0.0) for model in models]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, overall_accuracies, width, label='Overall Accuracy', alpha=0.7)
            bars2 = ax.bar(x + width/2, draw_accuracies, width, label='Draw Accuracy', alpha=0.7)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            ax.set_title('Overall vs Draw Accuracy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Draw accuracy comparison saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error plotting draw accuracy comparison: {e}")


def main():
    """
    Example usage and testing
    """
    print("🎯 Testing Hybrid 1X2 Metrics")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = Hybrid1X2Evaluator()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True outcomes (0=Home, 1=Draw, 2=Away)
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.45, 0.27, 0.28])
    
    # Sample predictions for different models
    # ML model (slightly overconfident)
    ml_probs = np.random.dirichlet([2, 1, 1.5], n_samples)
    
    # Scoreline model (more balanced)
    scoreline_probs = np.random.dirichlet([1.8, 1.3, 1.4], n_samples)
    
    # Hybrid model (best of both)
    hybrid_probs = 0.6 * ml_probs + 0.4 * scoreline_probs
    
    # Evaluate each model
    print("\n📊 Evaluating Models:")
    ml_metrics = evaluator.evaluate_predictions(y_true, ml_probs, "ML")
    scoreline_metrics = evaluator.evaluate_predictions(y_true, scoreline_probs, "Scoreline")
    hybrid_metrics = evaluator.evaluate_predictions(y_true, hybrid_probs, "Hybrid")
    
    # Compare models
    results = {
        'ML': ml_metrics,
        'Scoreline': scoreline_metrics,
        'Hybrid': hybrid_metrics
    }
    
    comparison = evaluator.compare_models(results)
    
    print(f"\n🏆 Best Model: {comparison['best_model']}")
    print(f"\n📈 Key Metrics Comparison:")
    for metric in ['accuracy', 'draw_accuracy', 'log_loss', 'calibration_error']:
        print(f"   {metric}:")
        for model, metrics in results.items():
            print(f"     {model}: {metrics.get(metric, 0.0):.3f}")
    
    print(f"\n📊 Summary:")
    for key, value in comparison['summary'].items():
        print(f"   {key}: {value}")
    
    print("\n✅ Hybrid 1X2 Metrics test completed!")


if __name__ == "__main__":
    main()
