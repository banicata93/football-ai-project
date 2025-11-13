#!/usr/bin/env python3
"""
Performance мониторинг на per-league модели
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from core.league_utils import LEAGUE_ID_TO_SLUG, get_league_display_name
from core.utils import setup_logging


class PerLeaguePerformanceMonitor:
    """
    Performance мониторинг на per-league модели
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
    def load_training_metrics(self) -> Dict:
        """
        Зарежда метрики от тренировката
        
        Returns:
            Метрики по лиги
        """
        try:
            with open("logs/model_reports/ou25_per_league_summary.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Няма training метрики файл")
            return {}
    
    def analyze_model_performance(self, training_metrics: Dict) -> Dict:
        """
        Анализира performance на моделите
        
        Args:
            training_metrics: Метрики от тренировката
        
        Returns:
            Анализ на performance
        """
        self.logger.info("📊 Анализиране на model performance...")
        
        analysis = {
            'league_rankings': {},
            'performance_tiers': {'excellent': [], 'good': [], 'average': [], 'poor': []},
            'metrics_summary': {},
            'recommendations': []
        }
        
        if not training_metrics.get('metrics_by_league'):
            return analysis
        
        # Анализира всяка лига
        league_scores = {}
        
        for league_slug, metrics in training_metrics['metrics_by_league'].items():
            if not metrics:
                continue
            league_name = get_league_display_name(league_slug)
            
            # Composite score (по-ниски log_loss и brier_score са по-добри)
            accuracy = metrics.get('accuracy', 0)
            log_loss = metrics.get('log_loss', 1.0)
            brier_score = metrics.get('brier_score', 0.5)
            
            # Нормализиран score (0-100)
            score = (accuracy * 40 +  # 40% accuracy
                    (1 - min(log_loss, 2.0) / 2.0) * 35 +  # 35% log loss (inverted)
                    (1 - min(brier_score, 1.0)) * 25) * 100  # 25% brier score (inverted)
            
            league_scores[league_slug] = {
                'name': league_name,
                'score': score,
                'accuracy': accuracy,
                'log_loss': log_loss,
                'brier_score': brier_score,
                'matches': metrics.get('matches', 0),
                'calibrated': True  # Всички per-league модели са калибрирани
            }
        
        # Сортира по score
        sorted_leagues = sorted(league_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        analysis['league_rankings'] = dict(sorted_leagues)
        
        # Performance tiers
        for league_slug, data in sorted_leagues:
            if data['score'] >= 80:
                analysis['performance_tiers']['excellent'].append(league_slug)
            elif data['score'] >= 70:
                analysis['performance_tiers']['good'].append(league_slug)
            elif data['score'] >= 60:
                analysis['performance_tiers']['average'].append(league_slug)
            else:
                analysis['performance_tiers']['poor'].append(league_slug)
        
        # Summary statistics
        if league_scores:
            scores = [data['score'] for data in league_scores.values()]
            accuracies = [data['accuracy'] for data in league_scores.values()]
            log_losses = [data['log_loss'] for data in league_scores.values()]
            brier_scores = [data['brier_score'] for data in league_scores.values()]
            
            analysis['metrics_summary'] = {
                'avg_score': np.mean(scores),
                'median_score': np.median(scores),
                'std_score': np.std(scores),
                'avg_accuracy': np.mean(accuracies),
                'avg_log_loss': np.mean(log_losses),
                'avg_brier_score': np.mean(brier_scores),
                'total_leagues': len(league_scores)
            }
        
        # Препоръки
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """
        Генерира препоръки за подобрение
        
        Args:
            analysis: Анализ на performance
        
        Returns:
            Списък с препоръки
        """
        recommendations = []
        
        poor_leagues = analysis['performance_tiers']['poor']
        if poor_leagues:
            recommendations.append(
                f"🔴 {len(poor_leagues)} лиги с лош performance: "
                f"{', '.join([get_league_display_name(l) for l in poor_leagues])}. "
                f"Препоръчва се повече данни или feature engineering."
            )
        
        excellent_leagues = analysis['performance_tiers']['excellent']
        if excellent_leagues:
            recommendations.append(
                f"🟢 {len(excellent_leagues)} лиги с отличен performance: "
                f"{', '.join([get_league_display_name(l) for l in excellent_leagues])}. "
                f"Моделите са готови за production."
            )
        
        summary = analysis.get('metrics_summary', {})
        avg_accuracy = summary.get('avg_accuracy', 0)
        
        if avg_accuracy < 0.6:
            recommendations.append(
                f"⚠️ Средният accuracy ({avg_accuracy:.3f}) е под 60%. "
                f"Препоръчва се преразглеждане на feature selection."
            )
        elif avg_accuracy > 0.8:
            recommendations.append(
                f"✅ Отличен среден accuracy ({avg_accuracy:.3f}). "
                f"Моделите работят много добре."
            )
        
        return recommendations
    
    def generate_performance_report(self, analysis: Dict, output_file: str = None) -> str:
        """
        Генерира performance отчет
        
        Args:
            analysis: Анализ на performance
            output_file: Файл за запазване
        
        Returns:
            Текстов отчет
        """
        report_lines = [
            "📊 PERFORMANCE МОНИТОРИНГ: PER-LEAGUE OU2.5 МОДЕЛИ",
            "=" * 70,
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary
        summary = analysis.get('metrics_summary', {})
        if summary:
            report_lines.extend([
                "📈 ОБЩ ПРЕГЛЕД:",
                f"   Общо лиги: {summary['total_leagues']}",
                f"   Среден score: {summary['avg_score']:.1f}/100",
                f"   Медиан score: {summary['median_score']:.1f}/100",
                f"   Стандартно отклонение: {summary['std_score']:.1f}",
                f"   Среден accuracy: {summary['avg_accuracy']:.3f}",
                f"   Среден log loss: {summary['avg_log_loss']:.3f}",
                f"   Среден Brier score: {summary['avg_brier_score']:.3f}",
                ""
            ])
        
        # Performance tiers
        tiers = analysis.get('performance_tiers', {})
        if any(tiers.values()):
            report_lines.extend([
                "🏆 PERFORMANCE КАТЕГОРИИ:",
                ""
            ])
            
            tier_info = [
                ("🟢 Отлични (80-100)", tiers['excellent']),
                ("🟡 Добри (70-79)", tiers['good']),
                ("🟠 Средни (60-69)", tiers['average']),
                ("🔴 Слаби (<60)", tiers['poor'])
            ]
            
            for tier_name, leagues in tier_info:
                if leagues:
                    league_names = [get_league_display_name(l) for l in leagues]
                    report_lines.extend([
                        f"{tier_name}: {len(leagues)} лиги",
                        f"   {', '.join(league_names)}",
                        ""
                    ])
        
        # League rankings
        rankings = analysis.get('league_rankings', {})
        if rankings:
            report_lines.extend([
                "🥇 КЛАСИРАНЕ ПО PERFORMANCE:",
                ""
            ])
            
            for i, (league_slug, data) in enumerate(rankings.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
                
                report_lines.extend([
                    f"{medal} {data['name']}:",
                    f"    Score: {data['score']:.1f}/100",
                    f"    Accuracy: {data['accuracy']:.3f}",
                    f"    Log Loss: {data['log_loss']:.3f}",
                    f"    Brier Score: {data['brier_score']:.3f}",
                    f"    Мачове: {data['matches']}",
                    f"    Калибриран: {'✅' if data['calibrated'] else '❌'}",
                    ""
                ])
        
        # Препоръки
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "💡 ПРЕПОРЪКИ:",
                ""
            ])
            
            for rec in recommendations:
                report_lines.append(f"   {rec}")
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"📄 Performance отчет запазен в {output_file}")
        
        return report
    
    def create_performance_dashboard(self, analysis: Dict, output_dir: str = "reports/performance"):
        """
        Създава performance dashboard с графики
        
        Args:
            analysis: Анализ на performance
            output_dir: Директория за запазване
        """
        self.logger.info("📊 Създаване на performance dashboard...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        rankings = analysis.get('league_rankings', {})
        if not rankings:
            self.logger.warning("Няма данни за dashboard")
            return
        
        # Подготвя данни
        leagues = []
        scores = []
        accuracies = []
        log_losses = []
        brier_scores = []
        matches = []
        
        for league_slug, data in rankings.items():
            leagues.append(data['name'])
            scores.append(data['score'])
            accuracies.append(data['accuracy'])
            log_losses.append(data['log_loss'])
            brier_scores.append(data['brier_score'])
            matches.append(data['matches'])
        
        # Стил
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall Performance Score
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(leagues, scores, color=sns.color_palette("RdYlGn", len(leagues)))
        ax.set_xlabel('Performance Score')
        ax.set_title('Per-League OU2.5 Models Performance Ranking', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Добавя стойности на барове
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        ax1.bar(leagues, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Validation Accuracy by League', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Log Loss
        ax2.bar(leagues, log_losses, color='lightcoral', alpha=0.7)
        ax2.set_title('Validation Log Loss by League', fontweight='bold')
        ax2.set_ylabel('Log Loss')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Brier Score
        ax3.bar(leagues, brier_scores, color='lightgreen', alpha=0.7)
        ax3.set_title('Validation Brier Score by League', fontweight='bold')
        ax3.set_ylabel('Brier Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # Training Data Size
        ax4.bar(leagues, matches, color='gold', alpha=0.7)
        ax4.set_title('Training Data Size by League', fontweight='bold')
        ax4.set_ylabel('Number of Matches')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance Tiers Pie Chart
        tiers = analysis.get('performance_tiers', {})
        tier_counts = {k: len(v) for k, v in tiers.items() if v}
        
        if tier_counts:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            labels = []
            sizes = []
            colors = []
            
            tier_info = [
                ('excellent', 'Отлични (80-100)', '#2ecc71'),
                ('good', 'Добри (70-79)', '#f1c40f'),
                ('average', 'Средни (60-69)', '#e67e22'),
                ('poor', 'Слаби (<60)', '#e74c3c')
            ]
            
            for tier_key, tier_label, color in tier_info:
                if tier_key in tier_counts:
                    labels.append(f"{tier_label}\n({tier_counts[tier_key]} лиги)")
                    sizes.append(tier_counts[tier_key])
                    colors.append(color)
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title('Performance Distribution', fontsize=16, fontweight='bold')
            
            plt.savefig(f"{output_dir}/performance_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"📊 Dashboard запазен в {output_dir}/")


def main():
    """Главна функция за performance мониторинг"""
    logger = setup_logging()
    
    logger.info("📊 СТАРТИРАНЕ НА PERFORMANCE МОНИТОРИНГ")
    logger.info("=" * 60)
    
    try:
        # Инициализира monitor
        monitor = PerLeaguePerformanceMonitor()
        
        # Зарежда training метрики
        training_metrics = monitor.load_training_metrics()
        
        if not training_metrics:
            logger.error("❌ Няма training метрики за анализ")
            return
        
        # Анализира performance
        analysis = monitor.analyze_model_performance(training_metrics)
        
        # Генерира отчет
        report = monitor.generate_performance_report(
            analysis, 
            output_file="reports/performance_monitoring_report.txt"
        )
        
        print(report)
        
        # Създава dashboard
        monitor.create_performance_dashboard(analysis)
        
        # Запазва анализа
        analysis_file = "reports/performance_analysis.json"
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'training_metrics': training_metrics
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Анализ запазен в {analysis_file}")
        
        # Стартира adaptive learning cycle
        try:
            logger.info("🤖 Стартиране на adaptive learning cycle...")
            from pipelines.adaptive_trainer import AdaptiveTrainer
            
            adaptive_trainer = AdaptiveTrainer()
            adaptive_results = adaptive_trainer.adaptive_learning_cycle()
            
            if adaptive_results.get('enabled', False):
                summary = adaptive_results.get('summary', {})
                logger.info(
                    f"🤖 Adaptive learning завършен: "
                    f"{summary.get('total_retrained', 0)} retrained лиги, "
                    f"success rate: {summary.get('success_rate', 0):.1%}"
                )
                
                # Добавя adaptive резултатите към анализа
                analysis['adaptive_learning'] = adaptive_results
                
                # Обновява анализа файла
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'analysis': analysis,
                        'training_metrics': training_metrics,
                        'adaptive_results': adaptive_results
                    }, f, indent=2, ensure_ascii=False)
            else:
                logger.info("🔒 Adaptive learning е изключен")
                
        except Exception as e:
            logger.warning(f"⚠️ Грешка в adaptive learning: {e}")
        
        # Стартира ensemble weights optimization
        try:
            logger.info("🎯 Стартиране на ensemble weights optimization...")
            from pipelines.ensemble_optimizer import optimize_ensemble_weights
            
            ensemble_results = optimize_ensemble_weights()
            
            if ensemble_results.get('enabled', False):
                if ensemble_results.get('success', False):
                    if ensemble_results.get('weights_updated', False):
                        metrics = ensemble_results.get('metrics', {})
                        improvement = metrics.get('improvement', 0)
                        logger.info(
                            f"🎯 Ensemble optimization успешен: "
                            f"{improvement:.1%} подобрение в log_loss"
                        )
                    else:
                        logger.info("📊 Ensemble тегла не са променени (недостатъчно подобрение)")
                else:
                    error = ensemble_results.get('error', 'Unknown error')
                    logger.warning(f"⚠️ Ensemble optimization неуспешен: {error}")
                
                # Добавя ensemble резултатите към анализа
                analysis['ensemble_optimization'] = ensemble_results
                
                # Обновява анализа файла
                def convert_for_json(obj):
                    """Конвертира numpy типове за JSON serialization"""
                    import numpy as np
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    return obj
                
                analysis_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis,
                    'training_metrics': training_metrics,
                    'adaptive_results': adaptive_results if 'adaptive_results' in locals() else None,
                    'ensemble_results': ensemble_results
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_for_json(analysis_data), f, indent=2, ensure_ascii=False)
            else:
                logger.info("🔒 Ensemble optimization е изключена")
                
        except Exception as e:
            logger.warning(f"⚠️ Грешка в ensemble optimization: {e}")
        
        # Стартира advanced drift analysis
        try:
            logger.info("🔍 Стартиране на advanced drift analysis...")
            from pipelines.drift_analyzer import run_drift_analysis
            
            drift_results = run_drift_analysis()
            
            if drift_results.get('enabled', False):
                if drift_results.get('success', False):
                    drift_report = drift_results.get('drift_report', {})
                    overall_drift = drift_report.get('overall_drift', {})
                    
                    severity = overall_drift.get('severity', 'none')
                    score = overall_drift.get('score', 0)
                    detected = overall_drift.get('detected', False)
                    
                    if detected:
                        if severity == 'critical':
                            logger.critical(f"🚨 КРИТИЧЕН DRIFT открит: score={score:.3f}")
                        elif severity == 'high':
                            logger.warning(f"⚠️ ВИСОК DRIFT открит: score={score:.3f}")
                        elif severity == 'medium':
                            logger.info(f"📈 СРЕДЕН DRIFT открит: score={score:.3f}")
                        else:
                            logger.info(f"📊 НИСЪК DRIFT открит: score={score:.3f}")
                        
                        # Маркира high-risk зони
                        drift_types = drift_report.get('drift_types', {})
                        league_drift = drift_types.get('league_drift', {})
                        
                        if league_drift.get('detected', False):
                            league_details = league_drift.get('details', {}).get('leagues', {})
                            high_risk_leagues = [
                                league for league, info in league_details.items()
                                if info.get('drift_detected', False)
                            ]
                            
                            if high_risk_leagues:
                                logger.warning(f"🎯 High-risk лиги: {', '.join(high_risk_leagues)}")
                        
                        # Trigger adaptive learning при high drift
                        integration_config = drift_results.get('drift_report', {}).get('integration', {})
                        if (severity in ['high', 'critical'] and 
                            integration_config.get('trigger_adaptive_learning', True)):
                            logger.info("🔄 Triggering adaptive learning заради drift...")
                            
                            # Записва drift информация в adaptive learning history
                            try:
                                history_file = "logs/adaptive_learning_history.json"
                                
                                # Зарежда съществуващата история
                                if os.path.exists(history_file):
                                    with open(history_file, 'r', encoding='utf-8') as f:
                                        history_data = json.load(f)
                                        # Ако е dict, извлича историята
                                        if isinstance(history_data, dict):
                                            history = history_data.get('history', [])
                                        else:
                                            history = history_data if isinstance(history_data, list) else []
                                else:
                                    history = []
                                
                                # Добавя drift информация
                                drift_entry = {
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'drift_detection',
                                    'severity': severity,
                                    'drift_score': score,
                                    'high_risk_leagues': high_risk_leagues if 'high_risk_leagues' in locals() else [],
                                    'recommendations': drift_report.get('recommendations', [])
                                }
                                
                                history.append(drift_entry)
                                
                                # Запазва обновената история
                                with open(history_file, 'w', encoding='utf-8') as f:
                                    json.dump(history, f, indent=2, ensure_ascii=False)
                                
                                logger.info(f"💾 Drift информация записана в {history_file}")
                                
                            except Exception as hist_e:
                                logger.error(f"❌ Грешка при запис на drift история: {hist_e}")
                    else:
                        logger.info(f"✅ Няма значителен drift: score={score:.3f}")
                else:
                    error = drift_results.get('error', 'Unknown error')
                    logger.warning(f"⚠️ Drift analysis неуспешен: {error}")
                
                # Добавя drift резултатите към анализа
                analysis['drift_analysis'] = drift_results
                
                # Обновява анализа файла
                analysis_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis,
                    'training_metrics': training_metrics,
                    'adaptive_results': adaptive_results if 'adaptive_results' in locals() else None,
                    'ensemble_results': ensemble_results if 'ensemble_results' in locals() else None,
                    'drift_results': drift_results
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_for_json(analysis_data), f, indent=2, ensure_ascii=False)
            else:
                logger.info("🔒 Drift analysis е изключен")
                
        except Exception as e:
            logger.warning(f"⚠️ Грешка в drift analysis: {e}")
        
        # Стартира league ensemble optimization
        try:
            logger.info("🎯 Стартиране на league ensemble optimization...")
            from pipelines.league_ensemble_optimizer import run_league_ensemble_optimization
            
            league_ensemble_results = run_league_ensemble_optimization()
            
            if league_ensemble_results.get('enabled', False):
                if league_ensemble_results.get('success', False):
                    leagues_optimized = league_ensemble_results.get('leagues_optimized', 0)
                    leagues_updated = league_ensemble_results.get('leagues_updated', 0)
                    
                    if leagues_updated > 0:
                        success_rate = leagues_updated / leagues_optimized * 100 if leagues_optimized > 0 else 0
                        logger.info(
                            f"🎯 League ensemble optimization успешен: "
                            f"{leagues_updated}/{leagues_optimized} лиги обновени ({success_rate:.1f}%)"
                        )
                        
                        # Показва топ подобрения
                        league_results = league_ensemble_results.get('league_results', {})
                        if league_results:
                            top_improvements = sorted(
                                [(league, data['metrics']['improvement']) 
                                 for league, data in league_results.items()],
                                key=lambda x: x[1], reverse=True
                            )[:3]  # Топ 3
                            
                            logger.info("🏆 Топ подобрения:")
                            for league, improvement in top_improvements:
                                logger.info(f"   {league}: {improvement:.1%}")
                    else:
                        logger.info("📊 Няма лиги с достатъчно подобрение за обновяване")
                else:
                    error = league_ensemble_results.get('error', 'Unknown error')
                    logger.warning(f"⚠️ League ensemble optimization неуспешен: {error}")
                
                # Добавя league ensemble резултатите към анализа
                analysis['league_ensemble_optimization'] = league_ensemble_results
                
                # Обновява анализа файла
                analysis_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis,
                    'training_metrics': training_metrics,
                    'adaptive_results': adaptive_results if 'adaptive_results' in locals() else None,
                    'ensemble_results': ensemble_results if 'ensemble_results' in locals() else None,
                    'drift_results': drift_results if 'drift_results' in locals() else None,
                    'league_ensemble_results': league_ensemble_results
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_for_json(analysis_data), f, indent=2, ensure_ascii=False)
            else:
                logger.info("🔒 League ensemble optimization е изключена")
                
        except Exception as e:
            logger.warning(f"⚠️ Грешка в league ensemble optimization: {e}")
        
        logger.info("✅ Performance мониторинг завършен успешно")
        
    except Exception as e:
        logger.error(f"❌ Грешка при performance мониторинг: {e}")
        raise


if __name__ == "__main__":
    main()
