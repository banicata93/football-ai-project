#!/usr/bin/env python3
"""
A/B тестване на per-league vs global OU2.5 модели
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from core.league_utils import LEAGUE_ID_TO_SLUG, get_league_display_name
from core.utils import setup_logging


class PerLeagueABTester:
    """
    A/B тестване на per-league vs global модели
    """
    
    def __init__(self, api_url: str = "http://localhost:3000"):
        self.api_url = api_url
        self.logger = setup_logging()
        self.results = []
        
    def test_league_models(self, test_matches: pd.DataFrame, max_tests: int = 50) -> Dict:
        """
        Тества per-league модели срещу глобален модел
        
        Args:
            test_matches: Test данни
            max_tests: Максимален брой тестове на лига
        
        Returns:
            Резултати от A/B теста
        """
        self.logger.info("🧪 Стартиране на A/B тестване...")
        
        results_by_league = {}
        
        # Групира по лиги
        for league_id, league_data in test_matches.groupby('league_id'):
            if league_id not in LEAGUE_ID_TO_SLUG:
                continue
                
            league_slug = LEAGUE_ID_TO_SLUG[league_id]
            league_name = get_league_display_name(league_slug)
            
            self.logger.info(f"🏆 Тестване на {league_name} ({len(league_data)} мача)")
            
            # Ограничава броя тестове
            test_sample = league_data.sample(min(max_tests, len(league_data)), random_state=42)
            
            league_results = []
            
            for idx, match in test_sample.iterrows():
                try:
                    # Симулира мач с фиктивни отбори
                    home_team = f"Team_{match['home_team_id']}"
                    away_team = f"Team_{match['away_team_id']}"
                    
                    # API заявка
                    response = requests.post(
                        f"{self.api_url}/predict",
                        json={
                            "home_team": home_team,
                            "away_team": away_team,
                            "league": league_name
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        
                        # Извлича OU2.5 данни
                        ou25_pred = prediction.get('prediction_ou25', {})
                        model_source = prediction.get('model_sources', {}).get('ou25', 'unknown')
                        
                        # Реален резултат
                        actual_over = match['over_25']
                        predicted_prob = ou25_pred.get('prob_over', 0.5)
                        predicted_outcome = 1 if predicted_prob > 0.5 else 0
                        
                        # Метрики
                        correct = (predicted_outcome == actual_over)
                        log_loss = -(actual_over * np.log(max(predicted_prob, 1e-15)) + 
                                   (1 - actual_over) * np.log(max(1 - predicted_prob, 1e-15)))
                        brier_score = (predicted_prob - actual_over) ** 2
                        
                        league_results.append({
                            'match_id': match.get('match_id', idx),
                            'home_team_id': match['home_team_id'],
                            'away_team_id': match['away_team_id'],
                            'actual_over': actual_over,
                            'predicted_prob': predicted_prob,
                            'predicted_outcome': predicted_outcome,
                            'correct': correct,
                            'log_loss': log_loss,
                            'brier_score': brier_score,
                            'model_source': model_source,
                            'league_id': league_id,
                            'league_slug': league_slug,
                            'league_name': league_name
                        })
                        
                    else:
                        self.logger.warning(f"API грешка за мач {idx}: {response.status_code}")
                        
                except Exception as e:
                    self.logger.warning(f"Грешка при тестване на мач {idx}: {e}")
                    
                # Rate limiting
                time.sleep(0.1)
            
            if league_results:
                results_by_league[league_slug] = league_results
                self.logger.info(f"✅ {league_name}: {len(league_results)} успешни теста")
            else:
                self.logger.warning(f"❌ {league_name}: Няма успешни тестове")
        
        return results_by_league
    
    def analyze_results(self, results_by_league: Dict) -> Dict:
        """
        Анализира резултатите от A/B теста
        
        Args:
            results_by_league: Резултати по лиги
        
        Returns:
            Анализ на резултатите
        """
        self.logger.info("📊 Анализиране на A/B тест резултати...")
        
        analysis = {
            'league_performance': {},
            'model_comparison': {
                'league_ou25': {'accuracy': [], 'log_loss': [], 'brier_score': [], 'count': 0},
                'global_ou25': {'accuracy': [], 'log_loss': [], 'brier_score': [], 'count': 0}
            },
            'summary': {}
        }
        
        all_results = []
        
        for league_slug, league_results in results_by_league.items():
            league_name = get_league_display_name(league_slug)
            
            # Конвертира в DataFrame
            df = pd.DataFrame(league_results)
            all_results.extend(league_results)
            
            # Анализ по модел source
            league_model_results = df[df['model_source'] == 'league_ou25']
            global_model_results = df[df['model_source'] == 'global_ou25']
            
            league_perf = {
                'total_tests': len(df),
                'league_model_tests': len(league_model_results),
                'global_model_tests': len(global_model_results),
                'overall_accuracy': df['correct'].mean(),
                'overall_log_loss': df['log_loss'].mean(),
                'overall_brier_score': df['brier_score'].mean()
            }
            
            if len(league_model_results) > 0:
                league_perf['league_model_accuracy'] = league_model_results['correct'].mean()
                league_perf['league_model_log_loss'] = league_model_results['log_loss'].mean()
                league_perf['league_model_brier_score'] = league_model_results['brier_score'].mean()
                
                # Добавя към общия анализ
                analysis['model_comparison']['league_ou25']['accuracy'].extend(league_model_results['correct'].tolist())
                analysis['model_comparison']['league_ou25']['log_loss'].extend(league_model_results['log_loss'].tolist())
                analysis['model_comparison']['league_ou25']['brier_score'].extend(league_model_results['brier_score'].tolist())
                analysis['model_comparison']['league_ou25']['count'] += len(league_model_results)
            
            if len(global_model_results) > 0:
                league_perf['global_model_accuracy'] = global_model_results['correct'].mean()
                league_perf['global_model_log_loss'] = global_model_results['log_loss'].mean()
                league_perf['global_model_brier_score'] = global_model_results['brier_score'].mean()
                
                # Добавя към общия анализ
                analysis['model_comparison']['global_ou25']['accuracy'].extend(global_model_results['correct'].tolist())
                analysis['model_comparison']['global_ou25']['log_loss'].extend(global_model_results['log_loss'].tolist())
                analysis['model_comparison']['global_ou25']['brier_score'].extend(global_model_results['brier_score'].tolist())
                analysis['model_comparison']['global_ou25']['count'] += len(global_model_results)
            
            analysis['league_performance'][league_slug] = league_perf
        
        # Общ summary
        if all_results:
            all_df = pd.DataFrame(all_results)
            
            analysis['summary'] = {
                'total_tests': len(all_df),
                'leagues_tested': len(results_by_league),
                'league_model_usage': (all_df['model_source'] == 'league_ou25').sum(),
                'global_model_usage': (all_df['model_source'] == 'global_ou25').sum(),
                'overall_accuracy': all_df['correct'].mean(),
                'overall_log_loss': all_df['log_loss'].mean(),
                'overall_brier_score': all_df['brier_score'].mean()
            }
            
            # Сравнение между моделите
            for model_type in ['league_ou25', 'global_ou25']:
                model_data = analysis['model_comparison'][model_type]
                if model_data['count'] > 0:
                    model_data['avg_accuracy'] = np.mean(model_data['accuracy'])
                    model_data['avg_log_loss'] = np.mean(model_data['log_loss'])
                    model_data['avg_brier_score'] = np.mean(model_data['brier_score'])
        
        return analysis
    
    def generate_report(self, analysis: Dict, output_file: str = None) -> str:
        """
        Генерира отчет от A/B теста
        
        Args:
            analysis: Анализ на резултатите
            output_file: Файл за запазване
        
        Returns:
            Текстов отчет
        """
        report_lines = [
            "🧪 A/B ТЕСТ: PER-LEAGUE vs GLOBAL OU2.5 МОДЕЛИ",
            "=" * 60,
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary
        summary = analysis.get('summary', {})
        if summary:
            report_lines.extend([
                "📊 ОБЩ ПРЕГЛЕД:",
                f"   Общо тестове: {summary['total_tests']}",
                f"   Тествани лиги: {summary['leagues_tested']}",
                f"   League модел използван: {summary['league_model_usage']} пъти",
                f"   Global модел използван: {summary['global_model_usage']} пъти",
                f"   Общ accuracy: {summary['overall_accuracy']:.3f}",
                f"   Общ log loss: {summary['overall_log_loss']:.3f}",
                f"   Общ Brier score: {summary['overall_brier_score']:.3f}",
                ""
            ])
        
        # Model comparison
        model_comp = analysis.get('model_comparison', {})
        if model_comp:
            report_lines.extend([
                "⚖️ СРАВНЕНИЕ НА МОДЕЛИТЕ:",
                ""
            ])
            
            for model_type, data in model_comp.items():
                if data['count'] > 0:
                    model_name = "League-specific" if model_type == 'league_ou25' else "Global"
                    report_lines.extend([
                        f"🔹 {model_name} модел ({data['count']} теста):",
                        f"   Accuracy: {data['avg_accuracy']:.3f}",
                        f"   Log Loss: {data['avg_log_loss']:.3f}",
                        f"   Brier Score: {data['avg_brier_score']:.3f}",
                        ""
                    ])
            
            # Сравнение
            if (model_comp['league_ou25']['count'] > 0 and 
                model_comp['global_ou25']['count'] > 0):
                
                acc_diff = model_comp['league_ou25']['avg_accuracy'] - model_comp['global_ou25']['avg_accuracy']
                ll_diff = model_comp['league_ou25']['avg_log_loss'] - model_comp['global_ou25']['avg_log_loss']
                bs_diff = model_comp['league_ou25']['avg_brier_score'] - model_comp['global_ou25']['avg_brier_score']
                
                report_lines.extend([
                    "🎯 РАЗЛИКА (League - Global):",
                    f"   Accuracy: {acc_diff:+.3f} ({'подобрение' if acc_diff > 0 else 'влошение'})",
                    f"   Log Loss: {ll_diff:+.3f} ({'влошение' if ll_diff > 0 else 'подобрение'})",
                    f"   Brier Score: {bs_diff:+.3f} ({'влошение' if bs_diff > 0 else 'подобрение'})",
                    ""
                ])
        
        # League performance
        league_perf = analysis.get('league_performance', {})
        if league_perf:
            report_lines.extend([
                "🏆 PERFORMANCE ПО ЛИГИ:",
                ""
            ])
            
            for league_slug, perf in league_perf.items():
                league_name = get_league_display_name(league_slug)
                report_lines.extend([
                    f"🔸 {league_name}:",
                    f"   Общо тестове: {perf['total_tests']}",
                    f"   League модел: {perf['league_model_tests']} теста",
                    f"   Global модел: {perf['global_model_tests']} теста",
                    f"   Общ accuracy: {perf['overall_accuracy']:.3f}",
                    f"   Общ log loss: {perf['overall_log_loss']:.3f}",
                    f"   Общ Brier score: {perf['overall_brier_score']:.3f}",
                    ""
                ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"📄 Отчет запазен в {output_file}")
        
        return report


def main():
    """Главна функция за A/B тестване"""
    logger = setup_logging()
    
    logger.info("🧪 СТАРТИРАНЕ НА A/B ТЕСТВАНЕ")
    logger.info("=" * 50)
    
    try:
        # Зарежда test данни
        logger.info("📂 Зареждане на test данни...")
        test_df = pd.read_parquet("data/processed/test_poisson_predictions.parquet")
        
        # Добавя league_slug
        test_df['league_slug'] = test_df['league_id'].map(LEAGUE_ID_TO_SLUG)
        
        # Филтрира само лигите с тренирани модели
        trained_leagues = list(LEAGUE_ID_TO_SLUG.keys())
        test_df_filtered = test_df[test_df['league_id'].isin(trained_leagues)]
        
        logger.info(f"📊 Test данни: {len(test_df_filtered)} мача от {len(trained_leagues)} лиги")
        
        # Инициализира A/B tester
        tester = PerLeagueABTester()
        
        # Стартира тестването
        results = tester.test_league_models(test_df_filtered, max_tests=20)  # 20 теста на лига
        
        if results:
            # Анализира резултатите
            analysis = tester.analyze_results(results)
            
            # Генерира отчет
            report = tester.generate_report(
                analysis, 
                output_file="reports/ab_test_per_league_results.txt"
            )
            
            print(report)
            
            # Запазва резултатите
            results_file = "reports/ab_test_per_league_data.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results_by_league': results,
                    'analysis': analysis
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Данни запазени в {results_file}")
            
        else:
            logger.error("❌ Няма резултати от A/B теста")
    
    except Exception as e:
        logger.error(f"❌ Грешка при A/B тестване: {e}")
        raise


if __name__ == "__main__":
    main()
