"""
Adaptive Tuning система за автоматично коригиране на вероятностните параметри
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from monitoring.calibration_metrics import CalibrationMonitor, evaluate_calibration, evaluate_multiclass_calibration


class AdaptiveTuner:
    """
    Система за автоматично адаптиране на модела базирано на калибрационни метрики
    """
    
    def __init__(self, 
                 config_dir: str = "models/config",
                 tuning_log_file: str = "models/config/tuning_log.json"):
        self.config_dir = config_dir
        self.tuning_log_file = tuning_log_file
        
        # Създава директории
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(os.path.dirname(tuning_log_file), exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Калибрационен мониторинг
        self.calibration_monitor = CalibrationMonitor()
        
        # Thresholds за автоматични корекции
        self.thresholds = {
            'ece_threshold': 0.07,      # ECE > 0.07 → корекция
            'brier_threshold': 0.20,    # Brier > 0.20 → корекция
            'min_samples': 100,         # Минимум samples за корекция
            'max_corrections_per_day': 3  # Максимум корекции на ден
        }
        
        # Текущи параметри (ще се зареждат от конфигурация)
        self.current_params = self._load_current_params()
    
    def _load_current_params(self) -> Dict[str, Any]:
        """
        Зарежда текущите параметри от конфигурационни файлове
        
        Returns:
            Dictionary с текущите параметри
        """
        params_file = os.path.join(self.config_dir, "adaptive_params.json")
        
        # Default параметри
        default_params = {
            'soft_clipping': {
                'lo': 0.02,
                'hi': 0.98
            },
            'ensemble_weights': {
                'poisson': 0.3,
                'ml': 0.5,
                'elo': 0.2
            },
            'poisson_params': {
                'shrinkage_alpha': 0.2,
                'home_advantage_multiplier': 1.0  # Multiplier за league home advantages
            },
            'btts_calibration': {
                'scaling_factor': 0.85,
                'poisson_blend': 0.2
            },
            'confidence_params': {
                'entropy_weight': 0.6,
                'agreement_weight': 0.4
            }
        }
        
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    loaded_params = json.load(f)
                # Merge с defaults
                for key, value in default_params.items():
                    if key not in loaded_params:
                        loaded_params[key] = value
                return loaded_params
            except Exception as e:
                self.logger.warning(f"Failed to load params, using defaults: {e}")
        
        return default_params
    
    def _save_current_params(self):
        """
        Запазва текущите параметри в конфигурационен файл
        """
        params_file = os.path.join(self.config_dir, "adaptive_params.json")
        
        try:
            with open(params_file, 'w') as f:
                json.dump(self.current_params, f, indent=2)
            self.logger.info(f"Saved parameters to {params_file}")
        except Exception as e:
            self.logger.error(f"Failed to save parameters: {e}")
    
    def _log_tuning_action(self, action: str, reason: str, old_params: Dict, new_params: Dict):
        """
        Логва tuning действие
        
        Args:
            action: Тип на действието
            reason: Причина за действието
            old_params: Стари параметри
            new_params: Нови параметри
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'reason': reason,
            'old_params': old_params,
            'new_params': new_params,
            'changes': self._calculate_param_changes(old_params, new_params)
        }
        
        # Зарежда съществуващия лог
        tuning_log = []
        if os.path.exists(self.tuning_log_file):
            try:
                with open(self.tuning_log_file, 'r') as f:
                    tuning_log = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load tuning log: {e}")
        
        # Добавя новия запис
        tuning_log.append(log_entry)
        
        # Запазва обновения лог
        try:
            with open(self.tuning_log_file, 'w') as f:
                json.dump(tuning_log, f, indent=2)
            self.logger.info(f"Logged tuning action: {action}")
        except Exception as e:
            self.logger.error(f"Failed to save tuning log: {e}")
    
    def _calculate_param_changes(self, old_params: Dict, new_params: Dict) -> Dict:
        """
        Изчислява промените между стари и нови параметри
        
        Args:
            old_params: Стари параметри
            new_params: Нови параметри
        
        Returns:
            Dictionary с промените
        """
        changes = {}
        
        for key in new_params:
            if key in old_params:
                if isinstance(new_params[key], dict) and isinstance(old_params[key], dict):
                    # Рекурсивно за nested dictionaries
                    nested_changes = self._calculate_param_changes(old_params[key], new_params[key])
                    if nested_changes:
                        changes[key] = nested_changes
                elif new_params[key] != old_params[key]:
                    changes[key] = {
                        'old': old_params[key],
                        'new': new_params[key],
                        'change': new_params[key] - old_params[key] if isinstance(new_params[key], (int, float)) else 'changed'
                    }
        
        return changes
    
    def _check_recent_corrections(self) -> int:
        """
        Проверява броя корекции за последните 24 часа
        
        Returns:
            Брой корекции за последните 24 часа
        """
        if not os.path.exists(self.tuning_log_file):
            return 0
        
        try:
            with open(self.tuning_log_file, 'r') as f:
                tuning_log = json.load(f)
            
            # Филтрира последните 24 часа
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_corrections = 0
            
            for entry in tuning_log:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time and entry['action'] == 'auto_correction':
                    recent_corrections += 1
            
            return recent_corrections
            
        except Exception as e:
            self.logger.warning(f"Failed to check recent corrections: {e}")
            return 0
    
    def analyze_calibration_drift(self, days: int = 7) -> Dict[str, Any]:
        """
        Анализира calibration drift за последните дни
        
        Args:
            days: Брой дни за анализ
        
        Returns:
            Dictionary с анализа
        """
        report = self.calibration_monitor.generate_calibration_report(days)
        
        if 'error' in report:
            return report
        
        analysis = {
            'period': f'Last {days} days',
            'n_matches': report['n_matches'],
            'issues_detected': [],
            'recommendations': []
        }
        
        # Анализира всеки пазар
        markets = ['ou25', 'btts']
        for market in markets:
            if market in report:
                metrics = report[market]
                
                # Проверява ECE
                if metrics['ece'] > self.thresholds['ece_threshold']:
                    analysis['issues_detected'].append({
                        'market': market,
                        'issue': 'high_ece',
                        'value': metrics['ece'],
                        'threshold': self.thresholds['ece_threshold']
                    })
                
                # Проверява Brier Score
                if metrics['brier_score'] > self.thresholds['brier_threshold']:
                    analysis['issues_detected'].append({
                        'market': market,
                        'issue': 'high_brier',
                        'value': metrics['brier_score'],
                        'threshold': self.thresholds['brier_threshold']
                    })
        
        # Анализира 1X2 (overall метрики)
        if '1x2' in report and 'overall' in report['1x2']:
            metrics = report['1x2']['overall']
            
            if metrics['brier_score'] > self.thresholds['brier_threshold']:
                analysis['issues_detected'].append({
                    'market': '1x2',
                    'issue': 'high_brier',
                    'value': metrics['brier_score'],
                    'threshold': self.thresholds['brier_threshold']
                })
        
        # Генерира препоръки
        analysis['recommendations'] = self._generate_recommendations(analysis['issues_detected'])
        analysis['timestamp'] = datetime.now().isoformat()
        
        return analysis
    
    def _generate_recommendations(self, issues: list) -> list:
        """
        Генерира препоръки за корекции базирано на проблемите
        
        Args:
            issues: Списък с открити проблеми
        
        Returns:
            Списък с препоръки
        """
        recommendations = []
        
        for issue in issues:
            market = issue['market']
            issue_type = issue['issue']
            
            if issue_type == 'high_ece':
                if market in ['ou25', 'btts']:
                    recommendations.append({
                        'action': 'tighten_clipping',
                        'market': market,
                        'description': f'Намали soft clipping границите за {market} с 0.01',
                        'params': {
                            'soft_clipping': {
                                'lo': max(0.01, self.current_params['soft_clipping']['lo'] - 0.01),
                                'hi': min(0.99, self.current_params['soft_clipping']['hi'] + 0.01)
                            }
                        }
                    })
                elif market == '1x2':
                    recommendations.append({
                        'action': 'increase_poisson_weight',
                        'market': market,
                        'description': 'Увеличи Poisson weight в ensemble с 0.05',
                        'params': {
                            'ensemble_weights': {
                                'poisson': min(0.6, self.current_params['ensemble_weights']['poisson'] + 0.05),
                                'ml': max(0.2, self.current_params['ensemble_weights']['ml'] - 0.05)
                            }
                        }
                    })
            
            elif issue_type == 'high_brier':
                recommendations.append({
                    'action': 'adjust_calibration',
                    'market': market,
                    'description': f'Подобри калибрацията за {market}',
                    'params': {
                        'btts_calibration': {
                            'scaling_factor': max(0.7, self.current_params['btts_calibration']['scaling_factor'] - 0.05)
                        }
                    }
                })
        
        return recommendations
    
    def apply_automatic_corrections(self, analysis: Dict[str, Any]) -> bool:
        """
        Прилага автоматични корекции базирано на анализа
        
        Args:
            analysis: Calibration drift анализ
        
        Returns:
            True ако корекциите са приложени успешно
        """
        # Проверява дали има достатъчно данни
        if analysis['n_matches'] < self.thresholds['min_samples']:
            self.logger.info(f"Not enough samples for correction: {analysis['n_matches']} < {self.thresholds['min_samples']}")
            return False
        
        # Проверява дали не са направени твърде много корекции наскоро
        recent_corrections = self._check_recent_corrections()
        if recent_corrections >= self.thresholds['max_corrections_per_day']:
            self.logger.info(f"Too many recent corrections: {recent_corrections} >= {self.thresholds['max_corrections_per_day']}")
            return False
        
        # Проверява дали има проблеми за корекция
        if not analysis['issues_detected']:
            self.logger.info("No calibration issues detected, no corrections needed")
            return False
        
        # Запазва старите параметри
        old_params = self.current_params.copy()
        
        # Прилага препоръките
        corrections_applied = 0
        for recommendation in analysis['recommendations']:
            if corrections_applied >= 2:  # Максимум 2 корекции наведнъж
                break
            
            action = recommendation['action']
            params = recommendation['params']
            
            # Прилага параметрите
            for param_group, param_values in params.items():
                if param_group in self.current_params:
                    for param_name, param_value in param_values.items():
                        if param_name in self.current_params[param_group]:
                            self.current_params[param_group][param_name] = param_value
                            corrections_applied += 1
        
        if corrections_applied > 0:
            # Запазва новите параметри
            self._save_current_params()
            
            # Логва действието
            self._log_tuning_action(
                action='auto_correction',
                reason=f"Calibration drift detected: {len(analysis['issues_detected'])} issues",
                old_params=old_params,
                new_params=self.current_params
            )
            
            self.logger.info(f"Applied {corrections_applied} automatic corrections")
            return True
        
        return False
    
    def run_daily_monitoring(self) -> Dict[str, Any]:
        """
        Изпълнява дневния мониторинг и автоматични корекции
        
        Returns:
            Dictionary с резултатите от мониторинга
        """
        self.logger.info("Starting daily calibration monitoring...")
        
        # Анализира calibration drift
        analysis = self.analyze_calibration_drift(days=7)
        
        if 'error' in analysis:
            return {
                'status': 'error',
                'message': analysis['error'],
                'timestamp': datetime.now().isoformat()
            }
        
        # Прилага автоматични корекции ако е необходимо
        corrections_applied = self.apply_automatic_corrections(analysis)
        
        # Подготвя резултата
        result = {
            'status': 'completed',
            'analysis': analysis,
            'corrections_applied': corrections_applied,
            'current_params': self.current_params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Запазва дневния отчет
        daily_report_file = f"reports/calibration/daily_monitoring_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(daily_report_file), exist_ok=True)
        
        try:
            with open(daily_report_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Saved daily monitoring report: {daily_report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save daily report: {e}")
        
        return result
    
    def get_tuning_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Получава историята на tuning действията
        
        Args:
            days: Брой дни назад
        
        Returns:
            Dictionary с историята
        """
        if not os.path.exists(self.tuning_log_file):
            return {'history': [], 'summary': {'total_actions': 0}}
        
        try:
            with open(self.tuning_log_file, 'r') as f:
                tuning_log = json.load(f)
            
            # Филтрира по дата
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_log = []
            
            for entry in tuning_log:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_date:
                    filtered_log.append(entry)
            
            # Създава summary
            summary = {
                'total_actions': len(filtered_log),
                'auto_corrections': len([e for e in filtered_log if e['action'] == 'auto_correction']),
                'manual_adjustments': len([e for e in filtered_log if e['action'] == 'manual_adjustment']),
                'period': f'Last {days} days'
            }
            
            return {
                'history': filtered_log,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get tuning history: {e}'}


if __name__ == "__main__":
    # Тестване на AdaptiveTuner
    print("🧪 Тестване на AdaptiveTuner...")
    
    tuner = AdaptiveTuner(
        config_dir="test_config",
        tuning_log_file="test_config/tuning_log.json"
    )
    
    print("✅ AdaptiveTuner инициализиран")
    print(f"Current params: {tuner.current_params}")
    
    # Симулиран анализ
    mock_analysis = {
        'n_matches': 500,
        'issues_detected': [
            {
                'market': 'ou25',
                'issue': 'high_ece',
                'value': 0.08,
                'threshold': 0.07
            }
        ],
        'recommendations': []
    }
    
    # Генерира препоръки
    recommendations = tuner._generate_recommendations(mock_analysis['issues_detected'])
    print(f"✅ Generated recommendations: {len(recommendations)}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_config"):
        shutil.rmtree("test_config")
    
    print("✅ AdaptiveTuner работи!")
