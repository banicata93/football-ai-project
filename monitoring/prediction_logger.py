"""
Автоматично логване на прогнози за калибрационен мониторинг
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Any, Optional
import logging
from pathlib import Path


class PredictionLogger:
    """
    Клас за логване на прогнози и реални резултати
    """
    
    def __init__(self, 
                 history_file: str = "logs/predictions_history.parquet",
                 results_file: str = "logs/match_results.parquet"):
        self.history_file = history_file
        self.results_file = results_file
        
        # Създава директории
        history_dir = os.path.dirname(history_file)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        
        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, 
                      home_team: str,
                      away_team: str,
                      league: str,
                      prediction_data: Dict[str, Any],
                      match_date: Optional[str] = None) -> str:
        """
        Логва прогноза за мач
        
        Args:
            home_team: Име на домакина
            away_team: Име на гостите
            league: Лига
            prediction_data: Пълни данни от прогнозата
            match_date: Дата на мача (ако е различна от днес)
        
        Returns:
            Unique ID на прогнозата
        """
        # Генерира unique ID
        prediction_id = f"{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Подготвя данните за логване
        log_entry = {
            'prediction_id': prediction_id,
            'prediction_date': datetime.now().isoformat(),
            'match_date': match_date or datetime.now().strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            
            # 1X2 Predictions
            'pred_1x2_probs': [
                prediction_data['prediction_1x2']['prob_home_win'],
                prediction_data['prediction_1x2']['prob_draw'],
                prediction_data['prediction_1x2']['prob_away_win']
            ],
            'pred_1x2_outcome': prediction_data['prediction_1x2']['predicted_outcome'],
            'pred_1x2_confidence': prediction_data['prediction_1x2']['confidence'],
            
            # OU2.5 Predictions
            'pred_ou25_prob': prediction_data['prediction_ou25']['prob_over'],
            'pred_ou25_outcome': prediction_data['prediction_ou25']['predicted_outcome'],
            'pred_ou25_confidence': prediction_data['prediction_ou25']['confidence'],
            
            # BTTS Predictions
            'pred_btts_prob': prediction_data['prediction_btts']['prob_yes'],
            'pred_btts_outcome': prediction_data['prediction_btts']['predicted_outcome'],
            'pred_btts_confidence': prediction_data['prediction_btts']['confidence'],
            
            # FII Score
            'fii_score': prediction_data['fii']['score'],
            'fii_confidence_level': prediction_data['fii']['confidence_level'],
            
            # Poisson Analysis (ако е налично)
            'poisson_lambda_home': prediction_data.get('poisson_analysis', {}).get('lambda_home'),
            'poisson_lambda_away': prediction_data.get('poisson_analysis', {}).get('lambda_away'),
            'poisson_expected_goals': prediction_data.get('poisson_analysis', {}).get('expected_goals'),
            
            # Actual results (ще се попълнят по-късно)
            'actual_home_score': None,
            'actual_away_score': None,
            'result_updated_date': None,
            
            # Metadata (flatten для Parquet compatibility)
            'model_version_poisson': prediction_data.get('model_versions', {}).get('poisson', 'v1'),
            'model_version_1x2': prediction_data.get('model_versions', {}).get('1x2', 'v1'),
            'model_version_ou25': prediction_data.get('model_versions', {}).get('ou25', 'v1'),
            'model_version_btts': prediction_data.get('model_versions', {}).get('btts', 'v1'),
            'model_version_ensemble': prediction_data.get('model_versions', {}).get('ensemble', 'v1'),
            'api_version': prediction_data.get('api_version', 'v1')
        }
        
        # Конвертира в DataFrame
        df_new = pd.DataFrame([log_entry])
        
        # Добавя към историята
        if os.path.exists(self.history_file):
            try:
                df_existing = pd.read_parquet(self.history_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as e:
                self.logger.warning(f"Could not read existing history: {e}")
                df_combined = df_new
        else:
            df_combined = df_new
        
        # Запазва
        try:
            df_combined.to_parquet(self.history_file, index=False)
            self.logger.info(f"Logged prediction {prediction_id}")
        except Exception as e:
            self.logger.error(f"Failed to save prediction log: {e}")
        
        return prediction_id
    
    def update_match_result(self, 
                           home_team: str,
                           away_team: str,
                           home_score: int,
                           away_score: int,
                           match_date: str) -> bool:
        """
        Обновява реалния резултат за мач
        
        Args:
            home_team: Име на домакина
            away_team: Име на гостите
            home_score: Голове на домакина
            away_score: Голове на гостите
            match_date: Дата на мача
        
        Returns:
            True ако обновяването е успешно
        """
        if not os.path.exists(self.history_file):
            self.logger.warning("No prediction history found")
            return False
        
        try:
            df = pd.read_parquet(self.history_file)
            
            # Намира съответстващите прогнози
            mask = (
                (df['home_team'] == home_team) & 
                (df['away_team'] == away_team) & 
                (df['match_date'] == match_date) &
                (df['actual_home_score'].isna())  # Само необновени
            )
            
            if mask.sum() == 0:
                self.logger.warning(f"No matching predictions found for {home_team} vs {away_team} on {match_date}")
                return False
            
            # Обновява резултатите
            df.loc[mask, 'actual_home_score'] = home_score
            df.loc[mask, 'actual_away_score'] = away_score
            df.loc[mask, 'result_updated_date'] = datetime.now().isoformat()
            
            # Запазва
            df.to_parquet(self.history_file, index=False)
            
            updated_count = mask.sum()
            self.logger.info(f"Updated {updated_count} predictions for {home_team} vs {away_team}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update match result: {e}")
            return False
    
    def bulk_update_results(self, results_df: pd.DataFrame) -> int:
        """
        Bulk обновяване на резултати от DataFrame
        
        Args:
            results_df: DataFrame с колони: home_team, away_team, match_date, home_score, away_score
        
        Returns:
            Брой обновени записи
        """
        updated_count = 0
        
        for _, row in results_df.iterrows():
            success = self.update_match_result(
                row['home_team'],
                row['away_team'],
                row['home_score'],
                row['away_score'],
                row['match_date']
            )
            if success:
                updated_count += 1
        
        return updated_count
    
    def get_pending_matches(self, days_back: int = 7) -> pd.DataFrame:
        """
        Получава мачове без резултати от последните дни
        
        Args:
            days_back: Брой дни назад за търсене
        
        Returns:
            DataFrame с мачове без резултати
        """
        if not os.path.exists(self.history_file):
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(self.history_file)
            
            # Филтрира по дата
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df['match_date'] = pd.to_datetime(df['match_date'])
            df_recent = df[df['match_date'] >= cutoff_date].copy()
            
            # Намира мачове без резултати
            pending = df_recent[df_recent['actual_home_score'].isna()].copy()
            
            return pending[['prediction_id', 'home_team', 'away_team', 'match_date', 'league']]
            
        except Exception as e:
            self.logger.error(f"Failed to get pending matches: {e}")
            return pd.DataFrame()
    
    def cleanup_old_predictions(self, days_to_keep: int = 365):
        """
        Изчиства стари прогнози за да не се натрупват твърде много данни
        
        Args:
            days_to_keep: Брой дни за запазване
        """
        if not os.path.exists(self.history_file):
            return
        
        try:
            df = pd.read_parquet(self.history_file)
            
            # Изчислява cutoff дата
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
            
            # Запазва само скорошните
            df_recent = df[df['prediction_date'] >= cutoff_date].copy()
            
            # Запазва
            df_recent.to_parquet(self.history_file, index=False)
            
            removed_count = len(df) - len(df_recent)
            self.logger.info(f"Cleaned up {removed_count} old predictions")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old predictions: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получава статистики за логнатите прогнози
        
        Returns:
            Dictionary със статистики
        """
        if not os.path.exists(self.history_file):
            return {'error': 'No prediction history found'}
        
        try:
            df = pd.read_parquet(self.history_file)
            
            total_predictions = len(df)
            with_results = df['actual_home_score'].notna().sum()
            pending_results = total_predictions - with_results
            
            # Дата range
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
            date_range = {
                'earliest': df['prediction_date'].min().isoformat(),
                'latest': df['prediction_date'].max().isoformat()
            }
            
            # Leagues
            league_counts = df['league'].value_counts().to_dict()
            
            # Recent activity (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_predictions = df[df['prediction_date'] >= recent_cutoff]
            
            return {
                'total_predictions': int(total_predictions),
                'predictions_with_results': int(with_results),
                'pending_results': int(pending_results),
                'completion_rate': float(with_results / total_predictions) if total_predictions > 0 else 0,
                'date_range': date_range,
                'leagues': league_counts,
                'recent_activity': {
                    'last_7_days': len(recent_predictions),
                    'avg_per_day': len(recent_predictions) / 7
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get statistics: {e}'}


# Middleware за автоматично логване в API
def create_prediction_logging_middleware(logger: PredictionLogger):
    """
    Създава middleware за автоматично логване на API прогнози
    """
    def log_prediction_middleware(request_data: Dict, response_data: Dict):
        """
        Middleware функция за логване
        """
        try:
            # Извлича данни от request
            home_team = request_data.get('home_team')
            away_team = request_data.get('away_team')
            league = request_data.get('league', 'Unknown')
            match_date = request_data.get('date')
            
            if home_team and away_team and 'prediction_1x2' in response_data:
                logger.log_prediction(
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    prediction_data=response_data,
                    match_date=match_date
                )
        except Exception as e:
            logging.error(f"Failed to log prediction: {e}")
    
    return log_prediction_middleware


if __name__ == "__main__":
    # Тестване на PredictionLogger
    print("🧪 Тестване на PredictionLogger...")
    
    logger = PredictionLogger(
        history_file="test_predictions.parquet",
        results_file="test_results.parquet"
    )
    
    # Симулирана прогноза
    test_prediction = {
        'prediction_1x2': {
            'prob_home_win': 0.45,
            'prob_draw': 0.30,
            'prob_away_win': 0.25,
            'predicted_outcome': '1',
            'confidence': 0.45
        },
        'prediction_ou25': {
            'prob_over': 0.65,
            'prob_under': 0.35,
            'predicted_outcome': 'Over',
            'confidence': 0.65
        },
        'prediction_btts': {
            'prob_yes': 0.55,
            'prob_no': 0.45,
            'predicted_outcome': 'Yes',
            'confidence': 0.55
        },
        'fii': {
            'score': 7.2,
            'confidence_level': 'High'
        }
    }
    
    # Логва прогноза
    prediction_id = logger.log_prediction(
        home_team="Test Home",
        away_team="Test Away",
        league="Test League",
        prediction_data=test_prediction
    )
    
    print(f"✅ Logged prediction: {prediction_id}")
    
    # Обновява резултат
    success = logger.update_match_result(
        home_team="Test Home",
        away_team="Test Away",
        home_score=2,
        away_score=1,
        match_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    print(f"✅ Updated result: {success}")
    
    # Статистики
    stats = logger.get_statistics()
    print(f"✅ Statistics: {stats}")
    
    # Cleanup
    os.remove("test_predictions.parquet")
    print("✅ PredictionLogger работи!")
